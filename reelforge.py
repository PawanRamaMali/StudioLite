"""
ReelForge — AI-powered short video generation engine.
Integrated into StudioLite as a tool.
"""
import os
import sys
import json
import re
import base64
import time
import tempfile

import PIL.Image
if not hasattr(PIL.Image, "ANTIALIAS"):
    PIL.Image.ANTIALIAS = PIL.Image.LANCZOS

# Fix Windows encoding
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import requests
from uuid import uuid4

# Import from local mpv2 package (copied from MoneyPrinterV2/src)
from mpv2.config import (
    ROOT_DIR, assert_folder_structure,
    get_nanobanana2_api_key, get_nanobanana2_api_base_url, get_nanobanana2_model,
    get_nanobanana2_aspect_ratio, get_threads,
    get_fonts_dir, get_font, get_imagemagick_path,
    get_stt_provider, get_whisper_model, get_whisper_device, get_whisper_compute_type,
    equalize_subtitles,
)
from mpv2.utils import rem_temp_files, fetch_songs, choose_random_song
from mpv2.classes.Tts import TTS
from mpv2.llm_provider import select_model, generate_text, list_models, check_ollama_connection

from moviepy.editor import (
    ImageClip, AudioFileClip, concatenate_videoclips,
    CompositeVideoClip, CompositeAudioClip, TextClip, afx,
)
from moviepy.video.fx.all import crop
from moviepy.config import change_settings
from moviepy.video.tools.subtitles import SubtitlesClip

try:
    change_settings({"IMAGEMAGICK_BINARY": get_imagemagick_path()})
except Exception:
    pass

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------
CONFIG_PATH = os.path.join(ROOT_DIR, "config.json")


def load_config():
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)


def save_config(data):
    with open(CONFIG_PATH, "w") as f:
        json.dump(data, f, indent=2)


def list_local_models():
    d = os.path.join(ROOT_DIR, "models")
    if not os.path.isdir(d):
        return []
    return [f for f in os.listdir(d) if f.endswith(".safetensors")]


def list_ollama_models():
    try:
        return list_models()
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
_sdxl_pipe = None
_sdxl_loaded_model = None


def llm(prompt):
    return generate_text(prompt)


def rf_generate_script(subject, language, sentence_length):
    prompt = f"""Generate a script for a video in {sentence_length} sentences.

The script should be related to: {subject}
Language: {language}

Rules:
- Get straight to the point, no "welcome to this video"
- No markdown, no formatting, no title
- Only return the raw script text
- Each sentence should be short
- Write in {language}"""
    return re.sub(r"\*", "", llm(prompt))


def rf_generate_metadata(subject, script):
    title = llm(f"Generate a YouTube Video Title for: {subject}. Include hashtags. Under 100 characters. Only return the title.")
    description = llm(f"Generate a YouTube Video Description for this script: {script}. Only return the description.")
    return {"title": title, "description": description}


def rf_generate_prompts(subject, script):
    prompt = f"""Generate exactly 3 image prompts for: {subject}

Each prompt should describe a photorealistic or detailed digital art scene.
Include lighting, composition, style, and mood details.

Return ONLY a JSON array of 3 strings. Example:
["prompt 1", "prompt 2", "prompt 3"]

Script: {script}"""

    completion = str(llm(prompt)).replace("```json", "").replace("```", "").strip()

    try:
        result = json.loads(completion)
        if isinstance(result, dict):
            result = result.get("image_prompts", list(result.values())[0])
        if isinstance(result, list) and len(result) > 0:
            return result[:3]
    except Exception:
        pass

    r = re.compile(r"\[.*\]", re.DOTALL)
    matches = r.findall(completion)
    if matches:
        try:
            return json.loads(matches[0])[:3]
        except Exception:
            pass

    sentences = [s.strip() for s in script.split(".") if s.strip()]
    return [
        f"High quality digital illustration: {s}. Professional lighting, vivid colors, 4k, artstation"
        for s in sentences[:3]
    ]


def rf_generate_image_nanobanana2(prompt):
    api_key = get_nanobanana2_api_key()
    if not api_key:
        return None
    base_url = get_nanobanana2_api_base_url().rstrip("/")
    model = get_nanobanana2_model()
    aspect_ratio = get_nanobanana2_aspect_ratio()
    endpoint = f"{base_url}/models/{model}:generateContent"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"responseModalities": ["IMAGE"], "imageConfig": {"aspectRatio": aspect_ratio}},
    }
    try:
        resp = requests.post(endpoint, headers={"x-goog-api-key": api_key, "Content-Type": "application/json"}, json=payload, timeout=300)
        resp.raise_for_status()
        body = resp.json()
        for c in body.get("candidates", []):
            for p in c.get("content", {}).get("parts", []):
                inline = p.get("inlineData") or p.get("inline_data")
                if inline and inline.get("data"):
                    img_bytes = base64.b64decode(inline["data"])
                    path = os.path.join(ROOT_DIR, ".mp", f"{uuid4()}.png")
                    with open(path, "wb") as f:
                        f.write(img_bytes)
                    return path
    except Exception:
        pass
    return None


def rf_load_sdxl_pipeline(model_name=None):
    global _sdxl_pipe, _sdxl_loaded_model
    if _sdxl_pipe is not None and _sdxl_loaded_model == model_name:
        return

    from diffusers import StableDiffusionXLPipeline
    import torch

    if _sdxl_pipe is not None:
        del _sdxl_pipe
        _sdxl_pipe = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    models_dir = os.path.join(ROOT_DIR, "models")
    local_model = None

    if model_name and os.path.exists(os.path.join(models_dir, model_name)):
        local_model = os.path.join(models_dir, model_name)
    elif os.path.isdir(models_dir):
        for f in os.listdir(models_dir):
            if f.endswith(".safetensors"):
                local_model = os.path.join(models_dir, f)
                break

    if local_model:
        _sdxl_pipe = StableDiffusionXLPipeline.from_single_file(local_model, torch_dtype=torch.float16)
    else:
        from diffusers import AutoPipelineForText2Image
        _sdxl_pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")

    _sdxl_pipe = _sdxl_pipe.to("cuda")
    _sdxl_pipe.set_progress_bar_config(disable=True)
    _sdxl_loaded_model = model_name


def rf_generate_image_sdxl(prompt, model_name=None):
    rf_load_sdxl_pipeline(model_name)
    models_dir = os.path.join(ROOT_DIR, "models")
    has_local = model_name or (os.path.isdir(models_dir) and any(f.endswith(".safetensors") for f in os.listdir(models_dir)))

    try:
        if has_local:
            result = _sdxl_pipe(
                prompt=prompt,
                negative_prompt="blurry, low quality, deformed, ugly, bad anatomy, text, watermark",
                num_inference_steps=20, guidance_scale=7.0, width=768, height=1024,
            )
        else:
            result = _sdxl_pipe(prompt=prompt, num_inference_steps=4, guidance_scale=0.0, width=768, height=1024)
        img = result.images[0]
        path = os.path.join(ROOT_DIR, ".mp", f"{uuid4()}.png")
        img.save(path)
        return path
    except Exception as e:
        print(f"Image generation error: {e}")
        return None


def rf_generate_image(prompt, provider="sdxl_turbo", model_name=None):
    if provider == "nanobanana2":
        return rf_generate_image_nanobanana2(prompt)
    return rf_generate_image_sdxl(prompt, model_name)


def rf_generate_subtitles(audio_path):
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        return None

    model = WhisperModel(get_whisper_model(), device=get_whisper_device(), compute_type=get_whisper_compute_type())
    segments, _ = model.transcribe(audio_path, vad_filter=True)

    def fmt(s):
        ms = max(0, int(round(s * 1000)))
        return f"{ms // 3600000:02d}:{(ms % 3600000) // 60000:02d}:{(ms % 60000) // 1000:02d},{ms % 1000:03d}"

    lines = []
    for idx, seg in enumerate(segments, 1):
        t = str(seg.text).strip()
        if not t:
            continue
        lines += [str(idx), f"{fmt(seg.start)} --> {fmt(seg.end)}", t, ""]

    if not lines:
        return None
    srt_path = os.path.join(ROOT_DIR, ".mp", f"{uuid4()}.srt")
    with open(srt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return srt_path


def rf_combine_video(images, tts_path):
    # Use tempfile for the output to avoid path issues
    combined_path = os.path.join(tempfile.gettempdir(), f"reelforge_{uuid4()}.mp4")
    threads = get_threads()
    tts_clip = AudioFileClip(tts_path)
    max_duration = tts_clip.duration
    req_dur = max_duration / len(images)

    generator = lambda txt: TextClip(
        txt, font=os.path.join(get_fonts_dir(), get_font()),
        fontsize=100, color="#FFFF00", stroke_color="black", stroke_width=5,
        size=(1080, 1920), method="caption",
    )

    clips, tot_dur = [], 0
    while tot_dur < max_duration:
        for img_path in images:
            clip = ImageClip(img_path)
            clip.duration = req_dur
            clip = clip.set_fps(30)
            if round((clip.w / clip.h), 4) < 0.5625:
                clip = crop(clip, width=clip.w, height=round(clip.w / 0.5625), x_center=clip.w / 2, y_center=clip.h / 2)
            else:
                clip = crop(clip, width=round(0.5625 * clip.h), height=clip.h, x_center=clip.w / 2, y_center=clip.h / 2)
            clip = clip.resize((1080, 1920))
            clips.append(clip)
            tot_dur += clip.duration

    final_clip = concatenate_videoclips(clips).set_fps(30)
    song = choose_random_song()
    song_clip = AudioFileClip(song).set_fps(44100).fx(afx.volumex, 0.1)
    comp_audio = CompositeAudioClip([tts_clip.set_fps(44100), song_clip])
    final_clip = final_clip.set_audio(comp_audio).set_duration(tts_clip.duration)

    try:
        srt_path = rf_generate_subtitles(tts_path)
        if srt_path:
            equalize_subtitles(srt_path, 10)
            subs = SubtitlesClip(srt_path, generator)
            subs.set_pos(("center", "center"))
            final_clip = CompositeVideoClip([final_clip, subs])
    except Exception as e:
        print(f"Subtitles failed: {e}")

    # Specify temp_audiofile to avoid [Errno 22] on Windows
    temp_audio = os.path.join(tempfile.gettempdir(), f"reelforge_audio_{uuid4()}.mp3")
    final_clip.write_videofile(combined_path, threads=threads, temp_audiofile=temp_audio)
    # Cleanup temp audio
    if os.path.exists(temp_audio):
        os.remove(temp_audio)
    return combined_path


# ---------------------------------------------------------------------------
# Full generation pipeline
# ---------------------------------------------------------------------------
def rf_generate_full(topic, language, sentence_count, image_provider, sdxl_model, progress_callback=None):
    """
    Generate a complete short video. Returns dict with all outputs.
    progress_callback(step, total, message) is called for UI updates.
    """
    def progress(step, total, msg):
        if progress_callback:
            progress_callback(step, total, msg)

    assert_folder_structure()
    rem_temp_files()
    fetch_songs()

    cfg = load_config()
    model = cfg.get("ollama_model") or "llama3.2:3b"
    select_model(model)

    progress(1, 7, "Generating script...")
    script = rf_generate_script(topic, language, int(sentence_count))

    progress(2, 7, "Generating metadata...")
    metadata = rf_generate_metadata(topic, script)

    progress(3, 7, "Generating image prompts...")
    prompts = rf_generate_prompts(topic, script)

    progress(4, 7, "Generating images...")
    images = []
    for i, p in enumerate(prompts):
        path = rf_generate_image(p, provider=image_provider, model_name=sdxl_model if sdxl_model else None)
        if path:
            images.append(path)

    if not images:
        raise RuntimeError("No images were generated. Check your image provider settings.")

    progress(5, 7, "Generating TTS audio...")
    tts = TTS()
    clean_script = re.sub(r"[^\w\s.?!]", "", script)
    tts_path = os.path.join(ROOT_DIR, ".mp", f"{uuid4()}.wav")
    tts.synthesize(clean_script, tts_path)

    progress(6, 7, "Combining video...")
    video_path = rf_combine_video(images, tts_path)

    progress(7, 7, "Done!")

    # Copy outputs to temp dir so Streamlit can serve them
    import shutil
    out_dir = tempfile.mkdtemp(prefix="reelforge_")
    local_images = []
    for img_path in images:
        dest = os.path.join(out_dir, os.path.basename(img_path))
        shutil.copy2(img_path, dest)
        local_images.append(dest)
    local_video = os.path.join(out_dir, "output.mp4")
    shutil.copy2(video_path, local_video)

    return {
        "script": script,
        "title": metadata["title"],
        "description": metadata["description"],
        "image_prompts": prompts,
        "images": local_images,
        "tts_path": tts_path,
        "video_path": local_video,
    }
