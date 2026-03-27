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
import random

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

# ---------------------------------------------------------------------------
# Recommended SDXL Models for Realistic Human Faces
# ---------------------------------------------------------------------------
RECOMMENDED_IMAGE_MODELS = [
    {
        "name": "RealVisXL V4.0",
        "id": "SG161222/RealVisXL_V4.0",
        "file": "RealVisXL_V4.0.safetensors",
        "size": "6.5 GB",
        "description": "Best for photorealistic humans and faces",
        "style": "photorealistic",
    },
    {
        "name": "Juggernaut XL V9",
        "id": "RunDiffusion/Juggernaut-XL-v9",
        "file": "Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors",
        "size": "6.5 GB",
        "description": "Excellent for realistic portraits and people",
        "style": "photorealistic",
    },
    {
        "name": "DreamShaper XL",
        "id": "Lykon/dreamshaper-xl-v2-turbo",
        "file": "DreamShaperXL_Turbo_v2_1.safetensors",
        "size": "6.5 GB",
        "description": "Fast, versatile, good for stylized content",
        "style": "artistic",
    },
    {
        "name": "SDXL Turbo (Default)",
        "id": "stabilityai/sdxl-turbo",
        "file": "sd_xl_turbo_1.0_fp16.safetensors",
        "size": "6.5 GB",
        "description": "Fast generation, good for quick previews",
        "style": "general",
    },
]

# Image style presets for different content types
IMAGE_STYLE_PRESETS = {
    "photorealistic": {
        "positive": "photorealistic, ultra detailed, sharp focus, professional photography, 8k, natural lighting, realistic skin texture, detailed eyes",
        "negative": "cartoon, anime, illustration, painting, drawing, blurry, low quality, deformed, ugly, bad anatomy, text, watermark, oversaturated, plastic skin",
    },
    "cinematic": {
        "positive": "cinematic shot, movie still, dramatic lighting, film grain, shallow depth of field, professional color grading, 35mm film",
        "negative": "cartoon, anime, blurry, low quality, amateur, overexposed, flat lighting",
    },
    "artistic": {
        "positive": "digital art, trending on artstation, highly detailed, vivid colors, professional illustration, concept art",
        "negative": "blurry, low quality, deformed, ugly, bad anatomy, text, watermark",
    },
    "portrait": {
        "positive": "professional portrait, studio lighting, sharp focus, detailed face, realistic eyes, natural skin, DSLR quality, bokeh background",
        "negative": "deformed face, ugly, blurry, low quality, bad anatomy, distorted features, extra limbs, text, watermark",
    },
}

# Subtitle style presets
SUBTITLE_STYLES = {
    "classic": {"color": "#FFFFFF", "stroke_color": "black", "stroke_width": 3, "fontsize": 80},
    "bold_yellow": {"color": "#FFFF00", "stroke_color": "black", "stroke_width": 5, "fontsize": 100},
    "neon_pink": {"color": "#FF1493", "stroke_color": "#000000", "stroke_width": 4, "fontsize": 90},
    "minimal": {"color": "#FFFFFF", "stroke_color": "#333333", "stroke_width": 2, "fontsize": 70},
    "news": {"color": "#FFFFFF", "stroke_color": "#CC0000", "stroke_width": 4, "fontsize": 85},
}

# Video transition types
TRANSITION_TYPES = ["none", "crossfade", "fade_black", "slide_left", "zoom_in"]

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
from mpv2.llm_provider import (
    select_model, generate_text, list_models, check_ollama_connection,
    check_backend_status, select_backend, get_backend, get_gguf_models,
    download_model, RECOMMENDED_MODELS, check_llamacpp_available,
)

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


def download_image_model(model_id: str, filename: str = None, progress_callback=None) -> str:
    """Download an SDXL model from Hugging Face."""
    from huggingface_hub import hf_hub_download

    models_dir = os.path.join(ROOT_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)

    local_path = hf_hub_download(
        repo_id=model_id,
        filename=filename,
        local_dir=models_dir,
        local_dir_use_symlinks=False,
    )
    return local_path


# ---------------------------------------------------------------------------
# Video Effects
# ---------------------------------------------------------------------------
def apply_ken_burns(clip, effect="zoom_in", intensity=0.1):
    """Apply Ken Burns zoom/pan effect to a clip."""
    from moviepy.editor import vfx

    duration = clip.duration

    if effect == "zoom_in":
        # Start zoomed out, end zoomed in
        def zoom_func(t):
            return 1 + (intensity * t / duration)
        return clip.resize(zoom_func)

    elif effect == "zoom_out":
        # Start zoomed in, end zoomed out
        def zoom_func(t):
            return 1 + intensity - (intensity * t / duration)
        return clip.resize(zoom_func)

    elif effect == "pan_left":
        # Pan from right to left
        def position_func(t):
            x = int(clip.w * intensity * (1 - t / duration))
            return (x, 0)
        return clip.set_position(position_func)

    elif effect == "pan_right":
        # Pan from left to right
        def position_func(t):
            x = int(-clip.w * intensity + clip.w * intensity * t / duration)
            return (x, 0)
        return clip.set_position(position_func)

    return clip


def apply_transition(clip1, clip2, transition_type="crossfade", duration=0.5):
    """Apply transition between two clips."""
    from moviepy.editor import CompositeVideoClip, concatenate_videoclips

    if transition_type == "none" or duration <= 0:
        return concatenate_videoclips([clip1, clip2])

    elif transition_type == "crossfade":
        # Crossfade transition
        clip1 = clip1.crossfadeout(duration)
        clip2 = clip2.crossfadein(duration)
        # Overlap the clips
        clip2 = clip2.set_start(clip1.duration - duration)
        return CompositeVideoClip([clip1, clip2])

    elif transition_type == "fade_black":
        # Fade to black transition
        clip1 = clip1.fadeout(duration / 2)
        clip2 = clip2.fadein(duration / 2)
        return concatenate_videoclips([clip1, clip2])

    return concatenate_videoclips([clip1, clip2])


def apply_color_filter(clip, filter_type="none"):
    """Apply color filter/grading to clip."""
    from moviepy.editor import vfx

    if filter_type == "warm":
        return clip.fx(vfx.colorx, 1.1)  # Slightly warmer
    elif filter_type == "cool":
        return clip.fx(vfx.colorx, 0.9)  # Slightly cooler
    elif filter_type == "vintage":
        return clip.fx(vfx.colorx, 0.85)
    elif filter_type == "vivid":
        return clip.fx(vfx.colorx, 1.2)

    return clip


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


def rf_generate_image_sdxl(prompt, model_name=None, style="photorealistic", steps=None, guidance=None):
    rf_load_sdxl_pipeline(model_name)
    models_dir = os.path.join(ROOT_DIR, "models")
    has_local = model_name or (os.path.isdir(models_dir) and any(f.endswith(".safetensors") for f in os.listdir(models_dir)))

    # Get style preset
    style_preset = IMAGE_STYLE_PRESETS.get(style, IMAGE_STYLE_PRESETS["photorealistic"])

    try:
        # Enhanced prompt with style preset
        enhanced_prompt = f"{prompt}, {style_preset['positive']}"
        negative = style_preset['negative']

        if has_local:
            # Full SDXL model - more steps for quality
            result = _sdxl_pipe(
                prompt=enhanced_prompt,
                negative_prompt=negative,
                num_inference_steps=steps or 30,
                guidance_scale=guidance or 7.5,
                width=768,
                height=1344,
            )
        else:
            # SDXL Turbo - optimized for speed
            result = _sdxl_pipe(
                prompt=enhanced_prompt,
                negative_prompt=negative,
                num_inference_steps=steps or 8,
                guidance_scale=guidance or 2.0,
                width=768,
                height=1344,
            )
        img = result.images[0]
        path = os.path.join(ROOT_DIR, ".mp", f"{uuid4()}.png")
        img.save(path, quality=95)
        return path
    except Exception as e:
        print(f"Image generation error: {e}")
        return None


def rf_generate_image(prompt, provider="sdxl_turbo", model_name=None, style="photorealistic", steps=None, guidance=None):
    if provider == "nanobanana2":
        return rf_generate_image_nanobanana2(prompt)
    return rf_generate_image_sdxl(prompt, model_name, style, steps, guidance)


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


def rf_combine_video(images, tts_path, subtitle_style="bold_yellow", ken_burns_effect="zoom_in", transition="none", color_filter="none"):
    # Use tempfile for the output to avoid path issues
    combined_path = os.path.join(tempfile.gettempdir(), f"reelforge_{uuid4()}.mp4")
    threads = get_threads()
    tts_clip = AudioFileClip(tts_path)
    max_duration = tts_clip.duration
    req_dur = max_duration / len(images)

    # Get subtitle style
    sub_style = SUBTITLE_STYLES.get(subtitle_style, SUBTITLE_STYLES["bold_yellow"])

    generator = lambda txt: TextClip(
        txt, font=os.path.join(get_fonts_dir(), get_font()),
        fontsize=sub_style["fontsize"],
        color=sub_style["color"],
        stroke_color=sub_style["stroke_color"],
        stroke_width=sub_style["stroke_width"],
        size=(1080, 1920), method="caption",
    )

    # Ken Burns effects to cycle through
    kb_effects = ["zoom_in", "zoom_out", "pan_left", "pan_right"] if ken_burns_effect == "random" else [ken_burns_effect]

    clips, tot_dur = [], 0
    effect_idx = 0
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

            # Apply Ken Burns effect
            if ken_burns_effect != "none":
                current_effect = kb_effects[effect_idx % len(kb_effects)]
                try:
                    clip = apply_ken_burns(clip, effect=current_effect, intensity=0.08)
                except Exception as e:
                    print(f"Ken Burns effect failed: {e}")
                effect_idx += 1

            # Apply color filter
            if color_filter != "none":
                try:
                    clip = apply_color_filter(clip, color_filter)
                except Exception as e:
                    print(f"Color filter failed: {e}")

            clips.append(clip)
            tot_dur += clip.duration

    # Apply transitions between clips if requested
    if transition != "none" and len(clips) > 1:
        try:
            transitioned_clips = [clips[0]]
            for i in range(1, len(clips)):
                combined = apply_transition(transitioned_clips[-1], clips[i], transition, duration=0.3)
                transitioned_clips[-1] = combined
            final_clip = transitioned_clips[0].set_fps(30)
        except Exception as e:
            print(f"Transition failed, using simple concatenation: {e}")
            final_clip = concatenate_videoclips(clips).set_fps(30)
    else:
        final_clip = concatenate_videoclips(clips).set_fps(30)

    # Try to add background music, but make it optional
    try:
        song = choose_random_song()
        song_clip = AudioFileClip(song).fx(afx.volumex, 0.1)
        # Set duration to match TTS
        song_clip = song_clip.set_duration(tts_clip.duration)
        comp_audio = CompositeAudioClip([tts_clip, song_clip])
        comp_audio = comp_audio.set_duration(tts_clip.duration)
        final_clip = final_clip.set_audio(comp_audio).set_duration(tts_clip.duration)
    except Exception as e:
        print(f"Background music not available: {e}")
        # Use TTS audio only (no background music)
        final_clip = final_clip.set_audio(tts_clip).set_duration(tts_clip.duration)

    try:
        srt_path = rf_generate_subtitles(tts_path)
        if srt_path:
            equalize_subtitles(srt_path, 10)
            subs = SubtitlesClip(srt_path, generator)
            # set_pos returns a new clip, need to assign it
            subs = subs.set_pos(("center", "bottom"))
            subs = subs.set_duration(final_clip.duration)
            final_clip = CompositeVideoClip([final_clip, subs], size=(1080, 1920))
    except Exception as e:
        print(f"Subtitles failed: {e}")
        import traceback
        traceback.print_exc()

    # Specify temp_audiofile to avoid [Errno 22] on Windows
    temp_audio = os.path.join(tempfile.gettempdir(), f"reelforge_audio_{uuid4()}.m4a")
    try:
        final_clip.write_videofile(
            combined_path,
            threads=threads,
            temp_audiofile=temp_audio,
            fps=30,
            codec='libx264',
            audio_codec='aac',
            verbose=False,
            logger=None,
        )
    except AttributeError as e:
        # Fallback for MoviePy audio writer bug
        print(f"Audio writer issue, trying alternative method: {e}")
        final_clip.write_videofile(
            combined_path,
            threads=threads,
            fps=30,
            codec='libx264',
            audio=True,
            verbose=False,
            logger=None,
        )
    # Cleanup temp audio
    if os.path.exists(temp_audio):
        try:
            os.remove(temp_audio)
        except Exception:
            pass
    return combined_path


# ---------------------------------------------------------------------------
# Full generation pipeline
# ---------------------------------------------------------------------------
def rf_generate_full(
    topic,
    language,
    sentence_count,
    image_provider,
    sdxl_model,
    progress_callback=None,
    # New customization options
    image_style="photorealistic",
    image_steps=None,
    image_guidance=None,
    subtitle_style="bold_yellow",
    ken_burns_effect="zoom_in",
    transition="none",
    color_filter="none",
    num_images=3,
):
    """
    Generate a complete short video. Returns dict with all outputs.
    progress_callback(step, total, message) is called for UI updates.

    New options:
    - image_style: "photorealistic", "cinematic", "artistic", "portrait"
    - subtitle_style: "classic", "bold_yellow", "neon_pink", "minimal", "news"
    - ken_burns_effect: "none", "zoom_in", "zoom_out", "pan_left", "pan_right", "random"
    - transition: "none", "crossfade", "fade_black"
    - color_filter: "none", "warm", "cool", "vintage", "vivid"
    """
    def progress(step, total, msg):
        if progress_callback:
            progress_callback(step, total, msg)

    assert_folder_structure()
    rem_temp_files()
    fetch_songs()

    cfg = load_config()
    # Set backend and model
    backend = cfg.get("llm_backend", "llamacpp")
    select_backend(backend)
    if backend == "ollama":
        model = cfg.get("ollama_model") or "llama3.2:3b"
        select_model(model)
    else:
        # For llama.cpp, use the first available GGUF model or the configured one
        gguf_model = cfg.get("gguf_model", "")
        if gguf_model:
            select_model(gguf_model)

    progress(1, 8, "Generating script...")
    script = rf_generate_script(topic, language, int(sentence_count))

    progress(2, 8, "Generating metadata...")
    metadata = rf_generate_metadata(topic, script)

    progress(3, 8, "Generating image prompts...")
    prompts = rf_generate_prompts(topic, script)
    # Limit to requested number of images
    prompts = prompts[:num_images]

    progress(4, 8, f"Generating {len(prompts)} images ({image_style} style)...")
    images = []
    for i, p in enumerate(prompts):
        progress(4, 8, f"Generating image {i+1}/{len(prompts)}...")
        path = rf_generate_image(
            p,
            provider=image_provider,
            model_name=sdxl_model if sdxl_model else None,
            style=image_style,
            steps=image_steps,
            guidance=image_guidance,
        )
        if path:
            images.append(path)

    if not images:
        raise RuntimeError("No images were generated. Check your image provider settings.")

    progress(5, 8, "Generating TTS audio...")
    tts = TTS()
    clean_script = re.sub(r"[^\w\s.?!]", "", script)
    tts_path = os.path.join(ROOT_DIR, ".mp", f"{uuid4()}.wav")
    tts.synthesize(clean_script, tts_path)

    progress(6, 8, "Combining video with effects...")
    video_path = rf_combine_video(
        images,
        tts_path,
        subtitle_style=subtitle_style,
        ken_burns_effect=ken_burns_effect,
        transition=transition,
        color_filter=color_filter,
    )

    progress(7, 8, "Finalizing...")

    # Copy outputs to temp dir so Streamlit can serve them
    import shutil
    progress(8, 8, "Done!")
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
