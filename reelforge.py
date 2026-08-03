"""
ReelForge — AI-powered short video generation engine.
Integrated into StudioLite as a tool.

Scene-based pipeline:
1. Generate script as individual scenes
2. Each scene gets its own contextual image prompt
3. TTS generates audio with timestamps per scene
4. Video assembles each scene's image with its audio segment
"""
import os
import sys
import json
import re
import base64
import tempfile

import numpy as np
import soundfile as sf
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

# Video effect options (no transitions - they're unreliable)
MOTION_EFFECTS = ["none", "zoom_in", "zoom_out"]
COLOR_FILTERS = ["none", "warm", "cool", "vintage", "vivid"]

# Aspect ratio configurations for different video formats
ASPECT_RATIOS = {
    "9:16": {
        "name": "Vertical (TikTok/Reels)",
        "width": 1080,
        "height": 1920,
        "image_gen_width": 768,
        "image_gen_height": 1344,
        "ratio": 0.5625,
    },
    "16:9": {
        "name": "Landscape (YouTube)",
        "width": 1920,
        "height": 1080,
        "image_gen_width": 1344,
        "image_gen_height": 768,
        "ratio": 1.7778,
    },
    "1:1": {
        "name": "Square (Instagram)",
        "width": 1080,
        "height": 1080,
        "image_gen_width": 1024,
        "image_gen_height": 1024,
        "ratio": 1.0,
    },
    "4:5": {
        "name": "Portrait (Instagram Feed)",
        "width": 1080,
        "height": 1350,
        "image_gen_width": 896,
        "image_gen_height": 1152,
        "ratio": 0.8,
    },
}

# Import from local mpv2 package
from mpv2.config import (
    ROOT_DIR, assert_folder_structure,
    get_nanobanana2_api_key, get_nanobanana2_api_base_url, get_nanobanana2_model,
    get_nanobanana2_aspect_ratio, get_threads,
    get_fonts_dir, get_font, get_imagemagick_path,
    get_whisper_model, get_whisper_device, get_whisper_compute_type,
    equalize_subtitles,
    get_background_music_enabled, get_background_music_volume,
)
from mpv2.utils import rem_temp_files, fetch_songs, choose_random_song
from mpv2.classes.TtsFactory import get_tts_instance
from mpv2.audio_mixer import AudioMixer, list_background_music, get_music_dir
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
    return list_models()


def download_image_model(model_id: str, filename: str = None) -> str:
    """Download an SDXL model from Hugging Face."""
    from huggingface_hub import hf_hub_download
    models_dir = os.path.join(ROOT_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)
    return hf_hub_download(
        repo_id=model_id,
        filename=filename,
        local_dir=models_dir,
        local_dir_use_symlinks=False,
    )


# ---------------------------------------------------------------------------
# LLM Helper
# ---------------------------------------------------------------------------
def llm(prompt):
    return generate_text(prompt)


# ---------------------------------------------------------------------------
# Scene-Based Script Generation
# ---------------------------------------------------------------------------
def rf_generate_explainer_script(subject, language, num_scenes):
    """
    Generate an explainer script about the topic.
    Each scene has:
    - narration: Educational/explainer text that will be spoken (TTS)
    - visual: Description of what should be shown in the image

    Returns: list of {"narration": str, "visual": str}
    """
    prompt = f"""Create a {num_scenes}-part explainer script about: {subject}

For each part, provide:
1. NARRATION: Educational text explaining the concept (1-2 sentences, spoken by narrator)
2. VISUAL: What realistic image should accompany this narration (describe the scene to photograph)

Return as JSON array:
[
  {{"narration": "The narrator explains...", "visual": "A realistic photo of..."}},
  {{"narration": "Next point...", "visual": "An image showing..."}}
]

Rules:
- Narration should EXPLAIN the topic like a documentary
- Visual should describe a REALISTIC photograph/image that represents the concept
- Visual descriptions should include: setting, people (if any), objects, lighting
- Write narration in {language}
- No introductions or outros

Topic: {subject}"""

    response = llm(prompt)
    response = response.replace("```json", "").replace("```", "").strip()

    # Try to parse as JSON
    try:
        scenes = json.loads(response)
        if isinstance(scenes, list) and len(scenes) > 0:
            result = []
            for s in scenes[:num_scenes]:
                if isinstance(s, dict):
                    result.append({
                        "narration": str(s.get("narration", s.get("text", ""))).strip(),
                        "visual": str(s.get("visual", s.get("image", ""))).strip()
                    })
            if result:
                return result
    except json.JSONDecodeError:
        pass

    # Fallback: extract JSON from response
    match = re.search(r'\[.*\]', response, re.DOTALL)
    if match:
        try:
            scenes = json.loads(match.group())
            if isinstance(scenes, list):
                result = []
                for s in scenes[:num_scenes]:
                    if isinstance(s, dict):
                        result.append({
                            "narration": str(s.get("narration", s.get("text", ""))).strip(),
                            "visual": str(s.get("visual", s.get("image", ""))).strip()
                        })
                if result:
                    return result
        except json.JSONDecodeError:
            pass

    # Last fallback: generate simple scenes
    return [{"narration": f"Let me explain about {subject}.", "visual": f"Professional photo related to {subject}"}]


def rf_generate_image_prompt(visual_description, subject, style="photorealistic"):
    """
    Convert a visual description into a detailed image generation prompt.
    """
    style_preset = IMAGE_STYLE_PRESETS.get(style, IMAGE_STYLE_PRESETS["photorealistic"])

    # Build a detailed prompt
    prompt = f"{visual_description}, {style_preset['positive']}"

    return prompt


# ---------------------------------------------------------------------------
# Animated Subtitles - Split text into timed segments
# ---------------------------------------------------------------------------
def rf_split_text_for_subtitles(text, duration, words_per_segment=4):
    """
    Split narration text into timed subtitle segments.
    Returns list of {"text": str, "start": float, "end": float}
    """
    words = text.split()
    if not words:
        return [{"text": "", "start": 0, "end": duration}]

    segments = []
    num_segments = max(1, len(words) // words_per_segment)
    if len(words) % words_per_segment > 0:
        num_segments += 1

    time_per_segment = duration / num_segments

    for i in range(num_segments):
        start_word = i * words_per_segment
        end_word = min((i + 1) * words_per_segment, len(words))
        segment_text = ' '.join(words[start_word:end_word])

        segments.append({
            "text": segment_text,
            "start": i * time_per_segment,
            "end": (i + 1) * time_per_segment
        })

    return segments


def rf_create_subtitle_image(text, width=1080, height=200, style="bold_yellow"):
    """
    Create a transparent PNG with subtitle text using PIL.
    Returns PIL Image with RGBA (transparent background).
    """
    from PIL import Image, ImageDraw, ImageFont

    # Create transparent image
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    if not text.strip():
        return img

    # Get style settings
    sub_style = SUBTITLE_STYLES.get(style, SUBTITLE_STYLES["bold_yellow"])

    # Load font
    font_path = os.path.join(get_fonts_dir(), get_font())
    try:
        font = ImageFont.truetype(font_path, sub_style["fontsize"])
    except:
        font = ImageFont.load_default()

    # Get text size
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # Center text
    x = (width - text_width) // 2
    y = (height - text_height) // 2

    # Draw stroke (outline)
    stroke_width = sub_style.get("stroke_width", 3)
    stroke_color = sub_style.get("stroke_color", "black")
    text_color = sub_style.get("color", "#FFFFFF")

    for dx in range(-stroke_width, stroke_width + 1):
        for dy in range(-stroke_width, stroke_width + 1):
            if dx != 0 or dy != 0:
                draw.text((x + dx, y + dy), text, font=font, fill=stroke_color)

    draw.text((x, y), text, font=font, fill=text_color)

    return img


def rf_create_subtitle_clip(text, duration, style="bold_yellow"):
    """
    Create a MoviePy clip with subtitle text using PIL-generated frames.
    This bypasses ImageMagick completely.
    """
    import numpy as np

    # Create the subtitle image using PIL
    subtitle_img = rf_create_subtitle_image(text, width=1080, height=200, style=style)
    frame_array = np.array(subtitle_img)

    # Create ImageClip from numpy array
    subtitle_clip = ImageClip(frame_array, ismask=False)
    subtitle_clip = subtitle_clip.set_duration(duration)

    return subtitle_clip


def rf_generate_metadata(subject, scenes):
    """Generate video title and description."""
    script = " ".join(scenes)
    title = llm(f"Generate a YouTube title for: {subject}. Under 60 chars. Only return the title.")
    description = llm(f"Generate a YouTube description for: {script}. Under 200 chars. Only return the description.")
    return {"title": title.strip(), "description": description.strip()}


# ---------------------------------------------------------------------------
# Image Generation
# ---------------------------------------------------------------------------
_sdxl_pipe = None
_sdxl_loaded_model = None


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

    # On CUDA we use fp16 (fast, memory-efficient). On CPU we fall back to fp32
    # because most CPU kernels don't support fp16 and would raise. CPU inference
    # is 5-15 min per 1024px image but at least the pipeline works end-to-end.
    has_cuda = torch.cuda.is_available()
    dtype = torch.float16 if has_cuda else torch.float32
    device = "cuda" if has_cuda else "cpu"

    if local_model:
        _sdxl_pipe = StableDiffusionXLPipeline.from_single_file(local_model, torch_dtype=dtype)
    else:
        from diffusers import AutoPipelineForText2Image
        if has_cuda:
            _sdxl_pipe = AutoPipelineForText2Image.from_pretrained(
                "stabilityai/sdxl-turbo", torch_dtype=dtype, variant="fp16",
            )
        else:
            # variant="fp16" downloads the fp16 shards, which are useless on CPU
            # and half the size — but the pipeline would still convert them to
            # fp32 at load time. Skip variant so we pull the fp32 shards directly.
            _sdxl_pipe = AutoPipelineForText2Image.from_pretrained(
                "stabilityai/sdxl-turbo", torch_dtype=dtype,
            )

    _sdxl_pipe = _sdxl_pipe.to(device)
    _sdxl_pipe.set_progress_bar_config(disable=True)
    _sdxl_loaded_model = model_name


def rf_generate_image_nanobanana2(prompt):
    """Generate image using Gemini API."""
    api_key = get_nanobanana2_api_key()
    if not api_key:
        raise RuntimeError("NanoBanana2 API key not configured")

    base_url = get_nanobanana2_api_base_url().rstrip("/")
    model = get_nanobanana2_model()
    aspect_ratio = get_nanobanana2_aspect_ratio()
    endpoint = f"{base_url}/models/{model}:generateContent"

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "responseModalities": ["IMAGE"],
            "imageConfig": {"aspectRatio": aspect_ratio}
        },
    }

    resp = requests.post(
        endpoint,
        headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
        json=payload,
        timeout=300
    )
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

    raise RuntimeError("No image in API response")


def rf_generate_image_sdxl(prompt, model_name=None, style="photorealistic", steps=None, guidance=None, aspect_ratio="9:16", seed=None, ip_adapter_image=None):
    """Generate image using local SDXL with configurable aspect ratio and optional seed/IP-Adapter."""
    rf_load_sdxl_pipeline(model_name)

    models_dir = os.path.join(ROOT_DIR, "models")
    has_local = model_name or (os.path.isdir(models_dir) and any(f.endswith(".safetensors") for f in os.listdir(models_dir)))

    # Get dimensions from aspect ratio config
    ar_config = ASPECT_RATIOS.get(aspect_ratio, ASPECT_RATIOS["9:16"])
    img_width = ar_config["image_gen_width"]
    img_height = ar_config["image_gen_height"]

    style_preset = IMAGE_STYLE_PRESETS.get(style, IMAGE_STYLE_PRESETS["photorealistic"])
    enhanced_prompt = f"{prompt}, {style_preset['positive']}"
    negative = style_preset['negative']

    # Create generator with seed for reproducibility across views
    import torch as _torch
    generator = None
    if seed is not None:
        generator = _torch.Generator(device="cpu").manual_seed(seed)

    kwargs = dict(
        prompt=enhanced_prompt,
        negative_prompt=negative,
        num_inference_steps=steps or (30 if has_local else 8),
        guidance_scale=guidance or (7.5 if has_local else 2.0),
        width=img_width,
        height=img_height,
    )
    if generator is not None:
        kwargs["generator"] = generator

    # IP-Adapter image for multi-view consistency
    if ip_adapter_image is not None:
        try:
            kwargs["ip_adapter_image"] = ip_adapter_image
        except Exception:
            pass  # Pipeline may not have IP-Adapter loaded

    result = _sdxl_pipe(**kwargs)

    img = result.images[0]
    path = os.path.join(ROOT_DIR, ".mp", f"{uuid4()}.png")
    img.save(path, quality=95)
    return path


def rf_generate_image(prompt, provider="sdxl_turbo", model_name=None, style="photorealistic", steps=None, guidance=None, aspect_ratio="9:16", seed=None, ip_adapter_image=None):
    """Generate an image for the given prompt with configurable aspect ratio."""
    if provider == "nanobanana2":
        return rf_generate_image_nanobanana2(prompt)
    return rf_generate_image_sdxl(prompt, model_name, style, steps, guidance, aspect_ratio, seed=seed, ip_adapter_image=ip_adapter_image)


# ---------------------------------------------------------------------------
# Audio Processing Utilities (Phase 1 - Critical Fixes)
# ---------------------------------------------------------------------------
def rf_validate_audio_file(audio_path):
    """
    Validate that an audio file exists, has content, and isn't silent.
    Returns (is_valid, error_message).
    """
    if not audio_path or not os.path.exists(audio_path):
        return False, f"Audio file not found: {audio_path}"

    file_size = os.path.getsize(audio_path)
    if file_size < 1024:  # Less than 1KB
        return False, f"Audio file too small ({file_size} bytes): {audio_path}"

    try:
        audio_data, sample_rate = sf.read(audio_path)
        duration = len(audio_data) / sample_rate

        if duration < 0.1:
            return False, f"Audio too short ({duration:.2f}s): {audio_path}"

        # Check for silence (very low amplitude)
        max_amplitude = np.max(np.abs(audio_data))
        if max_amplitude < 0.001:
            return False, f"Audio appears silent (amplitude {max_amplitude:.6f}): {audio_path}"

        return True, f"Valid audio: {duration:.2f}s, {sample_rate}Hz"
    except Exception as e:
        return False, f"Failed to read audio: {e}"


def rf_clean_text_for_tts(text):
    """
    Clean text for TTS with smart symbol replacement.
    Preserves meaning by converting symbols to spoken equivalents.
    Adds natural pauses at sentence boundaries for better prosody.
    """
    if not text:
        return "Content generated."

    # Symbol to spoken word replacements
    replacements = {
        '&': ' and ',
        '%': ' percent ',
        '+': ' plus ',
        '=': ' equals ',
        '@': ' at ',
        '#': ' number ',
        '$': ' dollars ',
        '€': ' euros ',
        '£': ' pounds ',
        '°': ' degrees ',
        '/': ' or ',
        '~': ' approximately ',
        '>': ' greater than ',
        '<': ' less than ',
        '...': '... ',
        '--': ', ',
        '—': ', ',
        '–': ', ',
    }

    result = text
    for symbol, spoken in replacements.items():
        result = result.replace(symbol, spoken)

    # Remove remaining problematic characters but keep essential punctuation
    result = re.sub(r'[^\w\s.,!?;:\'\"-]', ' ', result)

    # Add natural breathing pauses: insert a brief pause after sentence-ending punctuation
    # Piper TTS respects periods/commas for pacing, so ensure spacing is clean
    result = re.sub(r'([.!?])\s*', r'\1  ', result)  # Double space after sentences
    result = re.sub(r'([,;:])\s*', r'\1 ', result)   # Single space after clauses

    # Normalize excessive whitespace but keep double-space sentence gaps
    result = re.sub(r'  +', '  ', result)
    result = re.sub(r' +([.,!?;:])', r'\1', result)  # No space before punctuation
    result = result.strip()

    # Ensure text isn't empty
    if not result or len(result) < 3:
        return "Content generated."

    return result


def rf_concatenate_audio_files(audio_paths, output_path, target_sample_rate=24000):
    """
    Concatenate multiple audio files sequentially using numpy.
    This is the CORRECT way to join audio files (not CompositeAudioClip).

    Args:
        audio_paths: List of paths to WAV files
        output_path: Path to write the combined audio
        target_sample_rate: Sample rate for output (default 24kHz for TTS)

    Returns:
        output_path on success
    """
    if not audio_paths:
        raise ValueError("No audio files to concatenate")

    all_audio = []
    actual_sample_rate = None

    for path in audio_paths:
        is_valid, msg = rf_validate_audio_file(path)
        if not is_valid:
            print(f"Warning: Skipping invalid audio - {msg}")
            continue

        audio_data, sr = sf.read(path)

        # Store sample rate from first file
        if actual_sample_rate is None:
            actual_sample_rate = sr

        # Resample if needed
        if sr != actual_sample_rate:
            from scipy import signal
            num_samples = int(len(audio_data) * actual_sample_rate / sr)
            audio_data = signal.resample(audio_data, num_samples)

        # Convert stereo to mono if needed
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)

        all_audio.append(audio_data)

    if not all_audio:
        raise ValueError("No valid audio files to concatenate")

    # Concatenate all audio sequentially
    combined = np.concatenate(all_audio)

    # Normalize to prevent clipping
    max_val = np.max(np.abs(combined))
    if max_val > 0.99:
        combined = combined * 0.95 / max_val

    sf.write(output_path, combined, actual_sample_rate or target_sample_rate)
    return output_path


# ---------------------------------------------------------------------------
# Per-Scene TTS Generation
# ---------------------------------------------------------------------------
def rf_generate_tts_per_scene(scenes_data, tts_instance):
    """
    Generate TTS audio for each scene separately.
    Uses the 'narration' field (what should be spoken) for TTS.
    Returns scenes_data with added 'audio_path' and 'duration' fields.

    Includes proper text cleaning and audio validation.
    """
    failed_scenes = []

    for i, scene in enumerate(scenes_data):
        # Use narration (explainer text) for TTS, not visual description
        narration = scene.get("narration", scene.get("text", ""))

        # Use smart text cleaning (preserves meaning)
        clean_text = rf_clean_text_for_tts(narration)

        audio_path = os.path.join(ROOT_DIR, ".mp", f"scene_{uuid4()}.wav")

        try:
            tts_instance.synthesize(clean_text, audio_path)

            # Validate the generated audio
            is_valid, msg = rf_validate_audio_file(audio_path)
            if not is_valid:
                print(f"Scene {i+1} audio validation failed: {msg}")
                # Try regenerating with fallback text
                fallback_text = f"Scene {i+1} of the presentation."
                tts_instance.synthesize(fallback_text, audio_path)
                is_valid, msg = rf_validate_audio_file(audio_path)
                if not is_valid:
                    failed_scenes.append((i, msg))
                    continue

            # Get audio duration
            audio_data, sample_rate = sf.read(audio_path)
            duration = len(audio_data) / sample_rate

            scene["audio_path"] = audio_path
            scene["duration"] = duration

        except Exception as e:
            print(f"TTS generation failed for scene {i+1}: {e}")
            failed_scenes.append((i, str(e)))
            continue

    # Report failures
    if failed_scenes:
        print(f"Warning: {len(failed_scenes)} scene(s) failed TTS generation:")
        for idx, msg in failed_scenes:
            print(f"  Scene {idx+1}: {msg}")

    # Ensure all scenes have duration (even if audio failed)
    for scene in scenes_data:
        if "duration" not in scene:
            scene["duration"] = 3.0  # Default 3 seconds for failed audio
            scene["audio_path"] = None

    return scenes_data


# ---------------------------------------------------------------------------
# Video Assembly with Animated Subtitles
# ---------------------------------------------------------------------------
def rf_combine_video(
    scenes_data,
    subtitle_style="bold_yellow",
    motion_effect="zoom_in",
    color_filter="none",
    aspect_ratio="9:16",
    music_enabled=False,
    music_path=None,
    music_volume=0.15,
):
    """
    Combine scenes into a video with animated subtitles.
    Subtitles appear one line at a time, updating as speech progresses.

    CRITICAL FIX: Uses numpy-based audio concatenation instead of CompositeAudioClip
    to ensure sequential audio playback (no missing audio).

    scenes_data: list of {"narration", "visual", "image_prompt", "image_path", "audio_path", "duration"}
    aspect_ratio: Output video aspect ratio ("9:16", "16:9", "1:1", "4:5")
    music_enabled: Whether to mix in background music
    music_path: Path to background music file (optional, will use random if not specified)
    music_volume: Volume level for background music (0.0-1.0)
    """
    combined_path = os.path.join(tempfile.gettempdir(), f"reelforge_{uuid4()}.mp4")
    threads = get_threads()

    # Get output dimensions from aspect ratio config
    ar_config = ASPECT_RATIOS.get(aspect_ratio, ASPECT_RATIOS["9:16"])
    output_width = ar_config["width"]
    output_height = ar_config["height"]
    target_ratio = ar_config["ratio"]

    # Calculate subtitle position based on aspect ratio
    subtitle_y_pos = int(output_height * 0.88)  # 88% from top

    # Collect all clips: video and subtitle layers
    all_clips = []
    audio_paths = []  # Collect paths, not clips (for numpy concatenation)
    current_time = 0.0

    for i, scene in enumerate(scenes_data):
        img_path = scene["image_path"]
        audio_path = scene.get("audio_path")
        duration = scene["duration"]
        scene_text = scene.get("narration", scene.get("text", ""))

        # Collect audio path for concatenation (skip if None)
        if audio_path and os.path.exists(audio_path):
            audio_paths.append(audio_path)

        # Create base image clip
        img_clip = ImageClip(img_path)
        img_clip = img_clip.set_duration(duration)
        img_clip = img_clip.set_fps(30)

        # Crop to target aspect ratio
        current_ratio = img_clip.w / img_clip.h
        if round(current_ratio, 4) < round(target_ratio, 4):
            # Image is taller than target - crop height
            new_height = round(img_clip.w / target_ratio)
            img_clip = crop(img_clip, width=img_clip.w, height=new_height,
                           x_center=img_clip.w / 2, y_center=img_clip.h / 2)
        else:
            # Image is wider than target - crop width
            new_width = round(target_ratio * img_clip.h)
            img_clip = crop(img_clip, width=new_width, height=img_clip.h,
                           x_center=img_clip.w / 2, y_center=img_clip.h / 2)
        img_clip = img_clip.resize((output_width, output_height))

        # Apply motion effect
        if motion_effect == "zoom_in":
            scene_dur = duration
            img_clip = img_clip.resize(lambda t, d=scene_dur: 1 + (0.1 * t / d))
        elif motion_effect == "zoom_out":
            scene_dur = duration
            img_clip = img_clip.resize(lambda t, d=scene_dur: 1.1 - (0.1 * t / d))

        # Apply color filter
        if color_filter == "warm":
            img_clip = img_clip.fx(afx.colorx, 1.1)
        elif color_filter == "cool":
            img_clip = img_clip.fx(afx.colorx, 0.9)
        elif color_filter == "vintage":
            img_clip = img_clip.fx(afx.colorx, 0.85)
        elif color_filter == "vivid":
            img_clip = img_clip.fx(afx.colorx, 1.2)

        # Set timing
        img_clip = img_clip.set_start(current_time)
        all_clips.append(img_clip)

        # Create animated subtitle segments for this scene
        subtitle_segments = rf_split_text_for_subtitles(scene_text, duration, words_per_segment=4)

        for seg in subtitle_segments:
            if not seg["text"].strip():
                continue

            try:
                # Create subtitle clip using PIL
                sub_clip = rf_create_subtitle_clip(seg["text"], seg["end"] - seg["start"], style=subtitle_style)
                # Position at bottom of frame
                sub_clip = sub_clip.set_position(("center", subtitle_y_pos))
                # Set timing relative to scene start
                sub_clip = sub_clip.set_start(current_time + seg["start"])
                all_clips.append(sub_clip)
            except Exception as e:
                print(f"Subtitle creation failed: {e}")
                continue

        # Store timing info
        scene["start_time"] = current_time
        scene["end_time"] = current_time + duration
        current_time += duration

    # Composite all video clips together
    final_video = CompositeVideoClip(all_clips, size=(output_width, output_height))
    final_video = final_video.set_fps(30)
    final_video = final_video.set_duration(current_time)

    # CRITICAL FIX: Concatenate audio using numpy (NOT CompositeAudioClip)
    # CompositeAudioClip overlays audio at set_start() times causing missing audio
    # Numpy concatenation creates proper sequential audio
    if audio_paths:
        combined_audio_path = os.path.join(ROOT_DIR, ".mp", f"combined_audio_{uuid4()}.wav")
        try:
            rf_concatenate_audio_files(audio_paths, combined_audio_path)

            # Mix with background music if enabled
            if music_enabled:
                try:
                    # Get music path - use provided or pick random from library
                    actual_music_path = music_path
                    if not actual_music_path:
                        # Try to use a random song from the music directory
                        available_music = list_background_music()
                        if available_music:
                            import random
                            music_file = random.choice(available_music)
                            actual_music_path = str(get_music_dir() / music_file)

                    if actual_music_path and os.path.exists(actual_music_path):
                        mixer = AudioMixer(
                            music_volume=music_volume,
                            duck_enabled=True,
                            duck_factor=0.3,
                        )
                        mixed_audio_path = os.path.join(
                            ROOT_DIR, ".mp", f"mixed_audio_{uuid4()}.wav"
                        )
                        mixer.mix(combined_audio_path, actual_music_path, mixed_audio_path)
                        combined_audio_path = mixed_audio_path
                        print(f"Background music mixed with volume {music_volume}")
                    else:
                        print("No background music file available, using narration only")
                except Exception as e:
                    print(f"Background music mixing failed: {e}")
                    # Continue with narration-only audio

            final_audio = AudioFileClip(combined_audio_path)
            final_video = final_video.set_audio(final_audio)
        except Exception as e:
            print(f"Audio concatenation failed: {e}")
            # Fallback: try individual audio clips with MoviePy (less reliable)
            try:
                audio_clips = [AudioFileClip(p) for p in audio_paths]
                from moviepy.editor import concatenate_audioclips
                final_audio = concatenate_audioclips(audio_clips)
                final_video = final_video.set_audio(final_audio)
            except Exception as e2:
                print(f"Fallback audio also failed: {e2}")
                # Video will have no audio

    # Write video file
    final_video.write_videofile(
        combined_path,
        threads=threads,
        fps=30,
        codec='libx264',
        audio_codec='aac',
        verbose=False,
        logger=None,
    )

    return combined_path


# ---------------------------------------------------------------------------
# Full Generation Pipeline
# ---------------------------------------------------------------------------
def rf_generate_full(
    topic,
    language,
    sentence_count,
    image_provider,
    sdxl_model,
    progress_callback=None,
    image_style="photorealistic",
    image_steps=None,
    image_guidance=None,
    subtitle_style="bold_yellow",
    ken_burns_effect="zoom_in",  # Kept for backward compat, maps to motion_effect
    transition="none",  # Ignored - transitions removed
    color_filter="none",
    num_images=3,
    aspect_ratio="9:16",  # Video format: "9:16", "16:9", "1:1", "4:5"
    music_enabled=None,  # Whether to add background music
    music_path=None,  # Path to specific music file (or None for random)
    music_volume=None,  # Volume for background music (0.0-1.0)
):
    """
    Generate a complete video with scene-based pipeline.

    Pipeline:
    1. Generate explainer script with SEPARATE narration (what to speak) and visual (what to show)
    2. Generate image prompts from visual descriptions
    3. Generate images
    4. Generate TTS from narration (explainer text)
    5. Burn narration text onto images as subtitles
    6. Combine into video (with optional background music)

    Returns dict with:
    - scenes: list of {narration, visual, image_prompt, image_path, audio_path, duration}
    - script: full narration text
    - video_path: output video
    - title, description: metadata
    """
    def progress(step, total, msg):
        if progress_callback:
            progress_callback(step, total, msg)

    assert_folder_structure()
    rem_temp_files()
    fetch_songs()

    # Configure LLM
    cfg = load_config()
    backend = cfg.get("llm_backend", "llamacpp")
    select_backend(backend)
    if backend == "ollama":
        model = cfg.get("ollama_model") or "llama3.2:3b"
        select_model(model)
    else:
        gguf_model = cfg.get("gguf_model", "")
        if gguf_model:
            select_model(gguf_model)

    # Step 1: Generate explainer script with narration + visual pairs
    progress(1, 7, "Generating explainer script...")
    scenes = rf_generate_explainer_script(topic, language, num_images)

    # Step 2: Create detailed image prompts from visual descriptions
    progress(2, 7, "Creating image prompts...")
    scenes_data = []
    for i, scene in enumerate(scenes):
        progress(2, 7, f"Creating prompt for scene {i+1}/{len(scenes)}...")
        # Use visual description (not narration) to generate image prompt
        image_prompt = rf_generate_image_prompt(scene["visual"], topic, image_style)
        scenes_data.append({
            "narration": scene["narration"],  # What TTS will speak
            "visual": scene["visual"],        # Visual description
            "image_prompt": image_prompt,     # Detailed prompt for image gen
            "image_path": None,
        })

    # Step 3: Generate images from visual descriptions
    progress(3, 7, f"Generating {len(scenes_data)} images...")
    for i, scene in enumerate(scenes_data):
        progress(3, 7, f"Generating image {i+1}/{len(scenes_data)}: {scene['visual'][:40]}...")
        image_path = rf_generate_image(
            scene["image_prompt"],
            provider=image_provider,
            model_name=sdxl_model,
            style=image_style,
            steps=image_steps,
            guidance=image_guidance,
            aspect_ratio=aspect_ratio,
        )
        scene["image_path"] = image_path

    # Verify all images generated
    missing = [s for s in scenes_data if not s["image_path"]]
    if missing:
        raise RuntimeError(f"Failed to generate images for {len(missing)} scenes")

    # Step 4: Generate metadata
    progress(4, 7, "Generating metadata...")
    full_script = " ".join([s["narration"] for s in scenes_data])
    metadata = rf_generate_metadata(topic, [s["narration"] for s in scenes_data])

    # Step 5: Generate TTS per scene from narration (for proper sync)
    progress(5, 7, "Generating speech for each scene...")
    tts = get_tts_instance()
    scenes_data = rf_generate_tts_per_scene(scenes_data, tts)

    # Step 6: Combine video with per-scene timing and text overlay
    progress(6, 7, "Creating video with synced audio and subtitles...")
    motion_effect = ken_burns_effect if ken_burns_effect in MOTION_EFFECTS else "none"

    # Get background music settings from config if not specified
    actual_music_enabled = music_enabled if music_enabled is not None else get_background_music_enabled()
    actual_music_volume = music_volume if music_volume is not None else get_background_music_volume()

    video_path = rf_combine_video(
        scenes_data,
        subtitle_style=subtitle_style,
        motion_effect=motion_effect,
        color_filter=color_filter,
        aspect_ratio=aspect_ratio,
        music_enabled=actual_music_enabled,
        music_path=music_path,
        music_volume=actual_music_volume,
    )

    # Step 7: Copy to output directory
    progress(7, 7, "Done!")
    import shutil
    out_dir = tempfile.mkdtemp(prefix="reelforge_")

    # Copy images
    for scene in scenes_data:
        if scene["image_path"]:
            dest = os.path.join(out_dir, os.path.basename(scene["image_path"]))
            shutil.copy2(scene["image_path"], dest)
            scene["image_path"] = dest

    # Copy video
    local_video = os.path.join(out_dir, "output.mp4")
    shutil.copy2(video_path, local_video)

    # Calculate total duration
    total_duration = sum(s.get("duration", 0) for s in scenes_data)

    return {
        "scenes": scenes_data,
        "script": full_script,
        "title": metadata["title"],
        "description": metadata["description"],
        "image_prompts": [s["image_prompt"] for s in scenes_data],
        "images": [s["image_path"] for s in scenes_data],
        "video_path": local_video,
        "total_duration": total_duration,
    }


# Backward compatibility alias
TRANSITION_TYPES = ["none"]  # Transitions removed - they were unreliable
