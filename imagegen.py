"""
Image generation, editing, and post-processing.

Backs the /api/v1/images/* endpoints. Three execution paths:

- **Local SDXL** (default): reuses ``reelforge.rf_load_sdxl_pipeline`` for
  text-to-image and adds dedicated img2img/inpaint pipelines that share the
  same UNet/VAE/text-encoder weights via ``from_pipe`` to avoid double-load.
- **Fooocus API** (optional): HTTP client for a Fooocus-API server at
  ``fooocus_api_url``. Activated when ``provider="fooocus"``. Fooocus styles
  are fetched live from its ``/v1/engines/styles`` endpoint.
- **NanoBanana2 / Gemini**: existing wrapper in reelforge.

Capabilities exposed:
- text-to-image (T2I)
- image-to-image (I2I) with strength 0-1
- inpaint (mask-based local edit)
- variation (low-strength I2I, "more like this")
- upscale (Real-ESRGAN if installed, else Lanczos via ``upscaler``)
- background removal (rembg if installed, else error)
- prompt enhancement (LLM rewrites raw prompt)
"""

import base64
import gc
import io
import json
import logging
import os
import random
from typing import Optional, Callable
from uuid import uuid4

import requests
import torch
import PIL.Image
import PIL.ImageOps

logger = logging.getLogger(__name__)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(ROOT_DIR, ".mp", "images")
os.makedirs(OUTPUT_DIR, exist_ok=True)

_img2img_pipe = None
_inpaint_pipe = None
_pix2pix_pipe = None  # timbrooks/instruct-pix2pix — instruction edits ("raise her hand", "make it night")
_qwen_edit_pipe = None  # Qwen-Image-Edit-2509 — current SOTA local instruction editor (Apr 2026)
_blip_processor = None
_blip_model = None

INSTRUCT_PIX2PIX_REPO = "timbrooks/instruct-pix2pix"

# Qwen-Image-Edit-2509 — Apache-2.0, GGUF Q4_K_S transformer + base components.
# 5-10x better than InstructPix2Pix on instruction edits per GEdit-Bench.
QWEN_EDIT_GGUF_DIR = os.path.join(ROOT_DIR, ".models", "qwen-image-edit-2509-gguf")
QWEN_EDIT_GGUF = os.path.join(QWEN_EDIT_GGUF_DIR, "Qwen-Image-Edit-2509-Q4_K_S.gguf")
QWEN_EDIT_BASE = os.path.join(ROOT_DIR, ".models", "qwen-image-edit-2509-base")
QWEN_EDIT_REPO = "Qwen/Qwen-Image-Edit-2509"  # for transformer config metadata

# Reasonable per-style defaults for the negative prompt textbox. The UI seeds
# this on first load so users start with a sane baseline.
DEFAULT_NEGATIVES = {
    "_base": (
        "blurry, low quality, jpeg artifacts, compression artifacts, deformed, "
        "bad anatomy, ugly, distorted, watermark, text, signature, username, "
        "extra limbs, missing limbs, fused fingers, too many fingers, mutated hands"
    ),
    "photorealistic": (
        "cartoon, anime, illustration, painting, drawing, plastic skin, "
        "blurry, low quality, deformed, ugly, bad anatomy, text, watermark, oversaturated"
    ),
    "cinematic": (
        "cartoon, anime, blurry, low quality, amateur, overexposed, flat lighting, "
        "phone camera, snapshot, low contrast, washed out colors, watermark, text"
    ),
    "anime": (
        "photorealistic, photograph, 3d render, blurry, low quality, deformed, "
        "ugly, bad anatomy, watermark, text"
    ),
    "3d_render": (
        "2d, flat shading, cartoon, blurry, low quality, deformed, ugly, "
        "bad anatomy, watermark, text"
    ),
    "digital_art": (
        "blurry, low quality, deformed, ugly, bad anatomy, text, watermark, "
        "amateur, low resolution"
    ),
    "portrait": (
        "deformed face, ugly, blurry, low quality, bad anatomy, distorted features, "
        "extra limbs, watermark, text, asymmetric eyes, crossed eyes"
    ),
    "fantasy": (
        "modern, photograph, mundane, blurry, low quality, deformed, watermark, text"
    ),
    "minimalist": (
        "cluttered, busy, complex, detailed background, photorealistic, "
        "blurry, low quality, watermark, text"
    ),
}


# ---------------------------------------------------------------------------
# Aspect-ratio / size presets
# ---------------------------------------------------------------------------

# SDXL native trains best at 1024-area resolutions. Each entry gives
# (width, height) snapped to multiples of 8.
SIZE_PRESETS = {
    "square_1024":     (1024, 1024),
    "square_1536":     (1536, 1536),
    "portrait_9x16":   (768, 1344),
    "portrait_2x3":    (832, 1216),
    "portrait_4x5":    (896, 1152),
    "landscape_16x9":  (1344, 768),
    "landscape_3x2":   (1216, 832),
    "landscape_4x3":   (1152, 896),
    "wallpaper_21x9":  (1536, 640),
}


# ---------------------------------------------------------------------------
# Models / providers enumeration
# ---------------------------------------------------------------------------

def list_local_sdxl_models() -> list:
    """List .safetensors files in the local ``models/`` directory."""
    models_dir = os.path.join(ROOT_DIR, "models")
    out = []
    if os.path.isdir(models_dir):
        for f in sorted(os.listdir(models_dir)):
            if f.endswith(".safetensors"):
                out.append({"id": f, "name": f.replace(".safetensors", ""), "path": os.path.join(models_dir, f)})
    # The default SDXL Turbo cached HF model is always available
    out.insert(0, {"id": "sdxl_turbo", "name": "SDXL Turbo (default)", "path": None})
    return out


STYLE_PRESETS = {
    "photorealistic": {
        "positive": "photorealistic, ultra detailed, sharp focus, professional photography, 8k, natural lighting, realistic skin texture, detailed eyes",
        "negative": "cartoon, anime, illustration, painting, drawing, blurry, low quality, deformed, ugly, bad anatomy, text, watermark",
    },
    "cinematic": {
        "positive": "cinematic shot, movie still, dramatic lighting, film grain, shallow depth of field, professional color grading, 35mm film",
        "negative": "cartoon, anime, blurry, low quality, amateur, overexposed, flat lighting",
    },
    "anime": {
        "positive": "anime style, vibrant colors, sharp lines, cel shading, detailed background, studio quality",
        "negative": "photorealistic, photograph, low quality, blurry, deformed, ugly, bad anatomy",
    },
    "3d_render": {
        "positive": "3d render, octane render, unreal engine 5, ray tracing, subsurface scattering, physically based rendering",
        "negative": "2d, flat, cartoon, blurry, low quality, deformed",
    },
    "digital_art": {
        "positive": "digital art, trending on artstation, highly detailed, vivid colors, professional illustration, concept art",
        "negative": "blurry, low quality, deformed, ugly, bad anatomy, text, watermark",
    },
    "portrait": {
        "positive": "professional portrait, studio lighting, sharp focus, detailed face, realistic eyes, natural skin, DSLR quality, bokeh",
        "negative": "deformed face, ugly, blurry, low quality, bad anatomy, distorted features, extra limbs",
    },
    "fantasy": {
        "positive": "fantasy art, epic, ethereal lighting, mystical atmosphere, intricate details, by Greg Rutkowski and Alphonse Mucha",
        "negative": "modern, photograph, blurry, low quality, deformed",
    },
    "minimalist": {
        "positive": "minimalist, clean lines, negative space, simple composition, flat design, pastel colors",
        "negative": "cluttered, busy, complex, detailed background, photorealistic",
    },
}


# ---------------------------------------------------------------------------
# Fooocus client
# ---------------------------------------------------------------------------

def _fooocus_url() -> str:
    """Fetch from config.json (existing setting) with localhost default."""
    try:
        from mpv2.config import get_fooocus_api_url
        return get_fooocus_api_url().rstrip("/")
    except Exception:
        return "http://127.0.0.1:8888"


def fooocus_available() -> bool:
    """Quick TCP-ping of the Fooocus API to determine if we should offer it."""
    try:
        r = requests.get(f"{_fooocus_url()}/ping", timeout=1.5)
        return r.status_code == 200
    except requests.RequestException:
        return False


def fooocus_styles() -> list:
    """Pull the current style list from Fooocus's engines endpoint."""
    if not fooocus_available():
        return []
    try:
        r = requests.get(f"{_fooocus_url()}/v1/engines/styles", timeout=5)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list):
            return data
        return data.get("styles") or []
    except Exception as e:
        logger.warning("Fooocus styles fetch failed: %s", e)
        return []


def fooocus_text2image(prompt: str, negative_prompt: str, width: int, height: int,
                       seed: Optional[int], style: Optional[str],
                       steps: int = 30) -> str:
    """Send a T2I request to a Fooocus-API server, return path to output PNG."""
    payload = {
        "prompt": prompt,
        "negative_prompt": negative_prompt or "",
        "style_selections": [style] if style and style != "(no style)" else [],
        "performance_selection": "Speed" if steps <= 20 else "Quality",
        "aspect_ratios_selection": f"{width}*{height}",
        "image_seed": seed if seed is not None else random.randint(0, 2**31 - 1),
        "image_number": 1,
        "async_process": False,
    }
    r = requests.post(
        f"{_fooocus_url()}/v1/generation/text-to-image",
        json=payload,
        timeout=600,
    )
    r.raise_for_status()
    data = r.json()
    items = data if isinstance(data, list) else data.get("results") or []
    if not items:
        raise RuntimeError(f"Fooocus returned no images: {data}")
    item = items[0]
    out_path = os.path.join(OUTPUT_DIR, f"fooocus_t2i_{uuid4().hex[:8]}.png")
    if "url" in item:
        img_data = requests.get(item["url"], timeout=60).content
    elif "base64" in item:
        img_data = base64.b64decode(item["base64"])
    else:
        raise RuntimeError(f"Unexpected Fooocus response shape: {item}")
    with open(out_path, "wb") as f:
        f.write(img_data)
    return out_path


def fooocus_image2image(image_path: str, prompt: str, negative_prompt: str,
                        denoise_strength: float, seed: Optional[int],
                        style: Optional[str]) -> str:
    """Send an image-prompt-modify request to Fooocus."""
    with open(image_path, "rb") as f:
        files = {"input_image": (os.path.basename(image_path), f, "image/png")}
        data = {
            "prompt": prompt,
            "negative_prompt": negative_prompt or "",
            "style_selections": [style] if style and style != "(no style)" else [],
            "image_seed": str(seed if seed is not None else random.randint(0, 2**31 - 1)),
            "image_number": "1",
            "denoising_strength": str(denoise_strength),
            "async_process": "false",
        }
        r = requests.post(
            f"{_fooocus_url()}/v1/generation/image-upscale-vary",
            files=files, data=data, timeout=600,
        )
    r.raise_for_status()
    body = r.json()
    items = body if isinstance(body, list) else body.get("results") or []
    if not items:
        raise RuntimeError(f"Fooocus returned no images: {body}")
    item = items[0]
    out_path = os.path.join(OUTPUT_DIR, f"fooocus_i2i_{uuid4().hex[:8]}.png")
    if "url" in item:
        img_data = requests.get(item["url"], timeout=60).content
    elif "base64" in item:
        img_data = base64.b64decode(item["base64"])
    else:
        raise RuntimeError(f"Unexpected Fooocus response shape: {item}")
    with open(out_path, "wb") as f:
        f.write(img_data)
    return out_path


# ---------------------------------------------------------------------------
# Local SDXL pipelines for img2img / inpaint
# ---------------------------------------------------------------------------

def _free_vram():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _load_img2img():
    """Load SDXL Img2Img sharing weights with the cached T2I pipeline."""
    global _img2img_pipe
    if _img2img_pipe is not None:
        return _img2img_pipe
    import reelforge
    reelforge.rf_load_sdxl_pipeline()
    if reelforge._sdxl_pipe is None:
        raise RuntimeError("SDXL pipeline failed to load")
    from diffusers import StableDiffusionXLImg2ImgPipeline
    _img2img_pipe = StableDiffusionXLImg2ImgPipeline(
        vae=reelforge._sdxl_pipe.vae,
        text_encoder=reelforge._sdxl_pipe.text_encoder,
        text_encoder_2=reelforge._sdxl_pipe.text_encoder_2,
        tokenizer=reelforge._sdxl_pipe.tokenizer,
        tokenizer_2=reelforge._sdxl_pipe.tokenizer_2,
        unet=reelforge._sdxl_pipe.unet,
        scheduler=reelforge._sdxl_pipe.scheduler,
    )
    if torch.cuda.is_available():
        _img2img_pipe.to("cuda")
    return _img2img_pipe


def qwen_edit_available() -> bool:
    """True only when the FULL GGUF transformer + ALL 4 text_encoder shards
    + VAE are present at expected sizes. Strict thresholds so an in-flight
    download isn't reported as ready — would crash GGUF loading."""
    # GGUF Q4_K_S full size is ~11.4 GB. Require >= 11 GB.
    if not (os.path.isfile(QWEN_EDIT_GGUF) and os.path.getsize(QWEN_EDIT_GGUF) >= 11 * 1024 * 1024 * 1024):
        return False
    # text_encoder is sharded 4 ways: 4.7+4.7+4.6+1.6 GB. Require all 4.
    expected_min = [4_500, 4_500, 4_500, 1_500]  # MB
    for i, min_mb in enumerate(expected_min, start=1):
        path = os.path.join(QWEN_EDIT_BASE, "text_encoder", f"model-{i:05d}-of-00004.safetensors")
        if not (os.path.isfile(path) and os.path.getsize(path) >= min_mb * 1024 * 1024):
            return False
    vae = os.path.join(QWEN_EDIT_BASE, "vae", "diffusion_pytorch_model.safetensors")
    if not (os.path.isfile(vae) and os.path.getsize(vae) >= 200 * 1024 * 1024):
        return False
    return True


def _load_qwen_edit(progress_callback: Optional[Callable] = None):
    """Load Qwen-Image-Edit-2509 with the GGUF Q4_K_S transformer.

    Pattern mirrors scene_generator._load_wan22_gguf: the transformer is
    pulled from a single .gguf file via from_single_file with a
    GGUFQuantizationConfig; the rest of the components (text_encoder,
    tokenizer, VAE, scheduler, processor) load from the cached base repo.

    Memory math on 12GB VRAM with model_cpu_offload:
      - Q4_K_S transformer: 11.4 GB on disk -> ~7-8 GB active
      - Qwen2-VL text_encoder: 15.4 GB FP16 -> CPU-offloaded between calls
      - VAE: 254 MB
    Total VRAM at any one time stays under ~10 GB during generation.
    """
    global _qwen_edit_pipe
    if _qwen_edit_pipe is not None:
        return _qwen_edit_pipe

    _free_vram()

    try:
        from diffusers import QwenImageEditPlusPipeline, GGUFQuantizationConfig
        # Diffusers' transformer class name for Qwen-Image. Try the 2509 variant
        # first since the architecture changed slightly.
        try:
            from diffusers import QwenImageTransformer2DModel as _QwenTransformer
        except ImportError:
            from diffusers.models import QwenImageTransformer2DModel as _QwenTransformer
    except ImportError as e:
        raise RuntimeError(
            "Qwen-Image-Edit requires a recent diffusers build. Install:\n"
            "    pip install -U git+https://github.com/huggingface/diffusers.git"
        ) from e

    if progress_callback:
        progress_callback(8, 100, "Loading Qwen-Image-Edit transformer (GGUF Q4)...")

    gguf_config = GGUFQuantizationConfig(compute_dtype=torch.bfloat16)
    transformer = _QwenTransformer.from_single_file(
        QWEN_EDIT_GGUF,
        quantization_config=gguf_config,
        config=QWEN_EDIT_REPO,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
    )

    if progress_callback:
        progress_callback(20, 100, "Loading Qwen2-VL text encoder + VAE...")

    _qwen_edit_pipe = QwenImageEditPlusPipeline.from_pretrained(
        QWEN_EDIT_BASE,
        transformer=transformer,
        torch_dtype=torch.bfloat16,
    )
    # Sequential offload: swap submodules (single layers / blocks) one at a
    # time instead of whole pipeline components. Peaks at ~2-3 GB VRAM during
    # attention compute instead of trying to hold 7-8 GB resident. Slower per
    # forward pass but avoids the pathological cache-thrash we hit with
    # enable_model_cpu_offload on a 12 GB card with 27 GB working set.
    _qwen_edit_pipe.enable_sequential_cpu_offload()
    try:
        _qwen_edit_pipe.vae.enable_tiling()
    except Exception:
        pass

    if progress_callback:
        progress_callback(32, 100, "Qwen-Image-Edit ready, generating...")
    return _qwen_edit_pipe


def _pix2pix_cached() -> bool:
    """True iff a snapshot of InstructPix2Pix is already on disk (so the next
    load won't trigger a multi-GB download)."""
    try:
        from huggingface_hub import try_to_load_from_cache
        # The unet weights file — if this resolves, the rest is also cached.
        path = try_to_load_from_cache(
            INSTRUCT_PIX2PIX_REPO,
            "unet/diffusion_pytorch_model.safetensors",
        )
        return bool(path) and os.path.isfile(path)
    except Exception:
        return False


def _watch_cache_size(repo_id: str, target_gb: float, progress_callback: Callable, stop_evt) -> None:
    """Background thread: poll HF cache dir for ``repo_id`` every 2s and emit
    download progress. Approximates [10, 30] of overall job progress."""
    import time as _t
    safe_id = repo_id.replace("/", "--")
    cache_root = os.path.expanduser("~/.cache/huggingface/hub/models--" + safe_id)
    while not stop_evt.is_set():
        try:
            total = 0
            for root, _, files in os.walk(cache_root):
                for f in files:
                    try:
                        total += os.path.getsize(os.path.join(root, f))
                    except OSError:
                        pass
            mb = total / (1024 * 1024)
            target_mb = target_gb * 1024
            frac = min(1.0, mb / target_mb) if target_mb else 0
            pct = 5 + int(frac * 25)  # 5 -> 30
            try:
                progress_callback(pct, 100, f"Downloading InstructPix2Pix... {mb:.0f}/{target_mb:.0f} MB")
            except Exception:
                pass
        except Exception:
            pass
        if stop_evt.wait(2.0):
            break


def _load_pix2pix(progress_callback: Optional[Callable] = None):
    """Load InstructPix2Pix — the right tool for instruction-based edits.

    SDXL Img2Img is poor at structural edits like "raise her hand" because it
    treats the input as a noised starting point and re-denoises toward the
    prompt; pose/object changes require dedicated training. InstructPix2Pix
    (Brooks et al., CVPR 2023) is fine-tuned on (source, instruction, edited)
    triples specifically for this kind of edit. SD-1.5 backbone, ~5 GB.

    First-run download progress is reported via ``progress_callback`` from a
    background watcher thread so the job doesn't appear stuck at 0%.
    """
    global _pix2pix_pipe
    if _pix2pix_pipe is not None:
        return _pix2pix_pipe
    from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
    import threading

    _free_vram()

    cached = _pix2pix_cached()
    watcher = None
    stop_evt = threading.Event()
    if progress_callback:
        if cached:
            progress_callback(5, 100, "Loading InstructPix2Pix (cached)...")
        else:
            progress_callback(2, 100, "Downloading InstructPix2Pix (~5 GB, first run only)...")
            watcher = threading.Thread(
                target=_watch_cache_size,
                args=(INSTRUCT_PIX2PIX_REPO, 5.0, progress_callback, stop_evt),
                daemon=True,
            )
            watcher.start()
    logger.info("Loading InstructPix2Pix from %s (cached=%s)", INSTRUCT_PIX2PIX_REPO, cached)

    try:
        _pix2pix_pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            INSTRUCT_PIX2PIX_REPO,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            safety_checker=None,
        )
    finally:
        stop_evt.set()
        if watcher is not None:
            watcher.join(timeout=3)

    _pix2pix_pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(_pix2pix_pipe.scheduler.config)
    if torch.cuda.is_available():
        _pix2pix_pipe.enable_model_cpu_offload()
    if progress_callback:
        progress_callback(32, 100, "InstructPix2Pix ready, generating...")
    return _pix2pix_pipe


def _load_inpaint():
    """Load SDXL Inpaint sharing weights with the cached T2I pipeline."""
    global _inpaint_pipe
    if _inpaint_pipe is not None:
        return _inpaint_pipe
    import reelforge
    reelforge.rf_load_sdxl_pipeline()
    if reelforge._sdxl_pipe is None:
        raise RuntimeError("SDXL pipeline failed to load")
    from diffusers import StableDiffusionXLInpaintPipeline
    _inpaint_pipe = StableDiffusionXLInpaintPipeline(
        vae=reelforge._sdxl_pipe.vae,
        text_encoder=reelforge._sdxl_pipe.text_encoder,
        text_encoder_2=reelforge._sdxl_pipe.text_encoder_2,
        tokenizer=reelforge._sdxl_pipe.tokenizer,
        tokenizer_2=reelforge._sdxl_pipe.tokenizer_2,
        unet=reelforge._sdxl_pipe.unet,
        scheduler=reelforge._sdxl_pipe.scheduler,
    )
    if torch.cuda.is_available():
        _inpaint_pipe.to("cuda")
    return _inpaint_pipe


def _round_to_8(x: int) -> int:
    return max(8, (int(x) // 8) * 8)


def _resolve_size(size: str, width: Optional[int], height: Optional[int]) -> tuple:
    if width and height:
        return _round_to_8(width), _round_to_8(height)
    return SIZE_PRESETS.get(size, SIZE_PRESETS["portrait_9x16"])


def _apply_style(prompt: str, negative: str, style: Optional[str]) -> tuple:
    if not style or style not in STYLE_PRESETS:
        return prompt, negative or ""
    s = STYLE_PRESETS[style]
    return f"{prompt}, {s['positive']}", f"{negative}, {s['negative']}".strip(", ")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def text_to_image(
    prompt: str,
    negative_prompt: str = "",
    size: str = "portrait_9x16",
    width: Optional[int] = None,
    height: Optional[int] = None,
    style: Optional[str] = "photorealistic",
    seed: Optional[int] = None,
    steps: int = 30,
    guidance: float = 7.5,
    provider: str = "sdxl",
    model: Optional[str] = None,
    progress_callback: Optional[Callable] = None,
) -> str:
    """Generate an image from text. Returns absolute output path."""
    w, h = _resolve_size(size, width, height)
    if progress_callback:
        progress_callback(5, 100, f"Provider={provider} {w}x{h}")

    if provider == "fooocus":
        out = fooocus_text2image(
            prompt=prompt, negative_prompt=negative_prompt,
            width=w, height=h, seed=seed, style=style, steps=steps,
        )
        if progress_callback:
            progress_callback(100, 100, "Done.")
        return out

    if provider == "nanobanana2":
        from reelforge import rf_generate_image_nanobanana2
        out = rf_generate_image_nanobanana2(prompt)
        if progress_callback:
            progress_callback(100, 100, "Done.")
        return out

    # Default: local SDXL
    pp, nn = _apply_style(prompt, negative_prompt, style)
    import reelforge
    reelforge.rf_load_sdxl_pipeline(model)
    if reelforge._sdxl_pipe is None:
        raise RuntimeError("SDXL pipeline failed to load")

    generator = torch.Generator(device="cpu").manual_seed(seed) if seed is not None else None
    if progress_callback:
        progress_callback(20, 100, f"SDXL: {steps} steps, guidance {guidance}")

    is_turbo = (model is None) or ("turbo" in (model or "").lower())
    eff_steps = min(steps, 8) if is_turbo else steps
    eff_guidance = 0.0 if is_turbo else guidance

    out_image = reelforge._sdxl_pipe(
        prompt=pp, negative_prompt=nn,
        num_inference_steps=eff_steps, guidance_scale=eff_guidance,
        width=w, height=h, generator=generator,
    ).images[0]

    out_path = os.path.join(OUTPUT_DIR, f"t2i_{uuid4().hex[:8]}.png")
    out_image.save(out_path, format="PNG")
    if progress_callback:
        progress_callback(100, 100, "Done.")
    return out_path


def _looks_like_instruction(prompt: str) -> bool:
    """Heuristic: is this an EDIT INSTRUCTION ('raise her hand', 'make it sunset')
    vs. a NEW SCENE description ('a wizard in a tower')? Edit instructions tend
    to start with imperative verbs and reference the source ('it', 'her', 'the').
    """
    p = prompt.strip().lower()
    if not p:
        return False
    edit_starts = (
        "make ", "change ", "turn ", "replace ", "remove ", "add ", "give ", "put ",
        "raise ", "lower ", "open ", "close ", "lift ", "lift up", "drop ",
        "convert ", "transform ", "set ", "let ", "have ", "show ", "hide ",
        "paint ", "color ", "recolor ", "tint ",
    )
    if any(p.startswith(v) for v in edit_starts):
        return True
    # References to the existing subject without describing a brand new scene.
    refers = (" it ", " her ", " his ", " their ", " the ")
    if any(r in f" {p} " for r in refers) and len(p.split()) <= 12:
        return True
    return False


def image_to_image(
    image_path: str,
    prompt: str,
    negative_prompt: str = "",
    strength: float = 0.7,
    style: Optional[str] = None,
    seed: Optional[int] = None,
    steps: int = 30,
    guidance: float = 7.5,
    provider: str = "sdxl",
    technique: str = "auto",  # "auto" | "qwen_edit" | "instruct" | "redraw"
    image_guidance: float = 1.5,  # Used by InstructPix2Pix; 1.0-1.8 typical
    progress_callback: Optional[Callable] = None,
) -> str:
    """Modify an existing image with a prompt.

    Backends, picked by ``technique``:

    - ``"qwen_edit"`` — Qwen-Image-Edit-2509 (GGUF Q4_K_S). Current SOTA local
      instruction editor (Apr 2026). Best at structural edits like "raise her
      hand", "make her sit". Native ControlNet. Defaults: true_cfg=4, steps=40.
    - ``"instruct"`` — InstructPix2Pix (legacy 2023). Cheap fallback; good for
      global style edits but very poor at pose/structure changes (~5% success).
    - ``"redraw"`` — SDXL Img2Img. Style/composition shifts where the prompt
      describes the target. Tunable: ``strength``.
    - ``"auto"`` — picks qwen_edit if available, else instruct for imperative
      prompts, else redraw.
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(image_path)

    if provider == "fooocus":
        return fooocus_image2image(
            image_path=image_path, prompt=prompt, negative_prompt=negative_prompt,
            denoise_strength=strength, seed=seed, style=style,
        )

    chosen = technique
    if chosen == "auto":
        if qwen_edit_available():
            chosen = "qwen_edit"
        elif _looks_like_instruction(prompt):
            chosen = "instruct"
        else:
            chosen = "redraw"

    pp, nn = _apply_style(prompt, negative_prompt, style)

    # Qwen-Image-Edit-2509 path — current best local editor for instruction edits
    if chosen == "qwen_edit":
        if not qwen_edit_available():
            raise RuntimeError(
                "Qwen-Image-Edit-2509 weights not found at "
                f"{QWEN_EDIT_GGUF}. Download GGUF + base components first."
            )
        pipe = _load_qwen_edit(progress_callback=progress_callback)
        init = PIL.Image.open(image_path).convert("RGB")
        # 12 GB cards with sequential offload work best at 512 long-side.
        # Each doubling of resolution roughly quadruples attention cost since
        # token count grows with H*W. 1024 -> 768 -> 512 each cuts step time
        # ~2x. User can upscale the result with the dedicated Upscale mode.
        w0, h0 = init.size
        if max(w0, h0) > 512:
            scale = 512 / max(w0, h0)
            w0, h0 = int(w0 * scale), int(h0 * scale)
        w0, h0 = _round_to_8(w0), _round_to_8(h0)
        init = init.resize((w0, h0), PIL.Image.LANCZOS)

        generator = torch.Generator(device="cpu").manual_seed(seed) if seed is not None else None
        # Qwen recommends true_cfg=4.0, guidance_scale=1.0, 40 steps (per
        # the official model card). Negative prompt is just " " by default.
        true_cfg = float(guidance) if guidance and guidance > 1.0 else 4.0
        if true_cfg > 8.0:
            true_cfg = 8.0  # Higher than ~6 produces saturated artifacts in Qwen.
        eff_steps = max(steps, 30)

        if progress_callback:
            progress_callback(35, 100,
                f"Qwen-Image-Edit: true_cfg={true_cfg} steps={eff_steps} ({w0}x{h0})")

        # Per-step callback so the UI shows real progress through the 30+ steps
        # rather than freezing at 35% for several minutes.
        _qwen_start = time.time()

        def _qwen_step_cb(pipe_obj, step_index, _timestep, cb_kwargs):
            if progress_callback:
                pct = 35 + int(((step_index + 1) / eff_steps) * 60)
                elapsed = time.time() - _qwen_start
                eta = (elapsed / max(step_index + 1, 1)) * (eff_steps - step_index - 1)
                progress_callback(
                    pct, 100,
                    f"Qwen step {step_index + 1}/{eff_steps} "
                    f"(elapsed {int(elapsed)}s, ETA {int(eta)}s)",
                )
            return cb_kwargs

        out_image = pipe(
            image=[init],
            prompt=pp,
            negative_prompt=nn or " ",
            true_cfg_scale=true_cfg,
            guidance_scale=1.0,
            num_inference_steps=eff_steps,
            num_images_per_prompt=1,
            generator=generator,
            callback_on_step_end=_qwen_step_cb,
        ).images[0]
        out_path = os.path.join(OUTPUT_DIR, f"qwen_edit_{uuid4().hex[:8]}.png")
        out_image.save(out_path, format="PNG")
        if progress_callback:
            progress_callback(100, 100, "Done.")
        return out_path

    # InstructPix2Pix path
    if chosen == "instruct":
        pipe = _load_pix2pix(progress_callback=progress_callback)
        init = PIL.Image.open(image_path).convert("RGB")
        # InstructPix2Pix is SD-1.5 — keep dims modest. 512 short-side is sweet spot.
        w, h = init.size
        if max(w, h) > 768:
            scale = 768 / max(w, h)
            w, h = int(w * scale), int(h * scale)
        w, h = _round_to_8(w), _round_to_8(h)
        init = init.resize((w, h), PIL.Image.LANCZOS)

        generator = torch.Generator(device="cpu").manual_seed(seed) if seed is not None else None
        if progress_callback:
            progress_callback(15, 100,
                f"InstructPix2Pix: text={guidance} image={image_guidance}")
        # InstructPix2Pix has its own param names: guidance_scale = text strength,
        # image_guidance_scale = how strongly to preserve the source.
        out_image = pipe(
            prompt=pp,
            image=init,
            num_inference_steps=max(steps, 30),
            guidance_scale=float(guidance) if guidance > 0 else 7.5,
            image_guidance_scale=float(image_guidance),
            generator=generator,
        ).images[0]
        out_path = os.path.join(OUTPUT_DIR, f"edit_{uuid4().hex[:8]}.png")
        out_image.save(out_path, format="PNG")
        if progress_callback:
            progress_callback(100, 100, "Done.")
        return out_path

    # SDXL Img2Img redraw path
    pipe = _load_img2img()
    init = PIL.Image.open(image_path).convert("RGB")
    w, h = init.size
    w, h = _round_to_8(w), _round_to_8(h)
    init = init.resize((w, h), PIL.Image.LANCZOS)

    generator = torch.Generator(device="cpu").manual_seed(seed) if seed is not None else None
    if progress_callback:
        progress_callback(15, 100, f"SDXL Img2Img strength={strength}")

    out_image = pipe(
        prompt=pp, negative_prompt=nn,
        image=init, strength=float(strength),
        num_inference_steps=steps, guidance_scale=guidance,
        generator=generator,
    ).images[0]

    out_path = os.path.join(OUTPUT_DIR, f"i2i_{uuid4().hex[:8]}.png")
    out_image.save(out_path, format="PNG")
    if progress_callback:
        progress_callback(100, 100, "Done.")
    return out_path


def inpaint(
    image_path: str,
    mask_path: str,
    prompt: str,
    negative_prompt: str = "",
    style: Optional[str] = None,
    seed: Optional[int] = None,
    steps: int = 30,
    guidance: float = 7.5,
    invert_mask: bool = False,
    progress_callback: Optional[Callable] = None,
) -> str:
    """Locally edit a region of the image. The mask is white = edit, black = keep
    by convention; pass ``invert_mask=True`` if your mask is the other way around."""
    if not (os.path.isfile(image_path) and os.path.isfile(mask_path)):
        raise FileNotFoundError("image_path or mask_path missing")

    pp, nn = _apply_style(prompt, negative_prompt, style)
    pipe = _load_inpaint()
    init = PIL.Image.open(image_path).convert("RGB")
    mask = PIL.Image.open(mask_path).convert("L")
    if invert_mask:
        mask = PIL.ImageOps.invert(mask)

    w, h = _round_to_8(init.size[0]), _round_to_8(init.size[1])
    init = init.resize((w, h), PIL.Image.LANCZOS)
    mask = mask.resize((w, h), PIL.Image.LANCZOS)

    generator = torch.Generator(device="cpu").manual_seed(seed) if seed is not None else None
    if progress_callback:
        progress_callback(15, 100, "Inpainting...")

    out_image = pipe(
        prompt=pp, negative_prompt=nn,
        image=init, mask_image=mask,
        num_inference_steps=steps, guidance_scale=guidance,
        generator=generator,
    ).images[0]

    out_path = os.path.join(OUTPUT_DIR, f"inpaint_{uuid4().hex[:8]}.png")
    out_image.save(out_path, format="PNG")
    if progress_callback:
        progress_callback(100, 100, "Done.")
    return out_path


def variation(
    image_path: str,
    prompt: Optional[str] = None,
    strength: float = 0.45,
    style: Optional[str] = None,
    seed: Optional[int] = None,
    progress_callback: Optional[Callable] = None,
) -> str:
    """"More like this" — reuse the source image with low denoising strength.
    Without a prompt, derives a generic structure-preserving prompt."""
    p = prompt or "high quality, detailed, sharp focus"
    return image_to_image(
        image_path=image_path, prompt=p, negative_prompt="",
        strength=float(strength), style=style, seed=seed,
        progress_callback=progress_callback,
    )


def upscale(
    image_path: str,
    scale: int = 2,
    method: str = "lanczos",
    progress_callback: Optional[Callable] = None,
) -> str:
    """2x or 4x upscale. ``method='realesrgan'`` falls back to lanczos when
    the package isn't installed (``upscale_image`` already handles that)."""
    if progress_callback:
        progress_callback(20, 100, f"Upscaling {scale}x via {method}")
    from upscaler import upscale_image as _upscale
    out_path = os.path.join(OUTPUT_DIR, f"up_{scale}x_{uuid4().hex[:8]}.png")
    _upscale(image_path, scale=scale, method=method, output_path=out_path)
    if progress_callback:
        progress_callback(100, 100, "Done.")
    return out_path


def remove_background(
    image_path: str,
    progress_callback: Optional[Callable] = None,
) -> str:
    """Cut out the foreground subject; transparent PNG out. Requires ``rembg``."""
    try:
        from rembg import remove as _rembg_remove
    except ImportError as e:
        raise RuntimeError(
            "Background removal requires the rembg package. Install with:\n"
            "    pip install rembg[gpu]"
        ) from e

    if progress_callback:
        progress_callback(20, 100, "Removing background...")
    with open(image_path, "rb") as f:
        out = _rembg_remove(f.read())
    out_path = os.path.join(OUTPUT_DIR, f"nobg_{uuid4().hex[:8]}.png")
    with open(out_path, "wb") as f:
        f.write(out)
    if progress_callback:
        progress_callback(100, 100, "Done.")
    return out_path


# ---------------------------------------------------------------------------
# LLM prompt enhancement
# ---------------------------------------------------------------------------

def _caption_image(image_path: str) -> str:
    """Generate a short caption of the image using BLIP. Lazy-loads on first call.

    Used by Enhance-with-AI in edit mode so the LLM has visual context for the
    rewrite. Returns "" on failure (we still want the LLM call to proceed)."""
    global _blip_processor, _blip_model
    try:
        if _blip_model is None:
            from transformers import BlipProcessor, BlipForConditionalGeneration
            model_id = "Salesforce/blip-image-captioning-base"
            logger.info("Loading BLIP captioner: %s", model_id)
            _blip_processor = BlipProcessor.from_pretrained(model_id)
            _blip_model = BlipForConditionalGeneration.from_pretrained(model_id)
            if torch.cuda.is_available():
                _blip_model = _blip_model.to("cuda")

        img = PIL.Image.open(image_path).convert("RGB")
        inputs = _blip_processor(img, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        with torch.no_grad():
            out = _blip_model.generate(**inputs, max_new_tokens=50)
        caption = _blip_processor.decode(out[0], skip_special_tokens=True)
        return caption.strip()
    except Exception as e:
        logger.warning("BLIP captioning failed: %s", e)
        return ""


def _configure_llm_from_config():
    """Match what /generate/script and /generate/story do: pull backend +
    model from config.json before invoking the LLM. Without this, the
    llm_provider module starts with backend='llamacpp', which fails when
    llama_cpp isn't installed even though Ollama is configured."""
    from mpv2.llm_provider import select_backend, select_model
    from reelforge import load_config
    cfg = load_config()
    backend = cfg.get("llm_backend", "ollama")
    select_backend(backend)
    if backend == "ollama":
        select_model(cfg.get("ollama_model") or "llama3.2:3b")
    else:
        gguf = cfg.get("gguf_model", "")
        if gguf:
            select_model(gguf)


def enhance_prompt(
    prompt: str,
    style: Optional[str] = None,
    mode: str = "expand",
    image_path: Optional[str] = None,
    negative_prompt: Optional[str] = None,
) -> dict:
    """Use the project LLM to rewrite a sparse prompt into a detailed one,
    AND produce a matching negative prompt.

    Args:
      prompt: The user's positive prompt to enhance.
      style: Optional preset name (photorealistic, cinematic, etc.) for tone.
      mode: "expand" (add detail) or "shorten" (compress to essentials).
      image_path: Optional source image path (Edit / Variation / Inpaint).
        When provided, BLIP captions the image and the LLM is told to keep
        scene-grounded — i.e. enhance the EDIT instruction in context, not
        invent a brand-new scene.
      negative_prompt: Optional existing negative the user has typed, so the
        LLM can refine rather than overwrite it.

    Returns: ``{"enhanced": str, "negative": str, "image_caption": str}``.

    Raises RuntimeError if the LLM call fails — surfaced verbatim by the API
    so the UI shows the real error rather than silently echoing the input.
    """
    raw = (prompt or "").strip()
    if not raw:
        return {"enhanced": "", "negative": negative_prompt or "", "image_caption": ""}

    style_hint = STYLE_PRESETS.get(style or "", {}).get("positive", "")
    style_neg = STYLE_PRESETS.get(style or "", {}).get("negative", "")

    image_caption = ""
    if image_path:
        image_caption = _caption_image(image_path)

    if mode == "negative":
        instruction = (
            "The user has a positive prompt and wants a STRONG NEGATIVE prompt for "
            "image generation. List 15-25 comma-separated artifact and quality terms "
            "tailored to this prompt: things that commonly go wrong (anatomy, hands, "
            "faces, lighting, composition) plus mood/style anti-patterns. Do NOT "
            "rewrite the positive prompt — keep it unchanged."
        )
        if style_neg:
            instruction += f"\nInclude these style-aware terms: {style_neg}"
    elif mode == "shorten":
        instruction = (
            "Compress the user's image-generation prompt to its visual essentials "
            "(subject, action, mood) in 12 words or fewer. Then write a short "
            "negative prompt (8-15 comma-separated terms) listing artifacts to avoid."
        )
    else:
        if image_caption:
            instruction = (
                "Below is a SOURCE IMAGE (described by an automatic captioner) and "
                "the user's EDIT INSTRUCTION. Rewrite the edit instruction into a "
                "single dense paragraph (35-60 words) that describes the FINAL "
                "image after applying the edit. Keep the subject and setting from "
                "the source image; only change what the edit instruction changes. "
                "Add specific visual detail, lighting, composition, mood. "
                "Then write a strong negative prompt (12-20 comma-separated "
                "artifact terms) tailored to the result."
            )
        else:
            instruction = (
                "Rewrite the user's image-generation prompt into a single dense "
                "paragraph (35-60 words). Add specific visual detail, lighting, "
                "composition, mood, camera lens hints. Do NOT change the subject. "
                "Then write a strong negative prompt (12-20 comma-separated artifact "
                "terms) tailored to that prompt."
            )
        if style_hint:
            instruction += f"\nStyle direction: {style_hint}"

    instruction += (
        "\n\nReturn ONLY a JSON object with two fields:\n"
        '  {"prompt": "<rewritten positive prompt>", '
        '"negative": "<comma-separated negative terms>"}\n'
        "No preface. No code fences. JSON only."
    )

    parts = [instruction]
    if image_caption:
        parts.append(f"SOURCE IMAGE: {image_caption}")
    if negative_prompt and negative_prompt.strip():
        parts.append(f"User's existing negative (refine, don't ignore): {negative_prompt.strip()}")
    parts.append(f"USER PROMPT: {raw}")
    full_prompt = "\n\n".join(parts)

    _configure_llm_from_config()
    from mpv2.llm_provider import generate_text
    out = (generate_text(full_prompt) or "").strip()

    if not out:
        raise RuntimeError("LLM returned empty response")

    # Try to parse JSON. LLMs often wrap output in code fences or add prose;
    # extract the first {...} block as a fallback.
    cleaned = out
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").strip()
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].lstrip()

    parsed = None
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        import re
        m = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if m:
            try:
                parsed = json.loads(m.group(0))
            except json.JSONDecodeError:
                parsed = None

    if isinstance(parsed, dict) and ("prompt" in parsed or "negative" in parsed):
        enhanced = str(parsed.get("prompt", "")).strip().strip('"\'')
        negative = str(parsed.get("negative") or parsed.get("negative_prompt") or "").strip().strip('"\'')
    else:
        # LLM didn't produce JSON — treat the whole reply as the relevant field.
        cleaned_text = cleaned
        for prefix in ("Enhanced prompt:", "Rewritten prompt:", "Prompt:",
                       "Negative prompt:", "Negative:", "Output:"):
            if cleaned_text.lower().startswith(prefix.lower()):
                cleaned_text = cleaned_text[len(prefix):].strip()
        if mode == "negative":
            enhanced = ""
            negative = cleaned_text
        else:
            enhanced = cleaned_text
            negative = style_neg or (negative_prompt or "")

    # In mode=negative the prompt is intentionally left untouched; preserve it.
    if mode == "negative":
        enhanced = raw
        if not negative:
            raise RuntimeError("LLM response did not contain a usable negative prompt")
    elif not enhanced:
        raise RuntimeError("LLM response did not contain a usable prompt")

    # Merge style-preset negative terms in if the LLM's negative is sparse.
    if style_neg and len(negative) < 30:
        negative = (negative + ", " + style_neg).strip(", ")

    return {
        "enhanced": enhanced,
        "negative": negative,
        "image_caption": image_caption,
    }


# ---------------------------------------------------------------------------
# History — last N images on disk
# ---------------------------------------------------------------------------

def list_history(limit: int = 50) -> list:
    """Return recent image generations under .mp/images/, newest first."""
    if not os.path.isdir(OUTPUT_DIR):
        return []
    files = []
    for f in os.listdir(OUTPUT_DIR):
        if not f.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
            continue
        full = os.path.join(OUTPUT_DIR, f)
        try:
            files.append((full, os.path.getmtime(full), os.path.getsize(full)))
        except OSError:
            pass
    files.sort(key=lambda x: x[1], reverse=True)
    return [
        {"filename": os.path.basename(p), "path": p, "size_bytes": sz, "mtime": mt}
        for p, mt, sz in files[:limit]
    ]


# ---------------------------------------------------------------------------
# Catalog — what the UI should show
# ---------------------------------------------------------------------------

def default_negative_for(style: Optional[str]) -> str:
    """Return a sensible default negative prompt for the given style.
    Used by the UI to pre-fill the box and by enhance mode='negative' as a
    floor when the LLM refuses or returns nothing."""
    if style and style in DEFAULT_NEGATIVES:
        return DEFAULT_NEGATIVES[style]
    return DEFAULT_NEGATIVES["_base"]


def catalog() -> dict:
    """Return everything the UI needs to populate selectors:
    providers, models, sizes, styles, optional Fooocus styles."""
    providers = [
        {"id": "sdxl", "name": "Local SDXL", "description": "Runs on your GPU. T2I/I2I/inpaint."},
    ]
    if fooocus_available():
        providers.append({
            "id": "fooocus",
            "name": "Fooocus API",
            "description": f"Remote at {_fooocus_url()}. T2I + style presets.",
        })
    # nanobanana2 only listed when an API key is configured.
    try:
        from mpv2.config import get_nanobanana2_api_key
        if get_nanobanana2_api_key():
            providers.append({"id": "nanobanana2", "name": "Gemini (NanoBanana2)", "description": "Hosted Gemini image API."})
    except Exception:
        pass

    qwen_ready = qwen_edit_available()
    edit_techniques = [
        {"id": "auto", "name": "Auto",
         "description": "Pick the best available editor based on prompt + downloaded models"},
    ]
    if qwen_ready:
        edit_techniques.append({
            "id": "qwen_edit", "name": "Qwen-Image-Edit",
            "description": "SOTA local editor (Apr 2026) — best for structural edits like 'raise her hand'",
        })
    else:
        edit_techniques.append({
            "id": "qwen_edit", "name": "Qwen-Image-Edit (downloading)",
            "description": "Weights not yet available — defaults to instruct/redraw",
        })
    edit_techniques.extend([
        {"id": "instruct", "name": "Instruction (legacy)",
         "description": "InstructPix2Pix 2023 — fast, good at global style edits, weak at structure"},
        {"id": "redraw", "name": "Redraw",
         "description": "SDXL Img2Img — describe the target image; source is a starting point"},
    ])

    return {
        "providers": providers,
        "models": list_local_sdxl_models(),
        "sizes": [{"id": k, "name": k.replace("_", " ").title(), "width": w, "height": h} for k, (w, h) in SIZE_PRESETS.items()],
        "styles": list(STYLE_PRESETS.keys()),
        "fooocus_styles": fooocus_styles(),
        "edit_techniques": edit_techniques,
        "qwen_edit_available": qwen_ready,
        "default_negatives": DEFAULT_NEGATIVES,
        "rembg_available": _rembg_importable(),
        "realesrgan_available": _realesrgan_importable(),
    }


def _rembg_importable() -> bool:
    try:
        import rembg  # noqa: F401
        return True
    except ImportError:
        return False


def _realesrgan_importable() -> bool:
    try:
        import realesrgan  # noqa: F401
        return True
    except ImportError:
        return False
