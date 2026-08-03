"""
Scene Generator - Enterprise-quality local video generation.

Engines (priority high -> low for "quality"; "draft" prefers LTX, "balanced" prefers Hunyuan):
  - Wan 2.2 14B I2V GGUF Q3  (quality tier, 85-90% of Runway/Kling)
  - HunyuanVideo 1.5 I2V 480p GGUF Q4  (balanced tier, 8.3B, lighter)
  - LTX-2.3 distilled GGUF Q3  (draft tier, fastest)
  - Wan VACE 1.3B  (fallback, local)
  - Wan T2V 1.3B  (last-resort, text-only)

All pipelines use SDXL (scene image) -> I2V (animate). GGUF quantization +
model CPU offload lets them fit in 12GB VRAM.
"""

import os
import gc
import math
import logging
import subprocess
import time
import torch
import PIL.Image
from uuid import uuid4
from typing import Optional, Callable

logger = logging.getLogger(__name__)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(ROOT_DIR, ".mp")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Wan 2.2 14B I2V GGUF (quality tier) ---
GGUF_DIR = os.path.join(ROOT_DIR, ".models", "wan22-i2v-14b-gguf")
GGUF_HIGH = os.path.join(GGUF_DIR, "HighNoise", "Wan2.2-I2V-A14B-HighNoise-Q3_K_S.gguf")
GGUF_LOW = os.path.join(GGUF_DIR, "LowNoise", "Wan2.2-I2V-A14B-LowNoise-Q3_K_S.gguf")
WAN22_CONFIG = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"  # For transformer config (not weights)

# --- HunyuanVideo 1.5 I2V 480p GGUF (balanced tier) ---
HUNYUAN15_GGUF_DIR = os.path.join(ROOT_DIR, ".models", "hunyuan-15-i2v-480p-gguf")
HUNYUAN15_GGUF = os.path.join(HUNYUAN15_GGUF_DIR, "hunyuanvideo1.5_480p_i2v_cfg_distilled-Q4_K_M.gguf")
HUNYUAN15_BASE = os.path.join(ROOT_DIR, ".models", "hunyuan-15-base")  # text_encoder + VAE + scheduler
HUNYUAN15_CONFIG_REPO = "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v_step_distilled"

# --- LTX-2.3 distilled GGUF (draft tier) ---
LTX23_GGUF_DIR = os.path.join(ROOT_DIR, ".models", "ltx-23-distilled-gguf")
LTX23_GGUF = os.path.join(LTX23_GGUF_DIR, "LTX-2.3-distilled-Q3_K_M.gguf")
LTX23_BASE = os.path.join(ROOT_DIR, ".models", "ltx-23-base")  # text_encoder + VAE + scheduler
LTX23_CONFIG_REPO = "CalamitousFelicitousness/LTX-2.3-distilled-Diffusers"

# --- VACE 1.3B fallback (local) ---
VACE_LOCAL = os.path.join(ROOT_DIR, ".models", "vace-1.3b")

_i2v_pipe = None
_i2v_engine = None  # Track which engine is currently loaded so we know when to swap

# Each engine has an architectural cap on frames-per-call. Longer durations
# must be produced by chaining: generate clip N, take its last frame as the
# image input for clip N+1, repeat. ENGINE_MAX_FRAMES sets the per-call cap.
ENGINE_MAX_FRAMES = {
    "wan22_14b_gguf": 49,         # 49 / 16 fps = 3.06s per clip
    "hunyuan15_i2v_gguf": 65,     # 65 / 16 fps = 4.06s per clip
    "ltx23_distilled_gguf": 121,  # 121 / 24 fps = 5.04s per clip
    "vace_1.3b": 81,              # 81 / 16 fps = 5.06s per clip
}

ENGINE_NATIVE_FPS = {
    "wan22_14b_gguf": 16,
    "hunyuan15_i2v_gguf": 16,
    "ltx23_distilled_gguf": 24,
    "vace_1.3b": 16,
}


def _free_vram():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _unload_sdxl():
    try:
        import reelforge
        if reelforge._sdxl_pipe is not None:
            del reelforge._sdxl_pipe
            reelforge._sdxl_pipe = None
    except Exception:
        pass
    _free_vram()


def _unload_i2v():
    global _i2v_pipe, _i2v_engine
    if _i2v_pipe is not None:
        del _i2v_pipe
        _i2v_pipe = None
        _i2v_engine = None
    _free_vram()


def gguf_available() -> bool:
    """Check if Wan 2.2 14B GGUF files are downloaded."""
    return os.path.isfile(GGUF_HIGH) and os.path.isfile(GGUF_LOW)


def _big_file(path: str, min_mb: int = 50) -> bool:
    """True only if the file exists and is at least ``min_mb`` MB.
    Guards against 15-byte HF 404 placeholders appearing at expected paths.
    """
    try:
        return os.path.isfile(path) and os.path.getsize(path) > min_mb * 1024 * 1024
    except OSError:
        return False


def _all_shards_present(dirpath: str, n_shards: int, min_mb: int = 500) -> bool:
    """True when all {1..n}-of-n.safetensors shards exist and are real (>min_mb)."""
    return all(
        _big_file(os.path.join(dirpath, f"model-{i:05d}-of-{n_shards:05d}.safetensors"), min_mb)
        for i in range(1, n_shards + 1)
    )


def hunyuan15_available() -> bool:
    """Check if HunyuanVideo 1.5 I2V GGUF + base components are downloaded."""
    if not _big_file(HUNYUAN15_GGUF, min_mb=500):
        return False
    te_dir = os.path.join(HUNYUAN15_BASE, "text_encoder")
    vae = os.path.join(HUNYUAN15_BASE, "vae", "diffusion_pytorch_model.safetensors")
    return _all_shards_present(te_dir, 3, min_mb=500) and _big_file(vae, min_mb=100)


def ltx23_available() -> bool:
    """Check if LTX-2.3 distilled GGUF + base components are downloaded."""
    if not _big_file(LTX23_GGUF, min_mb=500):
        return False
    vae = os.path.join(LTX23_BASE, "vae", "diffusion_pytorch_model.safetensors")
    if not _big_file(vae, min_mb=100):
        return False
    te_dir = os.path.join(LTX23_BASE, "text_encoder")
    # LTX-2.3 text_encoder = 5 shards of ~4.9GB each (~23GB total)
    return _all_shards_present(te_dir, 5, min_mb=500)


def vace_available() -> bool:
    """Check if VACE 1.3B local model is available."""
    if not os.path.isdir(VACE_LOCAL):
        return False
    s1 = os.path.join(VACE_LOCAL, "transformer", "diffusion_pytorch_model-00001-of-00002.safetensors")
    s2 = os.path.join(VACE_LOCAL, "transformer", "diffusion_pytorch_model-00002-of-00002.safetensors")
    t1 = os.path.join(VACE_LOCAL, "text_encoder", "model-00001-of-00003.safetensors")
    return all(os.path.isfile(f) for f in [s1, s2, t1])


# Engine tier labels shown in UI
ENGINES = {
    "wan22_14b_gguf": "Wan 2.2 14B (quality)",
    "hunyuan15_i2v_gguf": "HunyuanVideo 1.5 (balanced)",
    "ltx23_distilled_gguf": "LTX-2.3 distilled (draft)",
    "vace_1.3b": "Wan VACE 1.3B (fallback)",
    "t2v_1.3b": "Wan T2V 1.3B (last resort)",
}


def list_available_engines() -> list:
    """Return list of {id, name, available} for UI.

    All engines require CUDA — GGUF quantization and enable_model_cpu_offload
    both need a GPU device to offload to. On CPU-only systems every engine
    is reported unavailable.
    """
    has_cuda = torch.cuda.is_available()
    checks = [
        ("wan22_14b_gguf", has_cuda and gguf_available()),
        ("hunyuan15_i2v_gguf", has_cuda and hunyuan15_available()),
        ("ltx23_distilled_gguf", has_cuda and ltx23_available()),
        ("vace_1.3b", has_cuda and vace_available()),
        ("t2v_1.3b", has_cuda),
    ]
    return [{"id": eid, "name": ENGINES[eid], "available": avail} for eid, avail in checks]


def get_available_engine(preferred: Optional[str] = None) -> str:
    """Return the engine to use. If ``preferred`` is given and available, use it.
    Otherwise fall back to best available: Wan 2.2 > Hunyuan 1.5 > LTX-2.3 > VACE > T2V.

    Raises RuntimeError on CPU-only systems — none of these engines are viable
    without a CUDA GPU (weeks-long jobs at best, immediate crashes at worst).
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Video generation requires a CUDA GPU. Install NVIDIA drivers or "
            "use an image-only workflow."
        )
    # Preferred engine override (from UI or env)
    if preferred and preferred != "auto":
        if preferred == "wan22_14b_gguf" and gguf_available():
            return preferred
        if preferred == "hunyuan15_i2v_gguf" and hunyuan15_available():
            return preferred
        if preferred == "ltx23_distilled_gguf" and ltx23_available():
            return preferred
        if preferred == "vace_1.3b" and vace_available():
            return preferred
        if preferred == "t2v_1.3b":
            return preferred
        # Requested engine not available — fall through to auto-selection

    # Quality-preset shortcut
    if preferred == "quality" and gguf_available():
        return "wan22_14b_gguf"
    if preferred == "balanced" and hunyuan15_available():
        return "hunyuan15_i2v_gguf"
    if preferred == "draft" and ltx23_available():
        return "ltx23_distilled_gguf"

    # Auto: best available
    if gguf_available():
        return "wan22_14b_gguf"
    if hunyuan15_available():
        return "hunyuan15_i2v_gguf"
    if ltx23_available():
        return "ltx23_distilled_gguf"
    if vace_available():
        return "vace_1.3b"
    return "t2v_1.3b"


def _set_engine(engine_id: str):
    """Track which engine is loaded; swap if different."""
    global _i2v_engine
    if _i2v_engine is not None and _i2v_engine != engine_id:
        _unload_i2v()
    _i2v_engine = engine_id


def _load_wan22_gguf(progress_callback: Optional[Callable] = None):
    """Load Wan 2.2 14B I2V with GGUF quantization.

    Avoids downloading the full 14B FP16 model by reusing text_encoder/tokenizer
    from the already-cached Wan 2.1 T2V 1.3B model, and loading only the GGUF
    transformers + VAE config from the 2.2 14B repo.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("Wan 2.2 14B requires a CUDA GPU (GGUF + CPU offload need a device).")
    global _i2v_pipe
    _set_engine("wan22_14b_gguf")
    if _i2v_pipe is not None:
        return _i2v_pipe

    _unload_sdxl()

    if progress_callback:
        progress_callback(10, 100, "Loading Wan 2.2 14B I2V (GGUF Q3)...")

    from diffusers import WanImageToVideoPipeline, WanTransformer3DModel, AutoencoderKLWan, GGUFQuantizationConfig
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
    from transformers import UMT5EncoderModel, AutoTokenizer, CLIPVisionModelWithProjection, CLIPImageProcessor

    gguf_config = GGUFQuantizationConfig(compute_dtype=torch.bfloat16)

    # Use 1.3B T2V repo for shared components (text_encoder, tokenizer, VAE, scheduler)
    # These are identical across all Wan variants and already cached locally
    T2V_REPO = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"

    if progress_callback:
        progress_callback(12, 100, "Loading text encoder from cached 1.3B model...")

    tokenizer = AutoTokenizer.from_pretrained(T2V_REPO, subfolder="tokenizer")
    text_encoder = UMT5EncoderModel.from_pretrained(T2V_REPO, subfolder="text_encoder", torch_dtype=torch.bfloat16)

    if progress_callback:
        progress_callback(18, 100, "Loading HighNoise GGUF transformer (6.5GB)...")

    transformer = WanTransformer3DModel.from_single_file(
        GGUF_HIGH,
        quantization_config=gguf_config,
        config=WAN22_CONFIG,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
    )

    if progress_callback:
        progress_callback(24, 100, "Loading LowNoise GGUF transformer (6.5GB)...")

    transformer_2 = WanTransformer3DModel.from_single_file(
        GGUF_LOW,
        quantization_config=gguf_config,
        config=WAN22_CONFIG,
        subfolder="transformer_2",
        torch_dtype=torch.bfloat16,
    )

    if progress_callback:
        progress_callback(28, 100, "Loading VAE and assembling pipeline...")

    vae = AutoencoderKLWan.from_pretrained(T2V_REPO, subfolder="vae", torch_dtype=torch.bfloat16)
    scheduler = UniPCMultistepScheduler.from_pretrained(T2V_REPO, subfolder="scheduler")

    # Load CLIP image encoder for I2V conditioning
    try:
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            WAN22_CONFIG, subfolder="image_encoder", torch_dtype=torch.float16
        )
        image_processor = CLIPImageProcessor.from_pretrained(WAN22_CONFIG, subfolder="image_processor")
    except Exception:
        # Image encoder may not be separate — pipeline will handle it
        image_encoder = None
        image_processor = None

    # Assemble pipeline manually (no from_pretrained needed)
    _i2v_pipe = WanImageToVideoPipeline(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        transformer=transformer,
        transformer_2=transformer_2,
        vae=vae,
        scheduler=scheduler,
        image_encoder=image_encoder,
        image_processor=image_processor,
    )
    _i2v_pipe.enable_model_cpu_offload()
    _i2v_pipe.vae.enable_tiling()

    if progress_callback:
        progress_callback(32, 100, "Wan 2.2 14B GGUF loaded.")

    return _i2v_pipe


def _load_hunyuan15_gguf(progress_callback: Optional[Callable] = None):
    """Load HunyuanVideo 1.5 I2V 480p with GGUF Q4 quantization.

    Uses the CFG-distilled GGUF transformer + base components (text_encoder/VAE)
    from the diffusers-ready community repo at .models/hunyuan-15-base/.

    8.3B params with Q4_K_M fits in 12GB VRAM via enable_model_cpu_offload.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("HunyuanVideo 1.5 requires a CUDA GPU (GGUF + CPU offload need a device).")
    global _i2v_pipe
    _set_engine("hunyuan15_i2v_gguf")
    if _i2v_pipe is not None:
        return _i2v_pipe

    _unload_sdxl()

    if progress_callback:
        progress_callback(10, 100, "Loading HunyuanVideo 1.5 I2V (GGUF Q4)...")

    # Diffusers class name changed between versions; prefer the 15-specific one.
    try:
        from diffusers import HunyuanVideo15ImageToVideoPipeline, HunyuanVideo15Transformer3DModel
        PipeCls = HunyuanVideo15ImageToVideoPipeline
        TfmCls = HunyuanVideo15Transformer3DModel
    except ImportError:
        # Older diffusers exposed HunyuanVideo (original) but not 1.5 — escalate clearly.
        raise RuntimeError(
            "HunyuanVideo 1.5 requires diffusers with HunyuanVideo15ImageToVideoPipeline. "
            "Update diffusers: pip install -U git+https://github.com/huggingface/diffusers.git"
        )
    from diffusers import GGUFQuantizationConfig

    if progress_callback:
        progress_callback(14, 100, "Loading GGUF transformer (Q4, 4.7GB)...")

    gguf_config = GGUFQuantizationConfig(compute_dtype=torch.bfloat16)
    transformer = TfmCls.from_single_file(
        HUNYUAN15_GGUF,
        quantization_config=gguf_config,
        config=HUNYUAN15_CONFIG_REPO,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
    )

    if progress_callback:
        progress_callback(22, 100, "Loading base components (text encoder, VAE)...")

    # from_pretrained reads model_index.json and loads each subfolder, but we
    # want to override the transformer with the GGUF we just loaded.
    _i2v_pipe = PipeCls.from_pretrained(
        HUNYUAN15_BASE,
        transformer=transformer,
        torch_dtype=torch.bfloat16,
    )
    _i2v_pipe.enable_model_cpu_offload()
    try:
        _i2v_pipe.vae.enable_tiling()
    except Exception:
        pass

    if progress_callback:
        progress_callback(30, 100, "HunyuanVideo 1.5 GGUF loaded.")

    return _i2v_pipe


def _load_ltx23_gguf(progress_callback: Optional[Callable] = None):
    """Load LTX-2.3 distilled with GGUF Q3 quantization.

    LTX-2.3 has no upstream diffusers support yet (pending model_index.json);
    uses CalamitousFelicitousness's community diffusers conversion at
    .models/ltx-23-base/ for base components + GGUF transformer.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("LTX-2.3 requires a CUDA GPU (GGUF + CPU offload need a device).")
    global _i2v_pipe
    _set_engine("ltx23_distilled_gguf")
    if _i2v_pipe is not None:
        return _i2v_pipe

    _unload_sdxl()

    if progress_callback:
        progress_callback(10, 100, "Loading LTX-2.3 distilled (GGUF Q3)...")

    # LTX pipeline classes — names differ across diffusers versions.
    # Try newest first, fall back to older names.
    PipeCls = None
    TfmCls = None
    try:
        from diffusers import LTXImageToVideoPipeline, LTXVideoTransformer3DModel
        PipeCls = LTXImageToVideoPipeline
        TfmCls = LTXVideoTransformer3DModel
    except ImportError:
        try:
            from diffusers import LTXPipeline, LTXVideoTransformer3DModel
            PipeCls = LTXPipeline
            TfmCls = LTXVideoTransformer3DModel
        except ImportError:
            raise RuntimeError(
                "LTX-2.3 requires diffusers with LTXImageToVideoPipeline. "
                "Update diffusers: pip install -U git+https://github.com/huggingface/diffusers.git"
            )
    from diffusers import GGUFQuantizationConfig

    if progress_callback:
        progress_callback(14, 100, "Loading GGUF transformer (Q3, 14GB)...")

    gguf_config = GGUFQuantizationConfig(compute_dtype=torch.bfloat16)
    transformer = TfmCls.from_single_file(
        LTX23_GGUF,
        quantization_config=gguf_config,
        config=LTX23_CONFIG_REPO,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
    )

    if progress_callback:
        progress_callback(22, 100, "Loading base components (T5 encoder, VAE)...")

    _i2v_pipe = PipeCls.from_pretrained(
        LTX23_BASE,
        transformer=transformer,
        torch_dtype=torch.bfloat16,
    )
    _i2v_pipe.enable_model_cpu_offload()
    try:
        _i2v_pipe.vae.enable_tiling()
    except Exception:
        pass

    if progress_callback:
        progress_callback(30, 100, "LTX-2.3 distilled GGUF loaded.")

    return _i2v_pipe


def _load_vace(progress_callback: Optional[Callable] = None):
    """Load VACE 1.3B as fallback."""
    if not torch.cuda.is_available():
        raise RuntimeError("VACE 1.3B requires a CUDA GPU (CPU offload needs a device).")
    global _i2v_pipe
    _set_engine("vace_1.3b")
    if _i2v_pipe is not None:
        return _i2v_pipe

    _unload_sdxl()

    if progress_callback:
        progress_callback(15, 100, "Loading VACE 1.3B...")

    from diffusers import WanVACEPipeline, AutoencoderKLWan
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

    vae = AutoencoderKLWan.from_pretrained(VACE_LOCAL, subfolder="vae", torch_dtype=torch.float32)
    _i2v_pipe = WanVACEPipeline.from_pretrained(VACE_LOCAL, vae=vae, torch_dtype=torch.bfloat16)
    _i2v_pipe.scheduler = UniPCMultistepScheduler.from_config(_i2v_pipe.scheduler.config, flow_shift=3.0)
    _i2v_pipe.enable_model_cpu_offload()
    _i2v_pipe.vae.enable_tiling()

    if progress_callback:
        progress_callback(30, 100, "VACE 1.3B loaded.")

    return _i2v_pipe


def generate_scene_image(prompt, negative_prompt="", width=832, height=480, seed=None):
    """Generate a scene image using SDXL Turbo."""
    _unload_i2v()
    from reelforge import rf_generate_image
    return rf_generate_image(prompt=prompt, provider="sdxl_turbo", style="photorealistic",
                             aspect_ratio="16:9", seed=seed)


def animate_wan22_gguf(image_path, prompt, negative_prompt="", height=480, width=832,
                       num_frames=49, num_inference_steps=30, guidance_scale=3.5,
                       fps=16, seed=None, progress_callback=None):
    """Animate using Wan 2.2 14B I2V GGUF — best quality."""
    pipe = _load_wan22_gguf(progress_callback)

    image = PIL.Image.open(image_path).convert("RGB").resize((width, height), PIL.Image.LANCZOS)

    generator = torch.Generator(device="cpu")
    if seed is not None:
        generator.manual_seed(seed)

    neg = negative_prompt or "low quality, blurry, distorted, worst quality, static, no motion, ugly, deformed"
    start = time.time()

    def step_cb(pipe_obj, step_index, timestep, cb_kwargs):
        if progress_callback:
            pct = 35 + int((step_index / num_inference_steps) * 55)
            elapsed = time.time() - start
            eta = (elapsed / max(step_index, 1)) * (num_inference_steps - step_index)
            progress_callback(pct, 100, f"Wan 2.2 14B step {step_index+1}/{num_inference_steps} (ETA: {int(eta)}s)")
        return cb_kwargs

    if progress_callback:
        progress_callback(35, 100, f"Generating video ({num_frames} frames, {num_inference_steps} steps)...")

    output = pipe(
        image=image,
        prompt=prompt,
        negative_prompt=neg,
        width=width,
        height=height,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
        callback_on_step_end=step_cb,
    )

    from diffusers.utils import export_to_video
    output_path = os.path.join(OUTPUT_DIR, f"wan22_{uuid4().hex[:8]}.mp4")
    export_to_video(output.frames[0], output_path, fps=fps)

    if progress_callback:
        progress_callback(100, 100, "Done.")
    return output_path


def animate_vace(image_path, prompt, negative_prompt="", height=480, width=832,
                 num_frames=81, num_inference_steps=30, guidance_scale=5.0,
                 fps=16, seed=None, progress_callback=None):
    """Animate using VACE 1.3B first-frame conditioning."""
    pipe = _load_vace(progress_callback)

    first_image = PIL.Image.open(image_path).convert("RGB").resize((width, height), PIL.Image.LANCZOS)
    video_frames = [first_image] + [PIL.Image.new("RGB", (width, height), (128, 128, 128))] * (num_frames - 1)
    mask_frames = [PIL.Image.new("L", (width, height), 0)] + [PIL.Image.new("L", (width, height), 255)] * (num_frames - 1)

    generator = torch.Generator(device="cpu")
    if seed is not None:
        generator.manual_seed(seed)

    neg = negative_prompt or "worst quality, low quality, static, no motion, frozen, blurry, ugly, deformed"
    start = time.time()

    def step_cb(pipe_obj, step_index, timestep, cb_kwargs):
        if progress_callback:
            pct = 35 + int((step_index / num_inference_steps) * 55)
            elapsed = time.time() - start
            eta = (elapsed / max(step_index, 1)) * (num_inference_steps - step_index)
            progress_callback(pct, 100, f"VACE step {step_index+1}/{num_inference_steps} (ETA: {int(eta)}s)")
        return cb_kwargs

    if progress_callback:
        progress_callback(35, 100, f"Animating ({num_frames} frames)...")

    output = pipe(video=video_frames, mask=mask_frames, prompt=prompt, negative_prompt=neg,
                  height=height, width=width, num_frames=num_frames,
                  num_inference_steps=num_inference_steps, guidance_scale=guidance_scale,
                  generator=generator, callback_on_step_end=step_cb)

    from diffusers.utils import export_to_video
    output_path = os.path.join(OUTPUT_DIR, f"vace_{uuid4().hex[:8]}.mp4")
    export_to_video(output.frames[0], output_path, fps=fps)

    if progress_callback:
        progress_callback(100, 100, "Done.")
    return output_path


def animate_hunyuan15(image_path, prompt, negative_prompt="", height=480, width=832,
                      num_frames=49, num_inference_steps=30, guidance_scale=6.0,
                      fps=16, seed=None, progress_callback=None):
    """Animate using HunyuanVideo 1.5 I2V GGUF — balanced quality/speed."""
    pipe = _load_hunyuan15_gguf(progress_callback)

    image = PIL.Image.open(image_path).convert("RGB").resize((width, height), PIL.Image.LANCZOS)
    generator = torch.Generator(device="cpu")
    if seed is not None:
        generator.manual_seed(seed)

    neg = negative_prompt or "low quality, blurry, distorted, worst quality, static, no motion, ugly, deformed"
    start = time.time()

    def step_cb(pipe_obj, step_index, timestep, cb_kwargs):
        if progress_callback:
            pct = 35 + int((step_index / num_inference_steps) * 55)
            elapsed = time.time() - start
            eta = (elapsed / max(step_index, 1)) * (num_inference_steps - step_index)
            progress_callback(pct, 100, f"HunyuanVideo 1.5 step {step_index+1}/{num_inference_steps} (ETA: {int(eta)}s)")
        return cb_kwargs

    if progress_callback:
        progress_callback(35, 100, f"Generating ({num_frames} frames, {num_inference_steps} steps)...")

    output = pipe(
        image=image,
        prompt=prompt,
        negative_prompt=neg,
        width=width,
        height=height,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
        callback_on_step_end=step_cb,
    )

    from diffusers.utils import export_to_video
    output_path = os.path.join(OUTPUT_DIR, f"hunyuan15_{uuid4().hex[:8]}.mp4")
    export_to_video(output.frames[0], output_path, fps=fps)

    if progress_callback:
        progress_callback(100, 100, "Done.")
    return output_path


def animate_ltx23(image_path, prompt, negative_prompt="", height=480, width=832,
                  num_frames=49, num_inference_steps=8, guidance_scale=1.0,
                  fps=24, seed=None, progress_callback=None):
    """Animate using LTX-2.3 distilled GGUF — fastest draft tier.

    LTX-2.3 distilled uses few steps (~8) and guidance_scale=1 (no CFG).
    """
    pipe = _load_ltx23_gguf(progress_callback)

    image = PIL.Image.open(image_path).convert("RGB").resize((width, height), PIL.Image.LANCZOS)
    generator = torch.Generator(device="cpu")
    if seed is not None:
        generator.manual_seed(seed)

    neg = negative_prompt or "low quality, blurry, distorted, worst quality, static"
    start = time.time()

    def step_cb(pipe_obj, step_index, timestep, cb_kwargs):
        if progress_callback:
            pct = 35 + int((step_index / num_inference_steps) * 55)
            elapsed = time.time() - start
            eta = (elapsed / max(step_index, 1)) * (num_inference_steps - step_index)
            progress_callback(pct, 100, f"LTX-2.3 step {step_index+1}/{num_inference_steps} (ETA: {int(eta)}s)")
        return cb_kwargs

    if progress_callback:
        progress_callback(35, 100, f"Generating ({num_frames} frames, {num_inference_steps} steps)...")

    output = pipe(
        image=image,
        prompt=prompt,
        negative_prompt=neg,
        width=width,
        height=height,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
        callback_on_step_end=step_cb,
    )

    from diffusers.utils import export_to_video
    output_path = os.path.join(OUTPUT_DIR, f"ltx23_{uuid4().hex[:8]}.mp4")
    export_to_video(output.frames[0], output_path, fps=fps)

    if progress_callback:
        progress_callback(100, 100, "Done.")
    return output_path


def animate_t2v(prompt, negative_prompt="", num_frames=49, num_inference_steps=30,
                guidance_scale=5.0, fps=16, seed=None, progress_callback=None):
    """Fallback: Wan T2V 1.3B (no image input)."""
    _unload_i2v()
    _unload_sdxl()
    from videogen import VideoGenerator, VideoGenConfig
    config = VideoGenConfig(engine="wan", wan_model="1.3b", wan_resolution="480p",
                            num_frames=num_frames, num_inference_steps=num_inference_steps,
                            guidance_scale=guidance_scale, fps=fps, seed=seed)
    gen = VideoGenerator(config)
    if progress_callback:
        progress_callback(35, 100, f"T2V 1.3B ({num_frames} frames, {num_inference_steps} steps)...")
    return gen.generate_text2video(prompt=prompt, negative_prompt=negative_prompt,
                                   progress_callback=progress_callback)


def _extract_last_frame(video_path: str, output_image_path: str) -> str:
    """Extract the last frame of ``video_path`` as a PNG for I2V continuation.

    Uses ffmpeg `-sseof -0.5` (seek to 0.5s before EOF) which is robust across
    container/codec combinations — `-sseof -0.04` can return an empty stream
    when the keyframe interval is large.
    """
    cmd = [
        "ffmpeg", "-y",
        "-sseof", "-0.5",
        "-i", video_path,
        "-update", "1",
        "-q:v", "2",
        "-frames:v", "1",
        output_image_path,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if not os.path.isfile(output_image_path) or os.path.getsize(output_image_path) < 1024:
        raise RuntimeError(f"Failed to extract last frame from {video_path}")
    return output_image_path


def _concat_clips(clip_paths: list, output_path: str) -> str:
    """Concatenate clips with ffmpeg's concat demuxer (no re-encode).

    Requires all clips to share codec/container — true here because every
    clip in a chain comes from the same engine + diffusers ``export_to_video``.
    """
    if len(clip_paths) == 1:
        return clip_paths[0]
    list_path = output_path + ".concat.txt"
    with open(list_path, "w", encoding="utf-8") as f:
        for c in clip_paths:
            f.write(f"file '{os.path.abspath(c).replace(chr(92), '/')}'\n")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
             "-i", list_path, "-c", "copy", output_path],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        # Codec mismatch fallback: re-encode at the same fps
        subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
             "-i", list_path, "-c:v", "libx264", "-preset", "veryfast",
             "-pix_fmt", "yuv420p", output_path],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    finally:
        try:
            os.remove(list_path)
        except OSError:
            pass
    return output_path


def _animate_chained(engine: str, image_path: str, prompt: str, negative_prompt: str,
                     target_duration: float, num_inference_steps: int,
                     guidance_scale: float, seed: Optional[int],
                     progress_callback: Optional[Callable]) -> str:
    """Generate enough chained I2V clips to cover ``target_duration``.

    Each clip after the first uses the previous clip's last frame as its image
    input, so motion continues smoothly. Final clips are concatenated with
    ffmpeg (no re-encode when codecs match).
    """
    fps = ENGINE_NATIVE_FPS[engine]
    max_frames = ENGINE_MAX_FRAMES[engine]
    seconds_per_clip = (max_frames - 1) / fps  # diffusers requires (4k+1) frames
    total_seconds = max(seconds_per_clip, float(target_duration))
    n_clips = max(1, math.ceil(total_seconds / seconds_per_clip))

    if progress_callback:
        progress_callback(
            10, 100,
            f"Chained generation: {n_clips} clip(s) × ~{seconds_per_clip:.1f}s @ {fps}fps for {total_seconds:.1f}s total",
        )

    clip_paths = []
    current_image = image_path
    chain_temp_frames: list = []

    for idx in range(n_clips):
        # Per-clip frame count: max for all but the last, which is trimmed to
        # exactly cover the remaining time.
        if idx == n_clips - 1 and n_clips > 1:
            remaining = total_seconds - idx * seconds_per_clip
            frames_needed = max(17, int(remaining * fps) + 1)
            # Round to (4k + 1) form expected by Wan/Hunyuan/VACE; LTX uses (8k + 1).
            stride = 8 if engine == "ltx23_distilled_gguf" else 4
            frames = ((frames_needed - 1) // stride) * stride + 1
            frames = min(frames, max_frames)
        else:
            frames = max_frames

        clip_seed = (seed + idx) if seed is not None else None

        def clip_cb(s: int, t: int, m: str, _i=idx, _n=n_clips):
            if progress_callback:
                base = 35 + int((_i / _n) * 55)
                span = int((1.0 / _n) * 55)
                pct = base + int((s / t) * span)
                progress_callback(pct, 100, f"Chain {_i+1}/{_n}: {m}")

        if engine == "wan22_14b_gguf":
            cp = animate_wan22_gguf(
                image_path=current_image, prompt=prompt, negative_prompt=negative_prompt,
                num_frames=frames, num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale, fps=fps, seed=clip_seed,
                progress_callback=clip_cb,
            )
        elif engine == "hunyuan15_i2v_gguf":
            cp = animate_hunyuan15(
                image_path=current_image, prompt=prompt, negative_prompt=negative_prompt,
                num_frames=frames, num_inference_steps=num_inference_steps,
                guidance_scale=min(guidance_scale, 6.5), fps=fps, seed=clip_seed,
                progress_callback=clip_cb,
            )
        elif engine == "ltx23_distilled_gguf":
            cp = animate_ltx23(
                image_path=current_image, prompt=prompt, negative_prompt=negative_prompt,
                num_frames=frames, num_inference_steps=8, guidance_scale=1.0,
                fps=fps, seed=clip_seed, progress_callback=clip_cb,
            )
        elif engine == "vace_1.3b":
            cp = animate_vace(
                image_path=current_image, prompt=prompt, negative_prompt=negative_prompt,
                num_frames=frames, num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale, fps=fps, seed=clip_seed,
                progress_callback=clip_cb,
            )
        else:
            raise ValueError(f"Chained generation not supported for engine={engine!r}")

        clip_paths.append(cp)

        if idx < n_clips - 1:
            next_image = os.path.join(OUTPUT_DIR, f"chain_frame_{uuid4().hex[:8]}.png")
            _extract_last_frame(cp, next_image)
            chain_temp_frames.append(next_image)
            current_image = next_image

    if len(clip_paths) == 1:
        return clip_paths[0]

    final_path = os.path.join(OUTPUT_DIR, f"chained_{engine.split('_')[0]}_{uuid4().hex[:8]}.mp4")
    _concat_clips(clip_paths, final_path)

    # Best-effort cleanup of intermediate frame extractions.
    for p in chain_temp_frames:
        try:
            os.remove(p)
        except OSError:
            pass

    return final_path


def generate_scene_video(prompt, negative_prompt="", target_duration=5.0, fps=16,
                         num_inference_steps=30, guidance_scale=5.0, seed=None,
                         progress_callback=None, preferred_engine: Optional[str] = None):
    """Full pipeline: pick best engine (respects ``preferred_engine`` override).

    preferred_engine: one of
      - "auto" / None  -> best available
      - "quality" / "wan22_14b_gguf"    -> Wan 2.2 14B
      - "balanced" / "hunyuan15_i2v_gguf" -> HunyuanVideo 1.5
      - "draft" / "ltx23_distilled_gguf" -> LTX-2.3 distilled
      - "vace_1.3b", "t2v_1.3b" -> older local fallbacks
    """
    # Env var can force the engine when preferred_engine unset
    if preferred_engine is None:
        preferred_engine = os.getenv("VIDEO_ENGINE", "auto") or "auto"

    engine = get_available_engine(preferred_engine)

    # --- SDXL step is shared across all I2V engines ---
    def _do_sdxl():
        if progress_callback:
            progress_callback(2, 100, "Step 1/2: Scene image (SDXL)...")
        return generate_scene_image(prompt=prompt, negative_prompt=negative_prompt, seed=seed)

    if engine in ENGINE_MAX_FRAMES:
        # All I2V engines share one chained-generation path. The chain helper
        # handles single-clip vs multi-clip uniformly (n_clips computed from
        # target_duration vs engine-native max).
        image_path = _do_sdxl()
        if progress_callback:
            progress_callback(8, 100, f"Step 2/2: {ENGINES.get(engine, engine)}...")
        return _animate_chained(
            engine=engine,
            image_path=image_path,
            prompt=prompt,
            negative_prompt=negative_prompt,
            target_duration=float(target_duration),
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            progress_callback=progress_callback,
        )

    # Last resort: T2V only (no image input, no chaining yet)
    num_frames = min(49, max(17, int(target_duration * fps)))
    num_frames = ((num_frames - 1) // 4) * 4 + 1
    if progress_callback:
        progress_callback(5, 100, "Using T2V 1.3B (no I2V engine available)...")
    return animate_t2v(prompt=prompt, negative_prompt=negative_prompt,
                       num_frames=num_frames, num_inference_steps=num_inference_steps,
                       guidance_scale=guidance_scale, fps=fps, seed=seed,
                       progress_callback=progress_callback)
