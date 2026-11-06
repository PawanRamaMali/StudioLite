"""PuLID identity embedding for SDXL scene image generation.

Why this exists: Wan's IP-Adapter path has weak temporal identity carry-over
across scenes. PuLID is the open equivalent of what Kling/Runway do under
the hood - a dedicated identity encoder that runs alongside SDXL during
sampling so the face stays on-model across a multi-scene short.

Interface pattern follows ``lipsync.py`` / ``talking_head.py``: ``available()``
gate, ``install_hint()`` for setup, pure argv/config builders tested without
weights, and one orchestration function that the Scene Generator calls
when ``use_identity`` is on.

Backend: PuLID for SDXL (https://github.com/ToTheBeginning/PuLID). Weights
live at ``.models/pulid/`` and are loaded via the PuLID package when
installed. Falls back to a plain SDXL call with the ref image as IP-Adapter
input when PuLID is not available - so switching the flag on degrades
gracefully instead of hard-failing.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Callable, Optional

logger = logging.getLogger(__name__)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PULID_WEIGHTS_DIR = os.path.join(ROOT_DIR, ".models", "pulid")
PULID_SDXL_CKPT = os.path.join(PULID_WEIGHTS_DIR, "pulid_sdxl.safetensors")
# PuLID also needs its face-encoder bundle (antelopev2 by default)
PULID_FACE_ENCODER_DIR = os.path.join(PULID_WEIGHTS_DIR, "antelopev2")


def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _has_pulid_package() -> bool:
    """True iff the ``pulid`` Python package is installed in the active env."""
    try:
        import importlib
        importlib.import_module("pulid")
        return True
    except ImportError:
        return False


def pulid_available() -> bool:
    """True iff PuLID is installable and all its weights are on disk."""
    if not _has_cuda():
        return False
    if not _has_pulid_package():
        return False
    if not os.path.isfile(PULID_SDXL_CKPT):
        return False
    if not os.path.isdir(PULID_FACE_ENCODER_DIR):
        return False
    return True


def install_hint() -> str:
    lines = ["Stronger identity embedding (PuLID for SDXL):"]
    if not _has_cuda():
        lines.append(" - Requires an NVIDIA GPU with CUDA drivers.")
    if not _has_pulid_package():
        lines.append(" - Install the package: pip install pulid")
    if not os.path.isfile(PULID_SDXL_CKPT):
        lines.append(
            f" - Download pulid_sdxl.safetensors into {PULID_WEIGHTS_DIR} "
            "from https://huggingface.co/guozinan/PuLID"
        )
    if not os.path.isdir(PULID_FACE_ENCODER_DIR):
        lines.append(
            f" - Extract the antelopev2 face-encoder into {PULID_FACE_ENCODER_DIR} "
            "(InsightFace models)"
        )
    if pulid_available():
        return "PuLID is set up correctly."
    return "\n".join(lines)


@dataclass
class IdentityConfig:
    """Pure config for a PuLID-augmented SDXL call."""
    reference_image: str          # path to a clean portrait of the target identity
    prompt: str
    negative_prompt: str = ""
    width: int = 1024
    height: int = 1024
    num_inference_steps: int = 25
    guidance_scale: float = 5.0
    id_scale: float = 0.8         # how strongly identity pulls - 0.6-1.0 is typical
    seed: Optional[int] = None
    output_path: str = ""         # writer picks a default if blank


def _resolve_output_path(cfg: IdentityConfig) -> str:
    if cfg.output_path:
        return cfg.output_path
    # Default: alongside the reference image, with a hash-ish suffix
    base = os.path.splitext(os.path.basename(cfg.reference_image))[0]
    tag = f"{abs(hash((cfg.prompt, cfg.seed, cfg.id_scale))) & 0xFFFF:04x}"
    out_dir = os.path.dirname(cfg.reference_image) or "."
    return os.path.join(out_dir, f"{base}_pulid_{tag}.png")


def _validate_config(cfg: IdentityConfig) -> None:
    if not os.path.isfile(cfg.reference_image):
        raise FileNotFoundError(f"Reference image not found: {cfg.reference_image}")
    if not cfg.prompt or not cfg.prompt.strip():
        raise ValueError("prompt is required")
    if cfg.width % 8 or cfg.height % 8:
        raise ValueError(
            f"width/height must be multiples of 8 (got {cfg.width}x{cfg.height})"
        )
    if cfg.id_scale < 0.0 or cfg.id_scale > 2.0:
        raise ValueError(f"id_scale must be 0.0-2.0 (got {cfg.id_scale})")
    # Common face-encoder failure: webp/heic refs. Catch early.
    ext = os.path.splitext(cfg.reference_image)[1].lower()
    if ext not in {".jpg", ".jpeg", ".png"}:
        raise ValueError(
            f"Reference image must be .jpg/.jpeg/.png (got {ext}). "
            "PuLID's face encoder fails silently on other formats."
        )


def generate_with_identity(cfg: IdentityConfig,
                           progress_callback: Optional[Callable] = None,
                           runner: Optional[Callable] = None) -> str:
    """Generate a scene image with PuLID identity carry-over.

    When PuLID isn't installed, falls back to a plain SDXL call using the
    reference image as an IP-Adapter input via ``rf_generate_image``. The
    fallback produces worse identity preservation than PuLID but doesn't
    hard-fail; callers get a working image they can still animate.

    ``runner`` is an injectable "do the inference" callable: it takes the
    config and returns the output path. Tests use a fake that writes a stub
    PNG; production passes ``_pulid_runner`` which calls diffusers.
    """
    _validate_config(cfg)
    out_path = _resolve_output_path(cfg)
    if progress_callback:
        progress_callback(5, 100, "Preparing identity generation...")

    if pulid_available():
        if progress_callback:
            progress_callback(15, 100, "Loading PuLID...")
        r = runner if runner is not None else _pulid_runner
        path = r(cfg, out_path)
    else:
        logger.info("PuLID not available; falling back to IP-Adapter path. "
                    "Install hint: %s", install_hint())
        if progress_callback:
            progress_callback(15, 100, "PuLID not installed - using IP-Adapter fallback...")
        r = runner if runner is not None else _ip_adapter_runner
        path = r(cfg, out_path)

    if not os.path.isfile(path) or os.path.getsize(path) < 1024:
        raise RuntimeError(f"Identity generation produced no usable output at {path}")
    if progress_callback:
        progress_callback(100, 100, "Done.")
    return path


def _pulid_runner(cfg: IdentityConfig, out_path: str) -> str:
    """Real inference path. Imported inside so tests don't pull diffusers."""
    import torch
    from PIL import Image
    try:
        # PuLID's own sampler wraps SDXL + the identity cross-attention
        from pulid.pipeline_v1_1 import PuLIDPipeline  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            f"PuLID package import failed: {e}. {install_hint()}"
        )
    ref = Image.open(cfg.reference_image).convert("RGB")
    generator = torch.Generator(device="cpu")
    if cfg.seed is not None:
        generator.manual_seed(int(cfg.seed))
    pipe = PuLIDPipeline.from_pretrained_sdxl(PULID_SDXL_CKPT)
    img = pipe(
        id_image=ref,
        prompt=cfg.prompt,
        negative_prompt=cfg.negative_prompt,
        width=cfg.width,
        height=cfg.height,
        num_inference_steps=cfg.num_inference_steps,
        guidance_scale=cfg.guidance_scale,
        id_scale=cfg.id_scale,
        generator=generator,
    )
    img.save(out_path, format="PNG")
    return out_path


def _ip_adapter_runner(cfg: IdentityConfig, out_path: str) -> str:
    """Graceful fallback: SDXL with the reference as IP-Adapter conditioning."""
    import shutil
    from reelforge import rf_generate_image
    # rf_generate_image already supports ip_adapter_image, so this is the same
    # degraded path we already use when a Characters portrait is attached.
    # Infer aspect from the config dims.
    ratio = cfg.width / max(1, cfg.height)
    if ratio < 0.7:
        aspect = "9:16"
    elif ratio < 0.95:
        aspect = "4:5"
    elif ratio < 1.3:
        aspect = "1:1"
    else:
        aspect = "16:9"
    img_path = rf_generate_image(
        prompt=cfg.prompt,
        provider="sdxl_turbo",
        style="photorealistic",
        aspect_ratio=aspect,
        seed=cfg.seed,
        ip_adapter_image=cfg.reference_image,
    )
    if img_path != out_path:
        shutil.move(img_path, out_path)
    return out_path
