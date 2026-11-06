"""Audio-driven talking-head video generation.

Lip sync AT generation time (not post-hoc): give it a portrait image + an
audio clip, get back a video where the person speaks the audio with the
right mouth shapes, expression, and head motion.

Two backends behind one interface:

- SadTalker (primary): proven, open weights, ~12GB VRAM, moderate quality.
  Source: https://github.com/OpenTalker/SadTalker
  Weights: downloaded into ``.models/sadtalker/``.
- Hallo3 (optional, better quality): higher quality, ~24GB VRAM, slower.
  Source: https://github.com/fudan-generative-vision/hallo3
  Weights: downloaded into ``.models/hallo3/``.

Pattern follows ``lipsync.py``: ``available()`` gate, ``install_hint()``, and
one top-level ``generate_talking_head()`` that dispatches to whichever
backend is installed. The argv builders and validation are pure functions,
unit-tested without needing the model weights.
"""

from __future__ import annotations

import logging
import os
import subprocess
from typing import Callable, Optional

logger = logging.getLogger(__name__)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- SadTalker paths ---
SADTALKER_REPO = os.path.join(ROOT_DIR, "third_party", "SadTalker")
SADTALKER_WEIGHTS = os.path.join(ROOT_DIR, ".models", "sadtalker")
SADTALKER_INFERENCE = os.path.join(SADTALKER_REPO, "inference.py")
# Core weights we check for - if any are missing the backend is "not available"
SADTALKER_REQUIRED = [
    "SadTalker_V0.0.2_256.safetensors",
    "mapping_00229-model.pth.tar",
    "GFPGANv1.4.pth",
]

# --- Hallo3 paths ---
HALLO3_REPO = os.path.join(ROOT_DIR, "third_party", "hallo3")
HALLO3_WEIGHTS = os.path.join(ROOT_DIR, ".models", "hallo3")
HALLO3_INFERENCE = os.path.join(HALLO3_REPO, "scripts", "inference_long.py")
HALLO3_CKPT = os.path.join(HALLO3_WEIGHTS, "hallo3.pt")


def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def sadtalker_available() -> bool:
    """True iff SadTalker is cloned, all required weights exist, GPU present."""
    if not _has_cuda():
        return False
    if not os.path.isdir(SADTALKER_REPO) or not os.path.isfile(SADTALKER_INFERENCE):
        return False
    if not os.path.isdir(SADTALKER_WEIGHTS):
        return False
    for name in SADTALKER_REQUIRED:
        if not os.path.isfile(os.path.join(SADTALKER_WEIGHTS, name)):
            return False
    return True


def hallo3_available() -> bool:
    """True iff Hallo3 source + weights + GPU are all present."""
    if not _has_cuda():
        return False
    if not os.path.isdir(HALLO3_REPO) or not os.path.isfile(HALLO3_INFERENCE):
        return False
    return os.path.isfile(HALLO3_CKPT)


def available_backends() -> list[str]:
    """Which backends are runnable right now. Empty list = none installed."""
    out = []
    if sadtalker_available():
        out.append("sadtalker")
    if hallo3_available():
        out.append("hallo3")
    return out


def install_hint() -> str:
    """Multi-backend install guidance, listing whichever backend is incomplete."""
    lines = ["Talking-head generation needs ONE of the following backends:"]
    if not _has_cuda():
        lines.append(" - An NVIDIA GPU with CUDA drivers (required for both backends).")

    lines.append("")
    lines.append("SadTalker (12GB VRAM, moderate quality):")
    if not os.path.isdir(SADTALKER_REPO):
        lines.append(
            f"  - Clone: git clone https://github.com/OpenTalker/SadTalker {SADTALKER_REPO}"
        )
    missing = [
        n for n in SADTALKER_REQUIRED
        if not os.path.isfile(os.path.join(SADTALKER_WEIGHTS, n))
    ]
    if missing:
        lines.append(
            f"  - Download weights into {SADTALKER_WEIGHTS}: {', '.join(missing)}"
        )
    if sadtalker_available():
        lines.append("  - OK, ready.")

    lines.append("")
    lines.append("Hallo3 (24GB VRAM, higher quality):")
    if not os.path.isdir(HALLO3_REPO):
        lines.append(
            f"  - Clone: git clone https://github.com/fudan-generative-vision/hallo3 {HALLO3_REPO}"
        )
    if not os.path.isfile(HALLO3_CKPT):
        lines.append(
            f"  - Download hallo3.pt from Hugging Face into {HALLO3_WEIGHTS}"
        )
    if hallo3_available():
        lines.append("  - OK, ready.")

    return "\n".join(lines)


# --- pure argv builders (tested without weights) ---


def build_sadtalker_argv(image_path: str, audio_path: str, output_dir: str,
                         *, enhancer: str = "gfpgan", still: bool = True,
                         preprocess: str = "crop", size: int = 256,
                         expression_scale: float = 1.0,
                         pose_style: int = 0) -> list[str]:
    """Return the SadTalker ``inference.py`` command line.

    Defaults are the ones SadTalker's README recommends for a talking-head
    portrait shot (as opposed to full-body or multi-subject).
    """
    return [
        "python", SADTALKER_INFERENCE,
        "--driven_audio", audio_path,
        "--source_image", image_path,
        "--result_dir", output_dir,
        "--still" if still else "--no-still",
        "--preprocess", preprocess,
        "--size", str(size),
        "--expression_scale", str(expression_scale),
        "--pose_style", str(pose_style),
        "--enhancer", enhancer,
        "--checkpoint_dir", SADTALKER_WEIGHTS,
    ]


def build_hallo3_argv(image_path: str, audio_path: str, output_path: str,
                      *, config_path: Optional[str] = None,
                      inference_steps: int = 20,
                      guidance_scale: float = 2.0) -> list[str]:
    """Return the Hallo3 inference command line."""
    cfg = config_path or os.path.join(HALLO3_REPO, "configs", "inference.yaml")
    return [
        "python", HALLO3_INFERENCE,
        "--config", cfg,
        "--source_image", image_path,
        "--driving_audio", audio_path,
        "--output", output_path,
        "--inference_steps", str(inference_steps),
        "--cfg_scale", str(guidance_scale),
        "--checkpoint", HALLO3_CKPT,
    ]


def _validate_inputs(image_path: str, audio_path: str) -> None:
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Portrait image not found: {image_path}")
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Driving audio not found: {audio_path}")
    # Common pitfall: webp/heic portraits break face detectors; the model
    # loaders expect jpg/png and silently produce a solid-color frame otherwise.
    ext = os.path.splitext(image_path)[1].lower()
    if ext not in {".jpg", ".jpeg", ".png"}:
        raise ValueError(
            f"Portrait must be .jpg/.jpeg/.png (got {ext}). "
            "Convert first - most talking-head models fail silently on other formats."
        )


def generate_talking_head(
    image_path: str,
    audio_path: str,
    output_path: Optional[str] = None,
    *,
    backend: str = "auto",            # "auto" | "sadtalker" | "hallo3"
    expression_scale: float = 1.0,    # SadTalker: 0.5 subtle, 1.5 expressive
    guidance_scale: float = 2.0,      # Hallo3 only
    inference_steps: int = 20,        # Hallo3 only
    progress_callback: Optional[Callable] = None,
    runner: Optional[Callable] = None,
) -> str:
    """Generate a lip-synced talking-head video.

    ``backend="auto"`` prefers Hallo3 (higher quality) when available and
    falls back to SadTalker. The ``runner`` callable is injectable so the
    unit tests don't invoke a 15 minute subprocess; production passes
    ``subprocess.run``.
    """
    _validate_inputs(image_path, audio_path)
    backends = available_backends()
    if not backends:
        raise RuntimeError(install_hint())

    if backend == "auto":
        picked = "hallo3" if "hallo3" in backends else "sadtalker"
    else:
        picked = backend
        if picked not in backends:
            raise RuntimeError(
                f"Backend {picked!r} is not available. Installed: {backends}. "
                f"{install_hint()}"
            )

    if output_path is None:
        base, _ = os.path.splitext(os.path.basename(image_path))
        output_path = os.path.join(os.path.dirname(image_path), f"{base}_talking.mp4")

    if progress_callback:
        progress_callback(5, 100, f"Starting {picked} (this may take several minutes)...")

    r = runner if runner is not None else (
        lambda a: subprocess.run(a, capture_output=True, text=True, check=False)
    )

    if picked == "sadtalker":
        out_dir = os.path.dirname(output_path) or "."
        os.makedirs(out_dir, exist_ok=True)
        argv = build_sadtalker_argv(
            image_path, audio_path, out_dir,
            expression_scale=expression_scale,
        )
    else:  # hallo3
        argv = build_hallo3_argv(
            image_path, audio_path, output_path,
            inference_steps=inference_steps, guidance_scale=guidance_scale,
        )

    proc = r(argv)
    if getattr(proc, "returncode", 1) != 0:
        stderr = getattr(proc, "stderr", "") or ""
        raise RuntimeError(
            f"{picked} inference failed (exit {getattr(proc, 'returncode', 1)}): "
            f"{stderr[-800:] if stderr else 'no stderr'}"
        )

    if picked == "sadtalker":
        # SadTalker writes to result_dir/<timestamp>/<name>.mp4 - find it.
        out_dir = os.path.dirname(output_path) or "."
        import glob
        candidates = sorted(
            glob.glob(os.path.join(out_dir, "**", "*.mp4"), recursive=True),
            key=os.path.getmtime, reverse=True,
        )
        if not candidates:
            raise RuntimeError(f"SadTalker finished but produced no .mp4 in {out_dir}")
        if candidates[0] != output_path:
            try:
                os.replace(candidates[0], output_path)
            except OSError:
                output_path = candidates[0]

    if not os.path.isfile(output_path) or os.path.getsize(output_path) < 1024:
        raise RuntimeError(f"{picked} produced no usable output at {output_path}")

    if progress_callback:
        progress_callback(100, 100, f"{picked} complete.")
    return output_path
