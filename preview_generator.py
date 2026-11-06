"""Low-res proxy preview for the video generator.

The idea: real production iteration means 20 takes to find one good one. A full
5-second 480p clip takes minutes; a tiny 256x448 x 17-frame draft at 4 steps
takes seconds. This module exposes a single ``build_preview_config`` call that
returns the aggressively-downscaled generator args, and ``preview_defaults``
that answers "how aggressive can we go for engine X without breaking it".

Pure functions only - no generator imports - so tests don't need a GPU.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class PreviewConfig:
    """What the generator is told to produce for a preview draft."""
    engine: str                  # "ltx23_distilled_gguf" | "wan22_14b_gguf" | ...
    width: int
    height: int
    num_frames: int              # keep tiny
    num_inference_steps: int     # LTX distilled: 4; others: 8-12 max
    guidance_scale: float
    target_duration_s: float
    fps: int
    seed: Optional[int] = None


# Per-engine aggressive-but-still-coherent preview settings.
# These are not what you ship - they're what you iterate on.
#
# Width/height floors: each engine has a minimum size below which the VAE
# produces mush. Found empirically.
#   LTX-2.3 distilled: anything >= 256 works; 32-stride.
#   Wan 2.2 14B:       anything >= 256; 16-stride.
#   HunyuanVideo 1.5:  needs >= 320; 16-stride.
#   VACE 1.3B:         needs >= 256; 16-stride.
_ENGINE_PREVIEW_DEFAULTS: dict[str, dict] = {
    "ltx23_distilled_gguf": {
        "stride": 32,
        "min_dim": 256,
        "num_inference_steps": 4,   # LTX distilled is single-digit steps
        "guidance_scale": 1.0,
        "fps": 24,
        "max_frames": 25,           # (8k+1) form; 25 = 1.04s at 24fps
    },
    "wan22_14b_gguf": {
        "stride": 16,
        "min_dim": 256,
        "num_inference_steps": 12,
        "guidance_scale": 3.5,
        "fps": 16,
        "max_frames": 25,           # (4k+1) form
    },
    "hunyuan15_i2v_gguf": {
        "stride": 16,
        "min_dim": 320,
        "num_inference_steps": 10,
        "guidance_scale": 6.0,
        "fps": 16,
        "max_frames": 25,
    },
    "vace_1.3b": {
        "stride": 16,
        "min_dim": 256,
        "num_inference_steps": 12,
        "guidance_scale": 5.0,
        "fps": 16,
        "max_frames": 25,
    },
}


# Social aspect ratios the preview supports; each maps to a minimal portrait
# or landscape box that respects the engine's stride at round-down time.
_PREVIEW_ASPECTS: dict[str, tuple[int, int]] = {
    "9:16": (256, 448),
    "16:9": (448, 256),
    "1:1":  (320, 320),
    "4:5":  (288, 360),
}


def _round_to_stride(value: int, stride: int) -> int:
    """Round ``value`` to the nearest multiple of ``stride``, at least ``stride``."""
    return max(stride, (int(value) // stride) * stride)


def preview_defaults(engine: str) -> dict:
    """Serializable dict describing the engine's preview settings.

    Falls back to the LTX-distilled defaults on unknown engine so a stale
    frontend id still gets something runnable.
    """
    base = _ENGINE_PREVIEW_DEFAULTS.get(engine, _ENGINE_PREVIEW_DEFAULTS["ltx23_distilled_gguf"])
    return {"engine": engine, **base, "aspects": list(_PREVIEW_ASPECTS.keys())}


def build_preview_config(prompt: str, engine: str = "ltx23_distilled_gguf",
                         aspect: str = "9:16",
                         target_duration_s: float = 1.0,
                         seed: Optional[int] = None) -> PreviewConfig:
    """Return a preview-tuned generator config.

    ``target_duration_s`` is a request, not a promise - the engine's
    ``max_frames`` ceiling and the (4k+1 / 8k+1) frame-count constraint
    often round it down. Preview previews are deliberately short (~1s).
    """
    defaults = _ENGINE_PREVIEW_DEFAULTS.get(engine, _ENGINE_PREVIEW_DEFAULTS["ltx23_distilled_gguf"])
    stride = defaults["stride"]
    min_dim = defaults["min_dim"]

    base_w, base_h = _PREVIEW_ASPECTS.get(aspect, _PREVIEW_ASPECTS["9:16"])
    w = max(min_dim, _round_to_stride(base_w, stride))
    h = max(min_dim, _round_to_stride(base_h, stride))

    fps = defaults["fps"]
    max_frames = defaults["max_frames"]
    # Engine frame-count constraint: Wan/Hunyuan/VACE need (4k + 1), LTX (8k + 1)
    frame_stride = 8 if engine == "ltx23_distilled_gguf" else 4
    raw_frames = max(frame_stride + 1, int(target_duration_s * fps) + 1)
    frames = ((raw_frames - 1) // frame_stride) * frame_stride + 1
    frames = min(frames, max_frames)

    return PreviewConfig(
        engine=engine,
        width=w,
        height=h,
        num_frames=frames,
        num_inference_steps=defaults["num_inference_steps"],
        guidance_scale=defaults["guidance_scale"],
        target_duration_s=(frames - 1) / fps,
        fps=fps,
        seed=seed,
    )


def preview_config_dict(prompt: str, engine: str = "ltx23_distilled_gguf",
                        aspect: str = "9:16",
                        target_duration_s: float = 1.0,
                        seed: Optional[int] = None) -> dict:
    """Same as :func:`build_preview_config` but returns a plain dict."""
    return asdict(build_preview_config(prompt, engine, aspect, target_duration_s, seed))


def estimate_preview_seconds(engine: str, num_frames: int, steps: int) -> float:
    """Rough wall-clock estimate for a preview job on a 24GB GPU.

    Numbers are empirical floors from the relevant engine's measured speed,
    not promises. Used by the frontend to show a "~4s" badge up front.
    """
    base = {
        "ltx23_distilled_gguf": 0.08,   # per (step * frame) - LTX distilled is very fast
        "wan22_14b_gguf": 0.45,
        "hunyuan15_i2v_gguf": 0.35,
        "vace_1.3b": 0.22,
    }.get(engine, 0.3)
    # 2s base overhead (SDXL image + pipeline load) + per-step cost
    return round(2.0 + base * max(1, steps) * max(1, num_frames) / 24, 1)
