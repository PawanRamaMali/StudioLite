"""RIFE (Real-time Intermediate Flow Estimation) frame interpolation.

Wraps `rife-ncnn-vulkan-python` with a function-shaped API that takes
raw file paths and returns a result dict instead of raising, so API
code can report failures as job errors with an install hint.

Supports 2x, 4x, and 8x via recursive midpoint interpolation: the
ncnn-vulkan Rife binding only computes the midpoint between two frames,
so 4x = midpoints of midpoints, 8x = one more pass.
"""
from __future__ import annotations

import os
import shutil
import tempfile
from typing import Any, Callable, Dict, List, Optional, Tuple


INSTALL_HINT = (
    "RIFE is not installed. "
    "Run `pip install rife-ncnn-vulkan-python` and ensure a Vulkan driver "
    "is present (any recent GPU driver on Windows/Linux)."
)


_SUPPORTED_MULTIPLIERS = (2, 4, 8)


def _passes_for(multiplier: int) -> int:
    """How many recursive 2x passes needed. 2 -> 1, 4 -> 2, 8 -> 3."""
    n = 0
    m = multiplier
    while m > 1:
        m //= 2
        n += 1
    return n


def _validate_multiplier(multiplier: int) -> Optional[str]:
    if multiplier == 1:
        return None  # no-op; callers can short-circuit, we handle it too
    if multiplier not in _SUPPORTED_MULTIPLIERS:
        return (f"RIFE multiplier must be one of {_SUPPORTED_MULTIPLIERS}, "
                f"got {multiplier}")
    return None


def _interpolate_once(frames: List[str], rife_engine,
                      work_dir: str, round_tag: str) -> List[str]:
    """One 2x pass: for every adjacent pair in `frames`, insert a
    midpoint produced by rife_engine.process(a, b)."""
    from PIL import Image as PILImage
    import numpy as np

    result: List[str] = []
    for i, fp in enumerate(frames):
        result.append(fp)
        if i >= len(frames) - 1:
            continue
        img_a = PILImage.open(fp)
        img_b = PILImage.open(frames[i + 1])
        mid = rife_engine.process(np.array(img_a), np.array(img_b))
        mid_path = os.path.join(work_dir, f"interp_{round_tag}_{i:05d}.png")
        PILImage.fromarray(mid).save(mid_path)
        result.append(mid_path)
    return result


def interpolate_video(
    src_abs: str,
    dst_abs: str,
    *,
    multiplier: int = 2,
    gpuid: int = 0,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> Dict[str, Any]:
    """Interpolate `src_abs` -> `dst_abs` at `multiplier`x framerate.

    Returns {ok: True, backend, multiplier, fps_in, fps_out, output_path}
    on success, or {ok: False, error, install_hint?} on any failure.
    Never raises. Original audio (if any) is preserved by muxing it
    back onto the interpolated video via `frames_to_video(audio_from=)`.
    """
    err = _validate_multiplier(multiplier)
    if err:
        return {"ok": False, "error": err}
    if not os.path.isfile(src_abs):
        return {"ok": False, "error": f"source not found: {src_abs}"}

    if multiplier == 1:
        # No interpolation requested: just copy the file through.
        try:
            os.makedirs(os.path.dirname(dst_abs) or ".", exist_ok=True)
            shutil.copyfile(src_abs, dst_abs)
            return {"ok": True, "backend": "rife-noop", "multiplier": 1,
                    "fps_in": None, "fps_out": None, "output_path": dst_abs}
        except OSError as e:
            return {"ok": False, "error": f"copy failed: {e}"}

    try:
        from rife_ncnn_vulkan_python import Rife
    except ImportError as e:
        return {"ok": False,
                "error": f"RIFE import failed: {e}",
                "install_hint": INSTALL_HINT}

    try:
        from video_editor import extract_frames, frames_to_video
    except ImportError as e:
        return {"ok": False, "error": f"video_editor unavailable: {e}"}

    work_dir = tempfile.mkdtemp(prefix="rife_")
    try:
        extracted = extract_frames(src_abs, work_dir)
        # extract_frames returns (frames: list, fps: float). Earlier call
        # sites forgot to unpack and treated the tuple as the frame list,
        # which crashed on the first PIL open. Unpack defensively so a
        # future return-shape change doesn't resurface the same bug.
        if (isinstance(extracted, tuple) and len(extracted) == 2
                and isinstance(extracted[0], list)):
            frames, fps_in = extracted
        elif isinstance(extracted, list):
            frames, fps_in = extracted, None
        else:
            return {"ok": False,
                    "error": f"extract_frames returned unexpected shape: {type(extracted).__name__}"}

        if len(frames) < 2:
            return {"ok": False,
                    "error": f"need at least 2 frames to interpolate, got {len(frames)}"}

        try:
            engine = Rife(gpuid=gpuid)
        except Exception as e:
            return {"ok": False,
                    "error": f"RIFE init failed on gpuid={gpuid}: {e}",
                    "install_hint": INSTALL_HINT}

        passes = _passes_for(multiplier)
        current = frames
        total = passes
        for p in range(passes):
            if progress_callback:
                progress_callback(p, total,
                                   f"RIFE pass {p+1}/{total} ({len(current)} frames in)")
            current = _interpolate_once(current, engine, work_dir, f"p{p}")

        if progress_callback:
            progress_callback(total, total, f"RIFE done, muxing {len(current)} frames")

        fps_out = (fps_in * multiplier) if fps_in else float(24 * multiplier)
        os.makedirs(os.path.dirname(dst_abs) or ".", exist_ok=True)
        frames_to_video(current, dst_abs, fps=fps_out, audio_from=src_abs)

        if not (os.path.exists(dst_abs) and os.path.getsize(dst_abs) > 1024):
            return {"ok": False, "error": "interpolation produced no output"}

        return {
            "ok": True,
            "backend": f"rife_x{multiplier}",
            "multiplier": multiplier,
            "fps_in": fps_in,
            "fps_out": fps_out,
            "output_path": dst_abs,
        }
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def is_available() -> Tuple[bool, str]:
    """Report whether the RIFE binding is importable on this machine."""
    try:
        import rife_ncnn_vulkan_python  # noqa: F401
    except ImportError:
        return False, INSTALL_HINT
    return True, "rife-ncnn-vulkan-python importable"
