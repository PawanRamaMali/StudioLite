"""Chroma-key (green-screen) removal helpers.

Pure-Python wrappers around ffmpeg's ``chromakey`` filter. Three modes:
- ``transparent``  -> VP9+alpha WebM with the key color replaced by alpha.
- ``color``        -> flat solid-color background behind the keyed clip.
- ``image``        -> still image behind the keyed clip, scaled+padded.
- ``video``        -> another video behind the keyed clip, trimmed to length.

ffmpeg argv assembly is pulled out as pure functions so the exact command
is unit-testable without launching ffmpeg. The top-level
``remove_chroma_key`` accepts an injectable ``runner`` that defaults to
``subprocess.run`` - tests pass a fake.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional


# Preset colour names -> 0xRRGGBB. ffmpeg's chromakey accepts bare hex
# strings or named colours, but we normalize to "0xRRGGBB" so tests and
# the UI can agree on the wire format.
_NAMED_KEYS = {
    "green":  "0x00FF00",
    "blue":   "0x0000FF",
    "red":    "0xFF0000",
    "cyan":   "0x00FFFF",
    "magenta": "0xFF00FF",
    "yellow": "0xFFFF00",
}

_HEX_RE = re.compile(r"^#?[0-9a-fA-F]{6}$")


def normalize_key_color(color: str) -> str:
    """Turn `#00FF00`, `00ff00`, `0x00FF00`, or `green` into `0x00FF00`.
    Unknown inputs fall back to pure green - green screens are the most
    common and we never want to fail the whole render over a misspelling."""
    c = (color or "").strip()
    low = c.lower()
    if low in _NAMED_KEYS:
        return _NAMED_KEYS[low]
    if low.startswith("0x") and len(low) == 8 and _HEX_RE.match(low[2:]):
        return "0x" + low[2:].upper()
    if _HEX_RE.match(c):
        return "0x" + c.lstrip("#").upper()
    return "0x00FF00"


@dataclass
class ChromaKeyStyle:
    """Key + compositing dials. ``similarity`` is how far from the key
    colour ffmpeg considers "green enough" (0..1, higher = more aggressive);
    ``blend`` softens the edge (0..1, higher = fuzzier, better for hair).
    Values past 0.5 start eating foreground; defaults match a typical
    well-lit green screen."""
    key_color: str = "green"
    similarity: float = 0.15
    blend: float = 0.08
    # Output dims are only used when generating a solid-colour background
    # or when sizing a still image to match the keyed clip. In video-bg
    # mode we take the keyed clip's native size from ffmpeg itself.
    out_width: int = 1280
    out_height: int = 720
    fps: int = 30
    background_color: str = "0x000000"  # background=color mode only
    audio: str = "copy"  # "copy" or "drop"


# ---------------------------------------------------------------------------
# Filter graph construction
# ---------------------------------------------------------------------------


def build_key_filter(style: ChromaKeyStyle) -> str:
    """``chromakey=color=…:similarity=…:blend=…`` clause for one video."""
    key = normalize_key_color(style.key_color)
    sim = max(0.0, min(1.0, float(style.similarity)))
    bld = max(0.0, min(1.0, float(style.blend)))
    return f"chromakey=color={key}:similarity={sim:.3f}:blend={bld:.3f}"


def build_transparent_cmd(
    src: str,
    dst: str,
    *,
    style: ChromaKeyStyle,
    ffmpeg_bin: str = "ffmpeg",
) -> List[str]:
    """Encode to VP9 + alpha WebM so the key becomes real transparency.
    yuva420p is the only alpha-capable layout libvpx-vp9 accepts, and
    -auto-alt-ref must be 0 for that pixel format in current ffmpeg."""
    vf = f"{build_key_filter(style)},format=yuva420p"
    cmd = [
        ffmpeg_bin, "-y",
        "-i", src,
        "-vf", vf,
        "-c:v", "libvpx-vp9",
        "-pix_fmt", "yuva420p",
        "-auto-alt-ref", "0",
        "-b:v", "0", "-crf", "30",
    ]
    if style.audio == "copy":
        cmd += ["-c:a", "libopus", "-b:a", "128k"]
    else:
        cmd += ["-an"]
    cmd += [dst]
    return cmd


def build_color_bg_cmd(
    src: str,
    dst: str,
    *,
    style: ChromaKeyStyle,
    ffmpeg_bin: str = "ffmpeg",
) -> List[str]:
    """Composite the keyed clip over a solid-colour frame via lavfi's
    ``color=`` source. ffmpeg generates the background frame; no need for
    an external PNG or video input."""
    bg_color = normalize_key_color(style.background_color)
    graph = (
        f"color=c={bg_color}:s={style.out_width}x{style.out_height}:"
        f"r={style.fps}[bg];"
        f"[0:v]{build_key_filter(style)},format=yuva420p[fg];"
        f"[bg][fg]overlay=shortest=1[vout]"
    )
    cmd = [
        ffmpeg_bin, "-y",
        "-i", src,
        "-filter_complex", graph,
        "-map", "[vout]",
    ]
    if style.audio == "copy":
        cmd += ["-map", "0:a?", "-c:a", "copy"]
    else:
        cmd += ["-an"]
    cmd += [
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-pix_fmt", "yuv420p",
        dst,
    ]
    return cmd


def build_image_bg_cmd(
    src: str,
    dst: str,
    *,
    background_path: str,
    style: ChromaKeyStyle,
    ffmpeg_bin: str = "ffmpeg",
) -> List[str]:
    """Composite the keyed clip over a still image. The image is scaled
    + padded to the output size then looped for the clip's duration via
    ``-loop 1`` + overlay's ``shortest=1``."""
    graph = (
        f"[1:v]scale={style.out_width}:{style.out_height}:"
        f"force_original_aspect_ratio=decrease:flags=lanczos,"
        f"pad={style.out_width}:{style.out_height}:(ow-iw)/2:(oh-ih)/2,"
        f"setsar=1[bg];"
        f"[0:v]{build_key_filter(style)},format=yuva420p[fg];"
        f"[bg][fg]overlay=shortest=1[vout]"
    )
    cmd = [
        ffmpeg_bin, "-y",
        "-i", src,
        "-loop", "1", "-i", background_path,
        "-filter_complex", graph,
        "-map", "[vout]",
    ]
    if style.audio == "copy":
        cmd += ["-map", "0:a?", "-c:a", "copy"]
    else:
        cmd += ["-an"]
    cmd += [
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-shortest",
        dst,
    ]
    return cmd


def build_video_bg_cmd(
    src: str,
    dst: str,
    *,
    background_path: str,
    style: ChromaKeyStyle,
    ffmpeg_bin: str = "ffmpeg",
) -> List[str]:
    """Composite the keyed clip over another video. Background is scaled
    to the output size; `shortest=1` trims the composite to whichever
    stream ends first (usually the foreground clip)."""
    graph = (
        f"[1:v]scale={style.out_width}:{style.out_height}:"
        f"force_original_aspect_ratio=decrease:flags=lanczos,"
        f"pad={style.out_width}:{style.out_height}:(ow-iw)/2:(oh-ih)/2,"
        f"setsar=1[bg];"
        f"[0:v]{build_key_filter(style)},format=yuva420p[fg];"
        f"[bg][fg]overlay=shortest=1[vout]"
    )
    cmd = [
        ffmpeg_bin, "-y",
        "-i", src,
        "-i", background_path,
        "-filter_complex", graph,
        "-map", "[vout]",
    ]
    if style.audio == "copy":
        cmd += ["-map", "0:a?", "-c:a", "copy"]
    else:
        cmd += ["-an"]
    cmd += [
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-shortest",
        dst,
    ]
    return cmd


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
_VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi"}


def _default_runner(cmd: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def _run(cmd, runner):
    try:
        proc = (runner or _default_runner)(cmd)
    except FileNotFoundError:
        return None, {
            "ok": False,
            "error": "ffmpeg not found",
            "install_hint": "Install ffmpeg and ensure it is on PATH.",
        }
    except OSError as e:
        return None, {"ok": False, "error": f"ffmpeg launch failed: {e}"}
    return proc, None


def _stderr_tail(proc) -> str:
    stderr = (getattr(proc, "stderr", "") or "").strip()
    return stderr.splitlines()[-1] if stderr else f"exit {proc.returncode}"


def remove_chroma_key(
    src_abs: str,
    dst_abs: str,
    *,
    mode: str = "transparent",
    background_path: Optional[str] = None,
    style: Optional[ChromaKeyStyle] = None,
    ffmpeg_bin: str = "ffmpeg",
    runner: Optional[Callable[[List[str]], subprocess.CompletedProcess]] = None,
) -> Dict[str, Any]:
    """Remove the keyed colour from ``src_abs`` into ``dst_abs``.

    ``mode`` picks the output type:
      - "transparent" - VP9+alpha .webm, uses ``dst_abs`` as given but will
        enforce a .webm extension and rewrite ``dst_abs`` to the corrected
        path on the returned dict.
      - "color"       - mp4 composited over ``style.background_color``.
      - "image"       - mp4 composited over ``background_path`` (still image).
      - "video"       - mp4 composited over ``background_path`` (video).

    Returns ``{ok, backend, mode, output_path}`` on success or
    ``{ok: False, error, install_hint?}`` on failure. Never raises.
    """
    if not os.path.isfile(src_abs):
        return {"ok": False, "error": f"source not found: {src_abs}"}
    style = style or ChromaKeyStyle()
    m = (mode or "transparent").lower()

    if m == "transparent":
        # Force .webm; VP9+alpha into an mp4 container fails silently.
        root, _ = os.path.splitext(dst_abs)
        dst_abs = root + ".webm"
    elif m in ("image", "video"):
        if not background_path or not os.path.isfile(background_path):
            return {"ok": False,
                    "error": f"background not found: {background_path}"}
        ext = os.path.splitext(background_path)[1].lower()
        if m == "image" and ext not in _IMAGE_EXTS:
            return {"ok": False,
                    "error": f"image background must be one of {sorted(_IMAGE_EXTS)}, got {ext}"}
        if m == "video" and ext not in _VIDEO_EXTS:
            return {"ok": False,
                    "error": f"video background must be one of {sorted(_VIDEO_EXTS)}, got {ext}"}
    elif m != "color":
        return {"ok": False, "error": f"unknown mode: {mode}"}

    os.makedirs(os.path.dirname(dst_abs) or ".", exist_ok=True)
    ff = shutil.which(ffmpeg_bin) or ffmpeg_bin

    if m == "transparent":
        cmd = build_transparent_cmd(src_abs, dst_abs, style=style, ffmpeg_bin=ff)
    elif m == "color":
        cmd = build_color_bg_cmd(src_abs, dst_abs, style=style, ffmpeg_bin=ff)
    elif m == "image":
        cmd = build_image_bg_cmd(
            src_abs, dst_abs,
            background_path=background_path,  # type: ignore[arg-type]
            style=style, ffmpeg_bin=ff,
        )
    else:  # video
        cmd = build_video_bg_cmd(
            src_abs, dst_abs,
            background_path=background_path,  # type: ignore[arg-type]
            style=style, ffmpeg_bin=ff,
        )

    proc, err = _run(cmd, runner)
    if err is not None:
        return err
    if proc.returncode != 0:
        return {"ok": False, "error": f"ffmpeg failed: {_stderr_tail(proc)}"}
    if not (os.path.exists(dst_abs) and os.path.getsize(dst_abs) > 1024):
        return {"ok": False, "error": "chroma key produced no output"}

    return {
        "ok": True,
        "backend": "ffmpeg_chromakey",
        "mode": m,
        "output_path": dst_abs,
    }
