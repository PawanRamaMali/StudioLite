"""Thumbnail generation helpers.

Pure Python wrappers around ffmpeg that produce three flavours of
thumbnails from a video: a single frame at a timestamp, a set of frames
at arbitrary timestamps (packed into a zip for one-file download), and
a contact-sheet grid of evenly-spaced frames tiled into a single PNG.
Each variant can optionally composite a PNG overlay (logo / watermark).

ffmpeg argv assembly is split out as pure functions so the exact
command shape is unit-testable without touching ffmpeg itself. The
top-level ``generate_thumbnails`` accepts an injectable ``runner`` that
defaults to ``subprocess.run`` - tests pass a fake.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import zipfile
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Public result shape
# ---------------------------------------------------------------------------
# Every public entry point returns a dict of this shape so the API layer
# can treat success and failure uniformly:
#   {ok: True, backend, mode, output_paths: [...], archive_path?: str}
#   {ok: False, error: str, install_hint?: str}


_ALLOWED_POSITIONS = {
    "top-left", "top-right", "bottom-left", "bottom-right", "center",
}


# ---------------------------------------------------------------------------
# Duration probe
# ---------------------------------------------------------------------------


def probe_duration(
    src: str,
    *,
    ffprobe_bin: str = "ffprobe",
    runner: Optional[Callable[[List[str]], subprocess.CompletedProcess]] = None,
) -> Optional[float]:
    """Return the video duration in seconds, or None if ffprobe is missing /
    unreadable. We return None instead of raising so grid mode can fall back
    to a sensible default sampling range when the probe fails."""
    cmd = [
        ffprobe_bin, "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        src,
    ]
    try:
        _runner = runner or (lambda c: subprocess.run(
            c, capture_output=True, text=True, check=False,
        ))
        proc = _runner(cmd)
    except (FileNotFoundError, OSError):
        return None
    if proc.returncode != 0:
        return None
    try:
        return float((proc.stdout or "").strip())
    except (TypeError, ValueError):
        return None


def evenly_spaced_timestamps(duration: float, count: int) -> List[float]:
    """Pick `count` timestamps evenly spaced across the video, nudged in from
    the exact endpoints so ffmpeg actually lands on a decoded frame instead
    of the keyframe before 0 or the EOF edge."""
    if count <= 0:
        return []
    if duration <= 0:
        return [0.0] * count
    if count == 1:
        return [duration / 2.0]
    # Split [0, duration] into `count` equal buckets and sample each bucket's
    # midpoint — nicer than inclusive linspace which doubles up at the ends.
    step = duration / count
    return [round(step * (i + 0.5), 3) for i in range(count)]


# ---------------------------------------------------------------------------
# Overlay filter construction
# ---------------------------------------------------------------------------


def _overlay_xy(position: str, pad: int = 20) -> str:
    """ffmpeg overlay coordinates. Width/height tokens are the standard
    `main_w`/`overlay_w` variables from libavfilter."""
    pos = position if position in _ALLOWED_POSITIONS else "bottom-right"
    if pos == "top-left":
        return f"{pad}:{pad}"
    if pos == "top-right":
        return f"main_w-overlay_w-{pad}:{pad}"
    if pos == "bottom-left":
        return f"{pad}:main_h-overlay_h-{pad}"
    if pos == "center":
        return "(main_w-overlay_w)/2:(main_h-overlay_h)/2"
    # bottom-right (default)
    return f"main_w-overlay_w-{pad}:main_h-overlay_h-{pad}"


def build_overlay_filter_complex(
    base_vf: str,
    overlay_path: str,
    position: str,
    opacity: float,
) -> str:
    """Return a -filter_complex graph that routes:
       [0:v] -> base_vf -> [v0]
       [1:v] -> alpha scale (opacity) -> [ov]
       [v0][ov] overlay=<xy> -> [vout]

    Opacity is applied via `colorchannelmixer=aa=<op>` on the overlay
    stream so partially-transparent logos work without touching the base.
    """
    op = max(0.0, min(1.0, float(opacity)))
    xy = _overlay_xy(position)
    base_label = "v0"
    chain = f"[0:v]{base_vf}[{base_label}]" if base_vf else f"[0:v]copy[{base_label}]"
    ov_chain = f"[1:v]format=rgba,colorchannelmixer=aa={op}[ov]"
    composite = f"[{base_label}][ov]overlay={xy}[vout]"
    return ";".join([chain, ov_chain, composite])


# ---------------------------------------------------------------------------
# Single-frame extraction
# ---------------------------------------------------------------------------


def build_single_cmd(
    src: str,
    dst: str,
    timestamp: float,
    *,
    width: int = 1280,
    overlay_path: Optional[str] = None,
    overlay_position: str = "bottom-right",
    overlay_opacity: float = 1.0,
    ffmpeg_bin: str = "ffmpeg",
) -> List[str]:
    """Argv for extracting one PNG frame. ``-ss`` before ``-i`` is the fast
    keyframe seek; good enough for thumbnails and keeps the command cheap."""
    base_vf = f"scale={width}:-1"
    if overlay_path:
        fc = build_overlay_filter_complex(
            base_vf, overlay_path, overlay_position, overlay_opacity,
        )
        return [
            ffmpeg_bin, "-y",
            "-ss", str(max(0.0, float(timestamp))),
            "-i", src,
            "-i", overlay_path,
            "-filter_complex", fc,
            "-map", "[vout]",
            "-frames:v", "1",
            dst,
        ]
    return [
        ffmpeg_bin, "-y",
        "-ss", str(max(0.0, float(timestamp))),
        "-i", src,
        "-frames:v", "1",
        "-vf", base_vf,
        dst,
    ]


# ---------------------------------------------------------------------------
# Grid / contact-sheet extraction
# ---------------------------------------------------------------------------


def build_grid_cmd(
    src: str,
    dst: str,
    *,
    cols: int,
    rows: int,
    width: int,
    duration: Optional[float],
    overlay_path: Optional[str] = None,
    overlay_position: str = "bottom-right",
    overlay_opacity: float = 1.0,
    ffmpeg_bin: str = "ffmpeg",
) -> List[str]:
    """Argv for a contact sheet. We sample ``cols*rows`` frames via the
    ``select`` filter at a derived period, scale each one to
    ``width/cols`` and tile them with the ``tile`` filter.

    When ``duration`` is None (ffprobe failed) we fall back to sampling
    every N-th frame with a reasonable default of N=300, which gives a
    useful sheet for most ~10-minute videos."""
    count = max(1, cols * rows)
    cell_w = max(32, width // cols)
    if duration and duration > 0:
        # Sample at evenly-spaced timestamps. ffmpeg's `fps=count/duration`
        # drops frames to roughly `count` evenly across the stream.
        rate = count / duration
        base_vf = (
            f"fps={rate:.6f},scale={cell_w}:-1,"
            f"tile={cols}x{rows}"
        )
    else:
        # Duration unknown — sample every 300th decoded frame.
        base_vf = (
            f"select='not(mod(n\\,300))',"
            f"scale={cell_w}:-1,tile={cols}x{rows}"
        )
    if overlay_path:
        fc = build_overlay_filter_complex(
            base_vf, overlay_path, overlay_position, overlay_opacity,
        )
        return [
            ffmpeg_bin, "-y",
            "-i", src,
            "-i", overlay_path,
            "-filter_complex", fc,
            "-map", "[vout]",
            "-frames:v", "1",
            dst,
        ]
    return [
        ffmpeg_bin, "-y",
        "-i", src,
        "-vf", base_vf,
        "-frames:v", "1",
        "-vsync", "vfr",
        dst,
    ]


# ---------------------------------------------------------------------------
# Runner plumbing
# ---------------------------------------------------------------------------


def _default_runner(cmd: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def _run(
    cmd: List[str],
    runner: Optional[Callable[[List[str]], subprocess.CompletedProcess]],
) -> Tuple[Optional[subprocess.CompletedProcess], Optional[Dict[str, Any]]]:
    """Execute ``cmd`` and return either (proc, None) on completion, or
    (None, error_dict) when the launch itself failed. Callers still have
    to check proc.returncode."""
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


def _stderr_tail(proc: subprocess.CompletedProcess) -> str:
    stderr = (getattr(proc, "stderr", "") or "").strip()
    return stderr.splitlines()[-1] if stderr else f"exit {proc.returncode}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_single(
    src_abs: str,
    dst_abs: str,
    timestamp: float,
    *,
    width: int = 1280,
    overlay_path: Optional[str] = None,
    overlay_position: str = "bottom-right",
    overlay_opacity: float = 1.0,
    ffmpeg_bin: str = "ffmpeg",
    runner: Optional[Callable[[List[str]], subprocess.CompletedProcess]] = None,
) -> Dict[str, Any]:
    if not os.path.isfile(src_abs):
        return {"ok": False, "error": f"source not found: {src_abs}"}
    if overlay_path and not os.path.isfile(overlay_path):
        return {"ok": False, "error": f"overlay not found: {overlay_path}"}
    os.makedirs(os.path.dirname(dst_abs) or ".", exist_ok=True)
    ff = shutil.which(ffmpeg_bin) or ffmpeg_bin
    cmd = build_single_cmd(
        src_abs, dst_abs, timestamp,
        width=width, overlay_path=overlay_path,
        overlay_position=overlay_position, overlay_opacity=overlay_opacity,
        ffmpeg_bin=ff,
    )
    proc, err = _run(cmd, runner)
    if err is not None:
        return err
    if proc.returncode != 0:
        return {"ok": False, "error": f"ffmpeg failed: {_stderr_tail(proc)}"}
    if not os.path.exists(dst_abs):
        return {"ok": False, "error": "thumbnail extraction produced no output"}
    return {
        "ok": True,
        "backend": "ffmpeg",
        "mode": "single",
        "output_paths": [dst_abs],
    }


def extract_grid(
    src_abs: str,
    dst_abs: str,
    *,
    cols: int = 3,
    rows: int = 3,
    width: int = 1920,
    duration: Optional[float] = None,
    overlay_path: Optional[str] = None,
    overlay_position: str = "bottom-right",
    overlay_opacity: float = 1.0,
    ffmpeg_bin: str = "ffmpeg",
    runner: Optional[Callable[[List[str]], subprocess.CompletedProcess]] = None,
) -> Dict[str, Any]:
    if not os.path.isfile(src_abs):
        return {"ok": False, "error": f"source not found: {src_abs}"}
    if overlay_path and not os.path.isfile(overlay_path):
        return {"ok": False, "error": f"overlay not found: {overlay_path}"}
    if cols <= 0 or rows <= 0:
        return {"ok": False, "error": "cols and rows must be positive"}
    os.makedirs(os.path.dirname(dst_abs) or ".", exist_ok=True)
    if duration is None:
        duration = probe_duration(src_abs, runner=runner)
    ff = shutil.which(ffmpeg_bin) or ffmpeg_bin
    cmd = build_grid_cmd(
        src_abs, dst_abs,
        cols=cols, rows=rows, width=width, duration=duration,
        overlay_path=overlay_path,
        overlay_position=overlay_position, overlay_opacity=overlay_opacity,
        ffmpeg_bin=ff,
    )
    proc, err = _run(cmd, runner)
    if err is not None:
        return err
    if proc.returncode != 0:
        return {"ok": False, "error": f"ffmpeg failed: {_stderr_tail(proc)}"}
    if not os.path.exists(dst_abs):
        return {"ok": False, "error": "contact-sheet produced no output"}
    return {
        "ok": True,
        "backend": "ffmpeg",
        "mode": "grid",
        "output_paths": [dst_abs],
    }


def extract_multiple(
    src_abs: str,
    out_dir: str,
    timestamps: List[float],
    *,
    name_prefix: str = "thumb",
    width: int = 1280,
    overlay_path: Optional[str] = None,
    overlay_position: str = "bottom-right",
    overlay_opacity: float = 1.0,
    archive_path: Optional[str] = None,
    ffmpeg_bin: str = "ffmpeg",
    runner: Optional[Callable[[List[str]], subprocess.CompletedProcess]] = None,
) -> Dict[str, Any]:
    """Extract one PNG per timestamp. If ``archive_path`` is given, pack
    the PNGs into a zip and return that path alongside the PNG list."""
    if not os.path.isfile(src_abs):
        return {"ok": False, "error": f"source not found: {src_abs}"}
    if overlay_path and not os.path.isfile(overlay_path):
        return {"ok": False, "error": f"overlay not found: {overlay_path}"}
    if not timestamps:
        return {"ok": False, "error": "at least one timestamp is required"}
    os.makedirs(out_dir, exist_ok=True)
    ff = shutil.which(ffmpeg_bin) or ffmpeg_bin
    paths: List[str] = []
    for i, ts in enumerate(timestamps):
        dst = os.path.join(out_dir, f"{name_prefix}_{i:03d}.png")
        cmd = build_single_cmd(
            src_abs, dst, ts,
            width=width, overlay_path=overlay_path,
            overlay_position=overlay_position, overlay_opacity=overlay_opacity,
            ffmpeg_bin=ff,
        )
        proc, err = _run(cmd, runner)
        if err is not None:
            return err
        if proc.returncode != 0:
            return {"ok": False,
                    "error": f"ffmpeg failed at t={ts}: {_stderr_tail(proc)}"}
        if not os.path.exists(dst):
            return {"ok": False, "error": f"no output at t={ts}"}
        paths.append(dst)

    result: Dict[str, Any] = {
        "ok": True,
        "backend": "ffmpeg",
        "mode": "multiple",
        "output_paths": paths,
    }
    if archive_path:
        try:
            with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for p in paths:
                    zf.write(p, arcname=os.path.basename(p))
        except OSError as e:
            return {"ok": False, "error": f"zip write failed: {e}"}
        result["archive_path"] = archive_path
    return result


def generate_thumbnails(
    src_abs: str,
    out_dir: str,
    *,
    mode: str = "single",
    timestamp: float = 0.0,
    timestamps: Optional[List[float]] = None,
    cols: int = 3,
    rows: int = 3,
    width: int = 1280,
    overlay_path: Optional[str] = None,
    overlay_position: str = "bottom-right",
    overlay_opacity: float = 1.0,
    job_id: str = "",
    runner: Optional[Callable[[List[str]], subprocess.CompletedProcess]] = None,
) -> Dict[str, Any]:
    """Dispatcher. Returns the standard result dict - never raises."""
    m = (mode or "single").lower()
    if m == "single":
        dst = os.path.join(out_dir, f"thumbnail_{job_id or 'out'}.png")
        return extract_single(
            src_abs, dst, timestamp,
            width=width, overlay_path=overlay_path,
            overlay_position=overlay_position, overlay_opacity=overlay_opacity,
            runner=runner,
        )
    if m == "grid":
        dst = os.path.join(out_dir, f"contact_{job_id or 'out'}.png")
        return extract_grid(
            src_abs, dst,
            cols=cols, rows=rows, width=width,
            overlay_path=overlay_path,
            overlay_position=overlay_position, overlay_opacity=overlay_opacity,
            runner=runner,
        )
    if m == "multiple":
        ts = list(timestamps or [])
        if not ts:
            dur = probe_duration(src_abs, runner=runner)
            # Fall back to 5 evenly-spaced frames when caller didn't pick any.
            ts = evenly_spaced_timestamps(dur or 60.0, 5)
        sub_dir = os.path.join(out_dir, f"thumbs_{job_id or 'out'}")
        archive = os.path.join(out_dir, f"thumbs_{job_id or 'out'}.zip")
        return extract_multiple(
            src_abs, sub_dir, ts,
            width=width, overlay_path=overlay_path,
            overlay_position=overlay_position, overlay_opacity=overlay_opacity,
            archive_path=archive,
            runner=runner,
        )
    return {"ok": False, "error": f"unknown mode: {mode}"}
