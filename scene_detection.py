"""Scene / shot detection helpers.

Runs ffmpeg's ``scdet`` filter to find scene cuts, then optionally extracts
a thumbnail per scene so the UI can show a storyboard. The scdet output
is parsed from ffmpeg's stderr: each detected change prints a line
containing ``lavfi.scd.time: X.XXX`` (seconds) and ``lavfi.scd.score: Y``.

Covers issue #21 (Video Analytics - Scene detection, shot detection).
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional


_SCENE_TIME_RE = re.compile(r"lavfi\.scd\.time[:= ]\s*([0-9]+(?:\.[0-9]+)?)")
_SCENE_SCORE_RE = re.compile(r"lavfi\.scd\.score[:= ]\s*([0-9]+(?:\.[0-9]+)?)")


def _run_default(cmd: list) -> Any:
    return subprocess.run(cmd, capture_output=True, text=True)


def _check_ffmpeg(ffmpeg_bin: str) -> Optional[str]:
    if shutil.which(ffmpeg_bin) is None:
        return "ffmpeg not on PATH. Install ffmpeg for scene detection."
    return None


@dataclass
class SceneDetectionStyle:
    """``threshold`` is the scdet cut threshold (0..100, lower = more cuts).
    10 is scdet's default. ``min_scene_length`` filters out fast flashes
    that would otherwise register as back-to-back cuts; it's applied after
    scdet in post-processing."""
    threshold: float = 10.0
    min_scene_length: float = 1.0   # seconds between consecutive cuts
    extract_thumbnails: bool = True
    thumb_width: int = 320


# ---------------------------------------------------------------------------
# ffmpeg argv + parsing
# ---------------------------------------------------------------------------


def build_scdet_cmd(src: str, *, threshold: float = 10.0,
                     ffmpeg_bin: str = "ffmpeg") -> list:
    """ffmpeg -i src -vf scdet=threshold=X -an -f null -"""
    thr = max(0.0, min(100.0, float(threshold)))
    return [
        ffmpeg_bin, "-i", src,
        "-vf", f"scdet=threshold={thr:.2f}",
        "-an", "-f", "null", "-",
    ]


def parse_scdet_output(stderr: str) -> List[Dict[str, float]]:
    """Walk ffmpeg's stderr and pull out {time, score} per cut.

    scdet prints one time line and one score line per cut (the order and
    exact framing differ between ffmpeg builds, so we pair them up by
    iteration rather than by regex locality)."""
    times: List[float] = []
    scores: List[float] = []
    for line in (stderr or "").splitlines():
        m_t = _SCENE_TIME_RE.search(line)
        if m_t:
            try:
                times.append(float(m_t.group(1)))
            except ValueError:
                pass
            continue
        m_s = _SCENE_SCORE_RE.search(line)
        if m_s:
            try:
                scores.append(float(m_s.group(1)))
            except ValueError:
                pass
    cuts: List[Dict[str, float]] = []
    for i, t in enumerate(times):
        cuts.append({
            "time": t,
            "score": scores[i] if i < len(scores) else 0.0,
        })
    return cuts


def filter_min_length(
    cuts: List[Dict[str, float]],
    min_scene_length: float,
) -> List[Dict[str, float]]:
    """Drop cuts that land within ``min_scene_length`` seconds of the prior
    kept cut. Guards against over-counting fast strobes and flash cuts."""
    kept: List[Dict[str, float]] = []
    last_t = -float("inf")
    for c in cuts:
        if c["time"] - last_t >= max(0.0, float(min_scene_length)):
            kept.append(c)
            last_t = c["time"]
    return kept


def probe_duration(
    path: str,
    *,
    ffprobe_bin: str = "ffprobe",
    runner: Callable[[list], Any] = _run_default,
) -> Optional[float]:
    if shutil.which(ffprobe_bin) is None:
        return None
    proc = runner([
        ffprobe_bin, "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path,
    ])
    if getattr(proc, "returncode", 1) != 0:
        return None
    out = (getattr(proc, "stdout", "") or "").strip()
    try:
        return float(out) if out else None
    except (ValueError, TypeError):
        return None


def build_thumbnail_cmd(
    src: str,
    dst: str,
    timestamp: float,
    *,
    width: int = 320,
    ffmpeg_bin: str = "ffmpeg",
) -> list:
    """Grab one frame at ``timestamp`` and scale to ``width`` keeping AR."""
    ts = max(0.0, float(timestamp))
    return [
        ffmpeg_bin, "-y",
        "-ss", f"{ts:.3f}",
        "-i", src,
        "-vframes", "1",
        "-vf", f"scale={int(width)}:-2",
        "-q:v", "3",
        dst,
    ]


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def detect_scenes(
    src: str,
    out_dir: str,
    *,
    style: Optional[SceneDetectionStyle] = None,
    ffmpeg_bin: str = "ffmpeg",
    ffprobe_bin: str = "ffprobe",
    runner: Callable[[list], Any] = _run_default,
) -> Dict[str, Any]:
    style = style or SceneDetectionStyle()
    if not os.path.isfile(src):
        return {"ok": False, "error": f"Source not found: {src}"}
    hint = _check_ffmpeg(ffmpeg_bin)
    if hint:
        return {"ok": False, "error": "ffmpeg missing", "install_hint": hint}

    cmd = build_scdet_cmd(src, threshold=style.threshold, ffmpeg_bin=ffmpeg_bin)
    proc = runner(cmd)
    rc = getattr(proc, "returncode", 1)
    if rc != 0:
        # ffmpeg often returns 0 even with an empty result; a non-zero code
        # is a real failure (missing codec, unreadable input).
        err = getattr(proc, "stderr", "") or ""
        last = err.strip().splitlines()[-1] if err else "ffmpeg scdet failed"
        return {"ok": False, "error": last}

    stderr = getattr(proc, "stderr", "") or ""
    raw = parse_scdet_output(stderr)
    cuts = filter_min_length(raw, style.min_scene_length)

    # Build scene ranges: [start, end) pairs. Prepend 0 and append duration.
    duration = probe_duration(src, ffprobe_bin=ffprobe_bin, runner=runner)
    breakpoints = [0.0] + [c["time"] for c in cuts]
    if duration is not None and duration > 0:
        breakpoints.append(duration)

    scenes: List[Dict[str, Any]] = []
    for i in range(len(breakpoints) - 1):
        start = breakpoints[i]
        end = breakpoints[i + 1]
        if end <= start:
            continue
        mid = start + (end - start) / 2.0
        scenes.append({
            "index": i,
            "start": start,
            "end": end,
            "duration": end - start,
            "midpoint": mid,
            "score": cuts[i - 1]["score"] if (i >= 1 and i - 1 < len(cuts)) else 0.0,
            "thumbnail_path": None,
        })

    # Thumbnails
    if style.extract_thumbnails and scenes:
        os.makedirs(out_dir, exist_ok=True)
        for sc in scenes:
            thumb = os.path.join(out_dir, f"scene_{sc['index']:04d}.jpg")
            tcmd = build_thumbnail_cmd(
                src, thumb, sc["midpoint"],
                width=style.thumb_width, ffmpeg_bin=ffmpeg_bin,
            )
            tproc = runner(tcmd)
            if getattr(tproc, "returncode", 1) == 0 and os.path.isfile(thumb) \
                    and os.path.getsize(thumb) > 0:
                sc["thumbnail_path"] = thumb

    return {
        "ok": True,
        "backend": "ffmpeg:scdet",
        "scene_count": len(scenes),
        "cut_count": len(cuts),
        "duration": duration,
        "scenes": scenes,
        "cuts": cuts,
    }
