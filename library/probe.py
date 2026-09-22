"""ffprobe wrapper — pulls duration / width / height / codec / fps from a
video file. Fail-open; on any error every field is None."""
from __future__ import annotations

import json
import logging
import os
import subprocess
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("studiolite.library.probe")


@dataclass
class ProbeResult:
    duration_sec: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    codec: Optional[str] = None
    fps: Optional[float] = None


def probe(path: str, *, timeout: float = 20.0) -> ProbeResult:
    if not os.path.isfile(path):
        return ProbeResult()
    cmd = [
        "ffprobe", "-hide_banner", "-loglevel", "error",
        "-print_format", "json",
        "-show_format", "-show_streams",
        path,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout, check=False)
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
        logger.debug("ffprobe failed on %s: %s", path, e)
        return ProbeResult()
    if proc.returncode != 0 or not proc.stdout:
        return ProbeResult()

    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return ProbeResult()

    # Duration lives in either the format or the video-stream section.
    duration: Optional[float] = None
    fmt = data.get("format") or {}
    if fmt.get("duration"):
        try: duration = float(fmt["duration"])
        except Exception: pass

    width = height = None
    codec = None
    fps: Optional[float] = None
    for s in data.get("streams") or []:
        if s.get("codec_type") != "video":
            continue
        codec = s.get("codec_name") or codec
        if s.get("width") and not width: width = int(s["width"])
        if s.get("height") and not height: height = int(s["height"])
        if duration is None and s.get("duration"):
            try: duration = float(s["duration"])
            except Exception: pass
        # avg_frame_rate is "30000/1001" — parse safely
        rate = s.get("avg_frame_rate") or s.get("r_frame_rate")
        if rate and "/" in rate:
            try:
                num, den = rate.split("/")
                num, den = float(num), float(den)
                if den > 0:
                    fps = round(num / den, 2)
            except Exception:
                pass
        break  # first video stream wins

    return ProbeResult(duration_sec=duration, width=width, height=height,
                       codec=codec, fps=fps)
