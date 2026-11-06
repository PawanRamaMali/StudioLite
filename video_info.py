"""Video metadata inspection helpers.

One ffprobe call per file gets every readable stream's dimensions, codec,
fps, duration and bitrate. The parser tolerates missing fields - partial
reads are still useful, so we return what we have and don't raise on a
single-stream file that lacks audio metadata.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


def _run_default(cmd: list) -> Any:
    return subprocess.run(cmd, capture_output=True, text=True)


def _check_ffprobe(ffprobe_bin: str) -> Optional[str]:
    if shutil.which(ffprobe_bin) is None:
        return "ffprobe not on PATH. Install ffmpeg (bundles ffprobe) to inspect videos."
    return None


def build_probe_cmd(src: str, *, ffprobe_bin: str = "ffprobe") -> list:
    return [
        ffprobe_bin, "-v", "error",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        src,
    ]


def _parse_fps(rate: Optional[str]) -> Optional[float]:
    """ffprobe emits fps as ``num/den``. ``0/0`` means "unknown" and we
    return None rather than divide-by-zero."""
    if not rate or "/" not in rate:
        return None
    try:
        num, den = rate.split("/")
        n = float(num)
        d = float(den)
        if d == 0:
            return None
        return n / d
    except (ValueError, ZeroDivisionError):
        return None


@dataclass
class VideoInfo:
    path: str
    format_name: str = ""
    duration: Optional[float] = None
    size_bytes: Optional[int] = None
    bitrate: Optional[int] = None
    streams: List[Dict[str, Any]] = field(default_factory=list)
    video: Optional[Dict[str, Any]] = None
    audio: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "format_name": self.format_name,
            "duration": self.duration,
            "size_bytes": self.size_bytes,
            "bitrate": self.bitrate,
            "streams": self.streams,
            "video": self.video,
            "audio": self.audio,
        }


def parse_probe_json(payload: str, src: str) -> VideoInfo:
    """Parse ffprobe -print_format json output into a VideoInfo.

    This is deliberately split out from the subprocess call so tests can
    feed realistic payloads without launching ffprobe."""
    info = VideoInfo(path=src)
    try:
        data = json.loads(payload or "{}")
    except (ValueError, TypeError):
        return info

    fmt = data.get("format") or {}
    info.format_name = fmt.get("format_name") or ""
    try:
        info.duration = float(fmt["duration"]) if fmt.get("duration") else None
    except (ValueError, TypeError):
        info.duration = None
    try:
        info.size_bytes = int(fmt["size"]) if fmt.get("size") else None
    except (ValueError, TypeError):
        info.size_bytes = None
    try:
        info.bitrate = int(fmt["bit_rate"]) if fmt.get("bit_rate") else None
    except (ValueError, TypeError):
        info.bitrate = None

    streams = data.get("streams") or []
    for s in streams:
        row = {
            "index": s.get("index"),
            "codec_type": s.get("codec_type"),
            "codec_name": s.get("codec_name"),
            "codec_long_name": s.get("codec_long_name"),
            "bit_rate": int(s["bit_rate"]) if (s.get("bit_rate") or "").isdigit() else None,
        }
        if s.get("codec_type") == "video":
            row.update({
                "width": s.get("width"),
                "height": s.get("height"),
                "pix_fmt": s.get("pix_fmt"),
                "fps": _parse_fps(s.get("avg_frame_rate") or s.get("r_frame_rate")),
                "has_bframes": s.get("has_b_frames"),
                "aspect_ratio": s.get("display_aspect_ratio"),
            })
            if info.video is None:
                info.video = row
        elif s.get("codec_type") == "audio":
            row.update({
                "sample_rate": int(s["sample_rate"]) if (s.get("sample_rate") or "").isdigit() else None,
                "channels": s.get("channels"),
                "channel_layout": s.get("channel_layout"),
            })
            if info.audio is None:
                info.audio = row
        info.streams.append(row)
    return info


def inspect_video(
    src: str,
    *,
    ffprobe_bin: str = "ffprobe",
    runner: Callable[[list], Any] = _run_default,
) -> Dict[str, Any]:
    """Return { ok, info } for a readable video, or { ok: False, error }."""
    if not os.path.isfile(src):
        return {"ok": False, "error": f"Source not found: {src}"}
    hint = _check_ffprobe(ffprobe_bin)
    if hint:
        return {"ok": False, "error": "ffprobe missing", "install_hint": hint}
    cmd = build_probe_cmd(src, ffprobe_bin=ffprobe_bin)
    proc = runner(cmd)
    rc = getattr(proc, "returncode", 1)
    if rc != 0:
        err = (getattr(proc, "stderr", "") or "").strip().splitlines()
        return {"ok": False, "error": err[-1] if err else "ffprobe failed"}
    stdout = getattr(proc, "stdout", "") or ""
    info = parse_probe_json(stdout, src)
    # Backfill size from disk if ffprobe skipped it.
    if info.size_bytes is None:
        try:
            info.size_bytes = os.path.getsize(src)
        except OSError:
            pass
    return {"ok": True, "info": info.to_dict()}
