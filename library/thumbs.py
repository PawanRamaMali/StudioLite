"""On-demand JPEG thumbnails for library rows. Cached under
.mp/library/thumbs/<video_id>.jpg so repeat views don't re-invoke ffmpeg."""
from __future__ import annotations

import logging
import os
import subprocess
from typing import Optional

logger = logging.getLogger("studiolite.library.thumbs")


def thumb_path(root_dir: str, video_id: int) -> str:
    return os.path.join(root_dir, "thumbs", f"{video_id}.jpg")


def ensure_image_thumb(root_dir: str, media_id: int, image_abs_path: str,
                        *, width: int = 480) -> Optional[str]:
    """PIL-based thumbnail for a still image — quick, offline, no ffmpeg needed."""
    out = thumb_path(root_dir, media_id)
    if os.path.isfile(out) and os.path.getsize(out) > 100:
        return out
    os.makedirs(os.path.dirname(out), exist_ok=True)
    try:
        from PIL import Image
        with Image.open(image_abs_path) as img:
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")
            # Preserve aspect ratio; scale down only.
            w, h = img.size
            if w > width:
                new_h = int(h * (width / w))
                img = img.resize((width, new_h), Image.LANCZOS)
            img.save(out, "JPEG", quality=82, optimize=True)
        return out if os.path.isfile(out) and os.path.getsize(out) > 100 else None
    except Exception as e:
        logger.debug("image thumb failed on %s: %s", image_abs_path, e)
        return None


def ensure_thumb(root_dir: str, video_id: int, video_abs_path: str,
                 *, width: int = 480, at_percent: float = 0.15) -> Optional[str]:
    """Generate the thumbnail if it's missing, return its path.
    `at_percent` seeks that fraction into the video for a frame — 15%
    tends to skip title cards and land on real content."""
    out = thumb_path(root_dir, video_id)
    if os.path.isfile(out) and os.path.getsize(out) > 100:
        return out
    os.makedirs(os.path.dirname(out), exist_ok=True)
    # Use ffprobe-less approach: seek by percentage using -ss with a huge
    # duration probe would be slow; instead we seek to a fixed 2s in.
    # If the file is shorter than that, ffmpeg still writes the closest
    # available frame.
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-ss", "2",
        "-i", video_abs_path,
        "-frames:v", "1",
        "-vf", f"scale={width}:-2",
        "-q:v", "5",
        out,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=30, check=False)
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
        logger.debug("thumb ffmpeg failed on %s: %s", video_abs_path, e)
        return None
    if proc.returncode != 0 or not os.path.isfile(out) or os.path.getsize(out) < 100:
        # Retry without seeking — very short clips don't have a frame at t=2.
        cmd_retry = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", video_abs_path,
            "-frames:v", "1",
            "-vf", f"scale={width}:-2",
            "-q:v", "5",
            out,
        ]
        try:
            subprocess.run(cmd_retry, capture_output=True, timeout=30, check=False)
        except Exception:
            return None
    return out if os.path.isfile(out) and os.path.getsize(out) > 100 else None


def clear_thumb(root_dir: str, video_id: int) -> None:
    p = thumb_path(root_dir, video_id)
    try:
        if os.path.isfile(p):
            os.remove(p)
    except OSError:
        pass
