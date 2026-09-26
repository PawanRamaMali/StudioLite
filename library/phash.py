"""Perceptual hash for near-duplicate detection.

`imagehash.phash` on a single downscaled frame is the workhorse. For
video we sample several evenly-spaced frames via ffmpeg, phash each,
and XOR-merge them into a signature. XOR-merge is deliberately lossy - 
it collapses ordering - but that's what we want: two rescales of the
same clip come out equal, and two similar-looking clips with different
edits stay close in Hamming distance.

Each hash is stored as hex; Hamming distance is bit-popcount of XOR.
"""
from __future__ import annotations

import io
import logging
import os
import subprocess
from typing import List, Optional

from PIL import Image
import imagehash

logger = logging.getLogger("studiolite.library.phash")

FRAME_COUNT = 5     # how many frames to sample per video
FRAME_WIDTH = 256   # decode target; big enough for phash to be stable


def _extract_frames(path: str, duration_sec: Optional[float]) -> List[Image.Image]:
    """Grab N frames spread evenly across the clip via ffmpeg's fps filter.
    Returns a list of PIL Images. Never raises - returns [] on any failure."""
    if not path or not os.path.isfile(path):
        return []
    # If we don't know the duration, ask ffmpeg for a fixed sampling.
    # fps filter with rate = FRAME_COUNT / duration if we have it; else use
    # `-vf select` at even video-percentage points via keyframe seek.
    if duration_sec and duration_sec > 0.5:
        rate = max(0.01, FRAME_COUNT / max(duration_sec, 1.0))
        vf = f"fps={rate},scale={FRAME_WIDTH}:-2"
    else:
        # Short clip - just grab up to FRAME_COUNT frames at 1 fps.
        vf = f"fps=1,scale={FRAME_WIDTH}:-2"
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", path,
        "-vf", vf,
        "-vframes", str(FRAME_COUNT),
        "-f", "image2pipe", "-vcodec", "mjpeg",
        "-",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=120, check=False)
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
        logger.debug("ffmpeg failed on %s: %s", path, e)
        return []
    if proc.returncode != 0 or not proc.stdout:
        return []

    frames: List[Image.Image] = []
    buf = proc.stdout
    # The pipe carries concatenated JPEGs. Walk them by finding the SOI/EOI
    # markers (FFD8...FFD9) rather than depending on any external splitter.
    i = 0
    while True:
        start = buf.find(b"\xff\xd8", i)
        if start < 0:
            break
        end = buf.find(b"\xff\xd9", start + 2)
        if end < 0:
            break
        end += 2
        try:
            img = Image.open(io.BytesIO(buf[start:end]))
            img.load()
            frames.append(img)
        except Exception:
            pass
        i = end
    return frames


def phash_video(path: str, duration_sec: Optional[float]) -> Optional[str]:
    """Return a hex phash for the whole video, or None on failure."""
    frames = _extract_frames(path, duration_sec)
    if not frames:
        return None
    accum: Optional[imagehash.ImageHash] = None
    for f in frames:
        try:
            h = imagehash.phash(f)  # 64-bit
        except Exception:
            continue
        if accum is None:
            accum = h
        else:
            # XOR-merge: sum XOR of every prior sample. Order-independent.
            accum = imagehash.ImageHash(accum.hash ^ h.hash)
    if accum is None:
        return None
    return str(accum)  # hex


def phash_image(path: str) -> Optional[str]:
    """Return a hex phash for a single still image. Handles anything Pillow
    can open; failures (RAW without a codec, corrupt files) return None."""
    if not path or not os.path.isfile(path):
        return None
    try:
        img = Image.open(path)
        img.load()
        # Convert to RGB - phash won't handle palettized modes cleanly.
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        return str(imagehash.phash(img))
    except Exception as e:
        logger.debug("phash_image failed on %s: %s", path, e)
        return None


def hamming(a: str, b: str) -> int:
    """Bit distance between two hex phashes. Length-invariant: pads the
    shorter side with 0s so a truncated hash doesn't blow up."""
    if not a or not b: return 64
    la, lb = len(a), len(b)
    if la != lb:
        if la < lb: a = a.ljust(lb, "0")
        else:       b = b.ljust(la, "0")
    try:
        ai = int(a, 16); bi = int(b, 16)
    except ValueError:
        return 64
    return (ai ^ bi).bit_count()
