"""T3 - Quality enhancement for library videos.

Wraps the existing ``upscaler.upscale_video`` (Real-ESRGAN if the model
is on disk, Lanczos otherwise) plus a face-restoration pass (GFPGAN) that
loads only if its weights are available.

Recommendations are heuristic and read-only - they don't touch a file,
they just answer "if I were to enhance this, which preset makes sense?"
The user always confirms the actual run.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from .store import LibraryStore, Video

logger = logging.getLogger("studiolite.library.enhance")


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------

PRESETS = {
    "fast_2x":   {"scale": 2, "method": "lanczos"},
    "quality_2x":{"scale": 2, "method": "realesrgan"},
    "ultra_4x":  {"scale": 4, "method": "realesrgan"},
    "anime_4x":  {"scale": 4, "method": "realesrgan_anime"},
}


@dataclass
class Recommendation:
    preset: str
    reason: str
    priority: int   # higher = more strongly recommended
    face_restore: bool = False


def _bitrate_bps(v: Video) -> Optional[float]:
    if not v.duration_sec or v.duration_sec <= 0:
        return None
    return (v.size_bytes * 8) / v.duration_sec


def recommend(v: Video) -> List[Recommendation]:
    """Return recommendations sorted by priority (best first). Empty list
    if the video looks fine already."""
    out: List[Recommendation] = []
    if not v.width or not v.height:
        return out  # can't recommend without dimensions
    w, h = v.width, v.height
    br = _bitrate_bps(v)
    px = w * h

    # Sub-720p - always upscale-friendly.
    if h < 720 and w < 1280:
        out.append(Recommendation(
            "quality_2x",
            reason=f"Low resolution ({w}×{h}). 2× brings it near 1080p.",
            priority=5,
            face_restore=True,
        ))
    elif h < 1080:
        out.append(Recommendation(
            "quality_2x",
            reason=f"Below 1080p ({w}×{h}). 2× cleans and sharpens for HD screens.",
            priority=3,
        ))
    else:
        out.append(Recommendation(
            "ultra_4x",
            reason=f"Already ≥1080p. 4× gets you closer to 4K.",
            priority=1,
        ))

    # Very low bitrate - the video is likely compression-scarred; grain +
    # blocking are exactly what Real-ESRGAN "quality" models fix.
    if br is not None and br < 1_200_000 and px < 1920 * 1080:
        out.append(Recommendation(
            "quality_2x",
            reason=f"Low bitrate ({br/1e6:.1f} Mbps) - heavy compression artifacts.",
            priority=4,
            face_restore=True,
        ))

    # Older codecs sometimes benefit even at higher res.
    if (v.codec or "").lower() in {"mpeg1video", "mpeg2video", "wmv3", "wmv2", "rv40", "rv30"}:
        out.append(Recommendation(
            "quality_2x",
            reason=f"Legacy codec ({v.codec}). Re-encoding with a modern pipeline improves fidelity.",
            priority=2,
        ))

    # De-duplicate by preset, keep highest-priority entry per preset.
    dedup: Dict[str, Recommendation] = {}
    for r in out:
        prev = dedup.get(r.preset)
        if prev is None or r.priority > prev.priority:
            dedup[r.preset] = r
    ranked = sorted(dedup.values(), key=lambda r: -r.priority)
    return ranked


# ---------------------------------------------------------------------------
# Face restoration (GFPGAN, best-effort)
# ---------------------------------------------------------------------------

def face_restore_available() -> bool:
    try:
        import gfpgan  # noqa: F401
        return True
    except Exception:
        return False


def _apply_face_restore(input_path: str, output_path: str,
                         progress_cb: Optional[Callable[[float, str], None]] = None) -> bool:
    """Frame-wise GFPGAN pass, best-effort. Returns True on success.
    If gfpgan / weights aren't available, returns False and the caller
    keeps the un-restored file."""
    try:
        from gfpgan import GFPGANer  # type: ignore
        import cv2
    except Exception as e:
        logger.info("GFPGAN not usable (%s); skipping face restore", e)
        return False

    model_path = os.environ.get(
        "GFPGAN_MODEL",
        os.path.expanduser("~/models/gfpgan/GFPGANv1.4.pth"),
    )
    if not os.path.isfile(model_path):
        logger.info("GFPGAN weights missing at %s; skipping face restore", model_path)
        return False

    try:
        restorer = GFPGANer(model_path=model_path, upscale=1,
                            arch="clean", channel_multiplier=2, bg_upsampler=None)
    except Exception as e:
        logger.info("GFPGAN load failed (%s); skipping", e)
        return False

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return False
    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    temp_dir = tempfile.mkdtemp(prefix="face_restore_")
    frames_dir = os.path.join(temp_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            _, _, out = restorer.enhance(frame, has_aligned=False, only_center_face=False)
            path = os.path.join(frames_dir, f"{idx:06d}.png")
            cv2.imwrite(path, out if out is not None else frame)
            idx += 1
            if progress_cb and (idx % 5 == 0 or idx == total):
                progress_cb(idx / max(total, 1), f"Face restore {idx}/{total}")
        cap.release()

        # Re-encode via ffmpeg - preserves audio from the source.
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-framerate", str(fps),
            "-i", os.path.join(frames_dir, "%06d.png"),
            "-i", input_path,
            "-map", "0:v", "-map", "1:a?",
            "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-pix_fmt", "yuv420p", "-c:a", "copy",
            output_path,
        ]
        proc = subprocess.run(cmd, capture_output=True, timeout=1200, check=False)
        return proc.returncode == 0 and os.path.isfile(output_path)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Enhance job runner
# ---------------------------------------------------------------------------

class EnhanceJob(threading.Thread):
    """Runs one enhancement pass on a single video."""

    def __init__(self, store: LibraryStore, video_id: int, preset: str,
                 output_dir: str,
                 on_update: Callable[[dict], None],
                 face_restore: bool = False,
                 cancel_event: Optional[threading.Event] = None):
        super().__init__(daemon=True, name=f"library-enhance-{video_id}-{int(time.time())}")
        self.store = store
        self.video_id = video_id
        self.preset = preset
        self.output_dir = output_dir
        self.on_update = on_update
        self.face_restore = face_restore
        self._cancel = cancel_event or threading.Event()
        self._error: Optional[str] = None

    def cancel(self) -> None: self._cancel.set()
    def _cancelled(self) -> bool: return self._cancel.is_set()

    def _push(self, message: str, progress: float, status: str = "running", **extras) -> None:
        try:
            self.on_update({
                "status": status, "progress": max(0.0, min(1.0, progress)),
                "message": message, "error": self._error, **extras,
            })
        except Exception:
            logger.exception("on_update raised; dropping")

    def run(self) -> None:
        v = self.store.get_video(self.video_id)
        if not v or not os.path.isfile(v.abs_path):
            self._error = "Source media not found"
            self._push(self._error, 1.0, "failed")
            return
        preset = PRESETS.get(self.preset)
        if not preset:
            self._error = f"Unknown preset: {self.preset}"
            self._push(self._error, 1.0, "failed")
            return
        os.makedirs(self.output_dir, exist_ok=True)
        is_image = (v.kind == "image")
        ext = ".png" if is_image else ".mp4"
        out_name = f"{v.id}__{self.preset}{'__face' if self.face_restore else ''}{ext}"
        out_path = os.path.join(self.output_dir, out_name)

        try:
            from upscaler import upscale_video, upscale_image
        except Exception as e:
            self._error = f"Upscaler unavailable: {e}"
            self._push(self._error, 1.0, "failed")
            return

        def _cb(cur: int, total: int, msg: str) -> None:
            if self._cancelled():
                raise RuntimeError("cancelled")
            self._push(msg, cur / max(total, 1) * (0.8 if self.face_restore else 1.0))

        try:
            self._push(f"Starting {'image' if is_image else 'video'} upscale…", 0.02)
            if is_image:
                upscale_image(
                    v.abs_path,
                    scale=preset["scale"],
                    method=preset["method"],
                    output_path=out_path,
                )
                self._push("Upscale complete", 0.90 if self.face_restore else 1.0)
            else:
                upscale_video(
                    v.abs_path,
                    scale=preset["scale"],
                    method=preset["method"],
                    output_path=out_path,
                    progress_callback=_cb,
                )
        except RuntimeError as e:
            if str(e) == "cancelled":
                self._push("Cancelled", 1.0, "cancelled")
                return
            self._error = str(e)
            self._push(self._error, 1.0, "failed")
            return
        except Exception as e:
            self._error = f"Upscale failed: {e}"
            self._push(self._error, 1.0, "failed")
            return

        # Optional face restoration pass on the upscaled video. Image-mode
        # face restore is a future add - GFPGAN accepts still images too,
        # but for T3 we keep it to the video path (which the recommender
        # already flags for us).
        if self.face_restore and not is_image and not self._cancelled():
            self._push("Face restoration…", 0.82)
            face_out = out_path.replace(".mp4", "_face.mp4")
            ok = _apply_face_restore(out_path, face_out,
                                     progress_cb=lambda p, m: self._push(m, 0.80 + 0.20 * p))
            if ok and os.path.isfile(face_out):
                try:
                    os.remove(out_path)
                except OSError: pass
                out_path = face_out

        self._push("Done", 1.0, "completed", output_path=out_path)


def sidecar_output_for(root_dir: str) -> str:
    return os.path.join(root_dir, "enhanced")
