"""Batch re-encode of legacy-codec videos into modern H.264 / H.265.

Aimed at library maintenance: MPEG-2 rips, WMV screencasts, RM/RMVB
downloads, DV camcorder footage — anything that most consumer players
still handle but that no longer plays cleanly in browsers, phones, or
modern editors. Runs ffmpeg per file with a cooperative cancel event.

Design mirrors ScanJob / EmbedJob / TranscribeJob so the shared jobs
dict serializes progress the same way.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import threading
import time
from typing import Any, Callable, Dict, List, Optional

from .store import LibraryStore

logger = logging.getLogger("studiolite.library.reencode")


# Codec names ffprobe/ffmpeg use, lowercased. Anything not in this set is
# considered modern-enough to keep as-is — re-encoding h264/hevc/av1/vp9
# just recompresses without benefit and can *lose* quality.
LEGACY_CODECS = frozenset({
    # MPEG family
    "mpeg1video", "mpeg2video", "mpeg4",  # DivX/Xvid also identify as mpeg4
    "msmpeg4v1", "msmpeg4v2", "msmpeg4v3",
    # Windows Media
    "wmv1", "wmv2", "wmv3", "vc1",
    # Real
    "rv10", "rv20", "rv30", "rv40",
    # DV / analog capture
    "dv", "dvvideo",
    # Legacy / unusual
    "h263", "flv1", "flv", "svq3", "cinepak", "indeo", "theora",
    "rawvideo",
})


TARGET_CODECS = {
    # (video codec, target file extension, ffmpeg encoder, extra args)
    "h264": ("h264", ".mp4", "libx264", ["-preset", "medium"]),
    "h265": ("hevc", ".mp4", "libx265", ["-preset", "medium", "-tag:v", "hvc1"]),
}


class ReencodeJob(threading.Thread):
    """ffmpeg-convert a list of videos into a modern codec.

    Options:
        video_ids       — library ids to process (ordered)
        target_codec    — "h264" (default) or "h265"
        crf             — quality knob; 18 is visually lossless, 23 is
                          "the internet default", 28 is small-file
        replace_original— if True, move the source to `<name>.legacy` and
                          replace it with the transcoded file so the
                          library keeps the same path. If False (default),
                          writes to `.mp/library/reencoded/<id>__<codec>.mp4`.
    """

    def __init__(self, store: LibraryStore, video_ids: List[int],
                 output_dir: str,
                 on_update: Callable[[dict], None],
                 target_codec: str = "h264",
                 crf: int = 20,
                 replace_original: bool = False,
                 cancel_event: Optional[threading.Event] = None):
        super().__init__(daemon=True, name=f"library-reencode-{int(time.time())}")
        self.store = store
        self.video_ids = video_ids
        self.output_dir = output_dir
        self.target_codec = target_codec if target_codec in TARGET_CODECS else "h264"
        self.crf = max(0, min(51, int(crf)))
        self.replace_original = replace_original
        self.on_update = on_update
        self._cancel = cancel_event or threading.Event()
        self._proc: Optional[subprocess.Popen] = None
        self.results: List[Dict[str, Any]] = []
        self.counts = {"total": len(video_ids), "done": 0, "skipped": 0,
                       "failed": 0, "bytes_saved": 0}

    def cancel(self) -> None:
        self._cancel.set()
        # Also kill the running ffmpeg so it doesn't finish the file we
        # just cancelled and waste more CPU.
        proc = self._proc
        if proc is not None and proc.poll() is None:
            try: proc.terminate()
            except Exception: pass

    def _cancelled(self) -> bool: return self._cancel.is_set()

    def _push(self, message: str, progress: float, status: str = "running",
              **extras) -> None:
        try:
            self.on_update({
                "status": "cancelled" if self._cancelled() and status == "running" else status,
                "progress": max(0.0, min(1.0, progress)),
                "message": message,
                "counts": dict(self.counts),
                **extras,
            })
        except Exception:
            logger.exception("on_update raised; dropping")

    def run(self) -> None:
        if self.counts["total"] == 0:
            self._push("Nothing to re-encode", 1.0, "completed")
            return

        _, ext, encoder, extra_args = TARGET_CODECS[self.target_codec]
        os.makedirs(self.output_dir, exist_ok=True)

        for i, vid in enumerate(self.video_ids, 1):
            if self._cancelled():
                self._push("Cancelled", i / max(self.counts["total"], 1), "cancelled")
                return

            v = self.store.get_video(vid)
            if not v or not os.path.isfile(v.abs_path):
                self.counts["skipped"] += 1
                self.results.append({"id": vid, "status": "skipped",
                                     "reason": "source missing"})
                continue

            src = v.abs_path
            src_size = v.size_bytes or os.path.getsize(src)
            if self.replace_original:
                # Write to a sibling temp file, keep the original at src+".legacy"
                # only after the transcode succeeds. Prevents a partial file
                # from overwriting the original on Ctrl-C.
                out_path = os.path.splitext(src)[0] + f".reenc{ext}"
            else:
                out_path = os.path.join(
                    self.output_dir,
                    f"{v.id}__{self.target_codec}{ext}",
                )

            self._push(
                f"[{i}/{self.counts['total']}] {os.path.basename(src)} → {self.target_codec}",
                (i - 0.5) / max(self.counts["total"], 1),
            )

            cmd = [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
                "-i", src,
                "-c:v", encoder,
                *extra_args,
                "-crf", str(self.crf),
                "-pix_fmt", "yuv420p",
                # Keep audio if there is any, re-encoded to a portable AAC.
                "-c:a", "aac", "-b:a", "192k",
                # Move moov atom to the front so the output plays as it
                # downloads / streams over HTTP range requests.
                "-movflags", "+faststart",
                out_path,
            ]
            try:
                self._proc = subprocess.Popen(cmd,
                                              stdout=subprocess.DEVNULL,
                                              stderr=subprocess.PIPE)
                _, stderr = self._proc.communicate()
                rc = self._proc.returncode
                self._proc = None
            except FileNotFoundError:
                self._push("ffmpeg not on PATH", 1.0, "failed",
                           error="ffmpeg binary not found")
                return
            except Exception as e:
                self.counts["failed"] += 1
                self.results.append({"id": vid, "status": "failed",
                                     "reason": str(e)})
                continue

            # Cancel handling: if ffmpeg was killed mid-encode the return
            # code is non-zero and we'll drop the partial in the next branch.
            # If it finished cleanly BEFORE the cancel arrived, keep the file
            # — the user's work is done for that video, no point discarding.
            if rc != 0 and self._cancelled() and os.path.exists(out_path):
                try: os.remove(out_path)
                except OSError: pass

            if rc != 0 or not os.path.isfile(out_path) or os.path.getsize(out_path) < 1024:
                err = (stderr.decode("utf-8", errors="replace")[-400:] if stderr else "")
                self.counts["failed"] += 1
                self.results.append({"id": vid, "status": "failed",
                                     "reason": f"ffmpeg exit {rc}: {err}"})
                if os.path.exists(out_path):
                    try: os.remove(out_path)
                    except OSError: pass
                continue

            new_size = os.path.getsize(out_path)
            saved = max(0, src_size - new_size)

            if self.replace_original:
                legacy = src + ".legacy"
                try:
                    if os.path.exists(legacy): os.remove(legacy)
                    shutil.move(src, legacy)
                    shutil.move(out_path, src)
                    # Force a re-scan of this row on the next pass by clearing
                    # every derived signal — codec/size have all changed.
                    self.store.update_video_metadata(
                        vid, sha256=None, phash_hex=None,
                        duration_sec=None, width=None, height=None,
                        codec=None, fps=None, scanned_at=None,
                    )
                    self.counts["done"] += 1
                    self.counts["bytes_saved"] += saved
                    self.results.append({
                        "id": vid, "status": "ok",
                        "output_path": src, "backup_path": legacy,
                        "bytes_saved": saved, "old_size": src_size, "new_size": new_size,
                    })
                except Exception as e:
                    self.counts["failed"] += 1
                    self.results.append({"id": vid, "status": "failed",
                                         "reason": f"replace failed: {e}"})
            else:
                self.counts["done"] += 1
                self.counts["bytes_saved"] += saved
                self.results.append({
                    "id": vid, "status": "ok",
                    "output_path": out_path,
                    "bytes_saved": saved, "old_size": src_size, "new_size": new_size,
                })

            self._push(
                f"[{i}/{self.counts['total']}] done",
                i / max(self.counts["total"], 1),
            )

        # Summary
        final_status = "completed"
        if self.counts["failed"] == self.counts["total"]:
            final_status = "failed"
        self._push(
            f"Re-encoded {self.counts['done']} · saved {_fmt_bytes(self.counts['bytes_saved'])}",
            1.0, final_status, results=self.results,
        )


def _fmt_bytes(n: int) -> str:
    if not n or n < 1024: return f"{int(n)} B"
    units = ["KB", "MB", "GB", "TB"]
    v = n / 1024.0
    i = 0
    while v >= 1024 and i < len(units) - 1:
        v /= 1024; i += 1
    return f"{v:.1f} {units[i]}"
