"""Scan job runner. Threaded, cooperative, updates both the library
``scans`` table and the shared ``jobs`` dict (studiolite's existing
persistent job store) so the Jobs panel can show library scans next to
render jobs."""
from __future__ import annotations

import logging
import os
import threading
import time
from typing import Callable, Dict, List, Optional

from . import scanner, probe as probe_mod, phash as phash_mod, hasher
from .store import LibraryStore

logger = logging.getLogger("studiolite.library.jobs")


class ScanJob(threading.Thread):
    """Runs one full scan (discover -> probe -> hash -> phash) on background
    thread. Cooperatively yields to a cancel event between videos.

    The `on_update` callback receives dicts of {status, progress, message,
    counts:{discovered,probed,hashed,phashed,skipped}, scan_id}.
    That's the shape the API jobs mirror lays on top of the shared
    job-store dict.
    """

    def __init__(self, store: LibraryStore, root_ids: List[int],
                 on_update: Callable[[dict], None],
                 cancel_event: Optional[threading.Event] = None):
        super().__init__(daemon=True, name=f"library-scan-{int(time.time())}")
        self.store = store
        self.root_ids = root_ids
        self.on_update = on_update
        self._cancel = cancel_event or threading.Event()
        self.scan_id: Optional[int] = None
        self.counts = {"discovered": 0, "probed": 0, "hashed": 0,
                       "phashed": 0, "skipped": 0}
        self._error: Optional[str] = None

    def cancel(self) -> None:
        self._cancel.set()

    # ---- helpers -----------------------------------------------------------

    def _cancelled(self) -> bool:
        return self._cancel.is_set()

    def _push(self, message: str, progress: float) -> None:
        payload = {
            "status": "cancelled" if self._cancelled() else "running",
            "progress": max(0.0, min(1.0, progress)),
            "message": message,
            "counts": dict(self.counts),
            "scan_id": self.scan_id,
        }
        try:
            self.on_update(payload)
        except Exception:
            logger.exception("on_update raised; dropping")

    def _push_final(self, status: str, message: str, error: Optional[str] = None) -> None:
        payload = {
            "status": status,
            "progress": 1.0 if status == "completed" else self._progress_estimate(),
            "message": message,
            "counts": dict(self.counts),
            "scan_id": self.scan_id,
            "error": error,
        }
        try:
            self.on_update(payload)
        except Exception:
            logger.exception("on_update raised; dropping")

    def _progress_estimate(self) -> float:
        # Rough four-phase progress: discover 20% + probe 20% + hash 30% + phash 30%.
        d = self.counts["discovered"]
        if d == 0: return 0.0
        p = self.counts["probed"] / d
        h = self.counts["hashed"] / d
        ph = self.counts["phashed"] / d
        return round(0.20 + 0.20 * p + 0.30 * h + 0.30 * ph, 3)

    # ---- main --------------------------------------------------------------

    def run(self) -> None:
        self.scan_id = self.store.create_scan(self.root_ids)
        self.store.update_scan(self.scan_id or -1, status="running")
        try:
            self._push("Discovering files…", 0.02)
            self.counts["discovered"] = scanner.discover(
                self.store, self.root_ids,
                progress=lambda kind, n: self._push(f"Discovered {n} videos…", 0.02 + 0.18 * min(1.0, n / 500)),
                cancel=self._cancelled,
            )
            self.store.update_scan(self.scan_id, discovered=self.counts["discovered"])
            self._push(f"Discovered {self.counts['discovered']} videos", 0.20)

            if self._cancelled():
                self._push_final("cancelled", "Scan cancelled by user")
                self.store.update_scan(self.scan_id, status="cancelled",
                                       finished_at=time.time(), **self._scan_counts())
                return

            # --- probe (ffprobe metadata) ---
            self._probe_pass()
            if self._cancelled():
                self._push_final("cancelled", "Scan cancelled by user")
                self.store.update_scan(self.scan_id, status="cancelled",
                                       finished_at=time.time(), **self._scan_counts())
                return

            # --- hash (sha256) ---
            self._hash_pass()
            if self._cancelled():
                self._push_final("cancelled", "Scan cancelled by user")
                self.store.update_scan(self.scan_id, status="cancelled",
                                       finished_at=time.time(), **self._scan_counts())
                return

            # --- phash (perceptual) ---
            self._phash_pass()
            if self._cancelled():
                self._push_final("cancelled", "Scan cancelled by user")
                self.store.update_scan(self.scan_id, status="cancelled",
                                       finished_at=time.time(), **self._scan_counts())
                return

            self._push_final("completed", "Scan complete")
            self.store.update_scan(self.scan_id, status="done",
                                   finished_at=time.time(), **self._scan_counts())
        except Exception as e:
            logger.exception("library scan failed")
            self._error = str(e)
            self._push_final("failed", "Scan failed", error=str(e))
            self.store.update_scan(self.scan_id or -1, status="failed",
                                   finished_at=time.time(), error=str(e),
                                   **self._scan_counts())

    def _scan_counts(self) -> Dict[str, int]:
        return {"discovered": self.counts["discovered"],
                "hashed":     self.counts["hashed"],
                "phashed":    self.counts["phashed"],
                "skipped":    self.counts["skipped"]}

    # ---- passes ------------------------------------------------------------

    def _probe_pass(self) -> None:
        remaining = self.store.iter_videos_needing_probe(limit=200_000)
        n = len(remaining)
        for i, v in enumerate(remaining, 1):
            if self._cancelled(): return
            r = probe_mod.probe(v.abs_path)
            if r.duration_sec or r.width or r.height:
                self.store.update_video_metadata(
                    v.id,
                    duration_sec=r.duration_sec,
                    width=r.width, height=r.height,
                    codec=r.codec, fps=r.fps,
                    scanned_at=time.time(),
                )
                self.counts["probed"] += 1
            else:
                self.counts["skipped"] += 1
            if i % 25 == 0 or i == n:
                self._push(f"Probing metadata… {i}/{n}",
                           0.20 + 0.20 * (i / max(n, 1)))

    def _hash_pass(self) -> None:
        remaining = self.store.iter_videos_needing_hash(limit=200_000)
        n = len(remaining)
        for i, v in enumerate(remaining, 1):
            if self._cancelled(): return
            if not os.path.isfile(v.abs_path):
                self.store.mark_missing(v.id)
                self.counts["skipped"] += 1
                continue
            digest = hasher.sha256_file(v.abs_path, cancel=self._cancelled)
            if digest:
                self.store.update_video_metadata(v.id, sha256=digest)
                self.counts["hashed"] += 1
            else:
                self.counts["skipped"] += 1
            if i % 5 == 0 or i == n:
                self._push(f"Hashing files… {i}/{n}",
                           0.40 + 0.30 * (i / max(n, 1)))

    def _phash_pass(self) -> None:
        remaining = self.store.iter_videos_needing_phash(limit=200_000)
        n = len(remaining)
        for i, v in enumerate(remaining, 1):
            if self._cancelled(): return
            if not os.path.isfile(v.abs_path):
                self.store.mark_missing(v.id)
                self.counts["skipped"] += 1
                continue
            ph = phash_mod.phash_video(v.abs_path, v.duration_sec)
            if ph:
                self.store.update_video_metadata(v.id, phash_hex=ph)
                self.counts["phashed"] += 1
            else:
                self.counts["skipped"] += 1
            if i % 3 == 0 or i == n:
                self._push(f"Perceptual hashing… {i}/{n}",
                           0.70 + 0.30 * (i / max(n, 1)))
