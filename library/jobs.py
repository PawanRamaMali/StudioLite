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

# T2 imports are lazy so a broken transformers install or missing weights
# doesn't break unrelated T1 stages that don't need them.

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


class EmbedJob(threading.Thread):
    """Encode CLIP embeddings + zero-shot tags for every video that doesn't
    have one for the current model. Cooperative cancel, in-place progress
    updates via the same on_update contract as ScanJob so the Jobs panel
    treats it identically."""

    def __init__(self, store: LibraryStore,
                 on_update: Callable[[dict], None],
                 cancel_event: Optional[threading.Event] = None,
                 with_tags: bool = True):
        super().__init__(daemon=True, name=f"library-embed-{int(time.time())}")
        self.store = store
        self.on_update = on_update
        self._cancel = cancel_event or threading.Event()
        self._with_tags = with_tags
        self.counts = {"total": 0, "embedded": 0, "tagged": 0, "skipped": 0}
        self._error: Optional[str] = None

    def cancel(self) -> None: self._cancel.set()
    def _cancelled(self) -> bool: return self._cancel.is_set()

    def _push(self, message: str, progress: float, status: str = "running") -> None:
        try:
            self.on_update({
                "status": "cancelled" if self._cancelled() and status == "running" else status,
                "progress": max(0.0, min(1.0, progress)),
                "message": message,
                "counts": dict(self.counts),
                "error": self._error,
            })
        except Exception:
            logger.exception("on_update raised; dropping")

    def run(self) -> None:
        try:
            from . import embeddings as _emb, tagger as _tagger
        except Exception as e:
            self._error = f"embed backend unavailable: {e}"
            self._push(self._error, 1.0, "failed")
            return
        try:
            rt = _emb.get_runtime()
        except _emb.EmbeddingUnavailable as e:
            self._error = str(e)
            self._push(self._error, 1.0, "failed")
            return
        except Exception as e:
            self._error = f"CLIP init failed: {e}"
            self._push(self._error, 1.0, "failed")
            return

        model_id = getattr(_emb, "_MODEL_ID", "openai/clip-vit-base-patch32")
        remaining = self.store.iter_videos_needing_embedding(limit=200_000, model_id=model_id)
        self.counts["total"] = len(remaining)
        if not remaining:
            self._push("Nothing to embed — all videos already indexed.", 1.0, "completed")
            return

        self._push(f"Encoding {len(remaining)} videos on {rt.device}…", 0.02)
        # Pre-warm the tag prompt matrix so per-video tag inference stays cheap.
        if self._with_tags:
            try: _tagger._ensure_prompts()
            except Exception: pass

        import json as _json
        for i, v in enumerate(remaining, 1):
            if self._cancelled():
                self._push("Cancelled", i / max(len(remaining), 1), "cancelled")
                return
            if not os.path.isfile(v.abs_path):
                self.store.mark_missing(v.id)
                self.counts["skipped"] += 1
                continue
            try:
                emb = _emb.encode_video(v.abs_path, v.duration_sec)
            except Exception as e:
                logger.warning("encode_video failed on %s: %s", v.abs_path, e)
                emb = None
            if emb is None:
                self.counts["skipped"] += 1
                if i % 2 == 0 or i == len(remaining):
                    self._push(f"Encoding… {i}/{len(remaining)} (last skipped)",
                               i / max(len(remaining), 1))
                continue
            tag_json = None
            if self._with_tags:
                try:
                    tags = _tagger.tag_embedding(emb)
                    tag_json = _json.dumps([{"tag": t.tag, "score": round(t.score, 4)}
                                            for t in tags])
                    if tags: self.counts["tagged"] += 1
                except Exception as e:
                    logger.warning("tagger failed on %s: %s", v.abs_path, e)
            self.store.set_embedding(v.id, _emb.pack_embedding(emb),
                                     model_id=model_id, tags_json=tag_json)
            self.counts["embedded"] += 1
            if i % 2 == 0 or i == len(remaining):
                self._push(f"Encoding… {i}/{len(remaining)}",
                           i / max(len(remaining), 1))

        self._push("Encoding complete", 1.0, "completed")
