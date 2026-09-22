"""Background whisper transcription for library videos.

Reuses the existing top-level ``transcriber.Transcriber`` (faster-whisper)
so both Live Transcribe and this share the same engine + model cache.

Design mirrors the other library jobs:
- Cooperative cancel via a threading.Event
- Same on_update contract as ScanJob/EmbedJob so the shared jobs dict
  serializes progress just like any other render job.

The job iterates videos that don't yet have a transcript for the current
model. Each transcript is persisted as JSON in the store and mirrored
into the FTS5 index (`transcripts_fts`) for fast text search.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Callable, Dict, List, Optional

from .store import LibraryStore

logger = logging.getLogger("studiolite.library.transcribe")


DEFAULT_MODEL_SIZE = os.environ.get("STUDIOLITE_LIBRARY_WHISPER_MODEL", "tiny")


class TranscribeJob(threading.Thread):
    """Transcribe every un-indexed library video sequentially. Cheap models
    (`tiny`, `base`) are the sweet spot for library indexing — the goal is
    "did anyone say X" rather than a broadcast-quality transcript."""

    def __init__(self, store: LibraryStore,
                 on_update: Callable[[dict], None],
                 cancel_event: Optional[threading.Event] = None,
                 model_size: str = DEFAULT_MODEL_SIZE,
                 language: Optional[str] = None):
        super().__init__(daemon=True, name=f"library-transcribe-{int(time.time())}")
        self.store = store
        self.on_update = on_update
        self._cancel = cancel_event or threading.Event()
        self.model_size = model_size
        self.language = language
        self.counts = {"total": 0, "done": 0, "skipped": 0, "empty": 0}
        self._error: Optional[str] = None
        # `transcript_model` we key off is a compound so switching model size
        # forces a re-transcribe of everything, which is what a user asking
        # for a bigger model would expect.
        self._model_key = f"faster-whisper:{model_size}"

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
                "model": self._model_key,
            })
        except Exception:
            logger.exception("on_update raised; dropping")

    def run(self) -> None:
        try:
            from transcriber import Transcriber, TranscriptionConfig, check_whisperx_installed
        except Exception as e:
            self._error = f"faster-whisper unavailable: {e}"
            self._push(self._error, 1.0, "failed")
            return

        ok, msg = check_whisperx_installed()
        if not ok:
            self._error = msg
            self._push(msg, 1.0, "failed")
            return

        remaining = self.store.iter_videos_needing_transcript(
            limit=200_000, model_id=self._model_key,
        )
        self.counts["total"] = len(remaining)
        if not remaining:
            self._push("Nothing to transcribe — everything is already indexed.",
                       1.0, "completed")
            return

        # Reuse one Transcriber across all videos so the model loads once.
        cfg = TranscriptionConfig(
            model_size=self.model_size,
            language=self.language,
            compute_type="float32",  # `transcribe` downshifts to int8 on CPU.
        )
        transcriber = Transcriber(cfg)
        self._push(f"Loading {self.model_size}…", 0.02)

        for i, v in enumerate(remaining, 1):
            if self._cancelled():
                self._push("Cancelled", i / max(self.counts["total"], 1), "cancelled")
                return
            if not os.path.isfile(v.abs_path):
                self.store.mark_missing(v.id)
                self.counts["skipped"] += 1
                continue

            def _cb(msg: str, _idx: int = i) -> None:
                # Fires while whisper crunches this one file; we can't feed
                # per-video sub-progress into the parent bar without more
                # plumbing, so just surface the model's own status text.
                if not self._cancelled():
                    self._push(f"[{_idx}/{self.counts['total']}] {msg}",
                               (i - 1 + 0.5) / max(self.counts["total"], 1))

            try:
                result = transcriber.transcribe(v.abs_path, progress_callback=_cb)
            except Exception as e:
                logger.warning("whisper failed on %s: %s", v.abs_path, e)
                self.counts["skipped"] += 1
                self._push(f"[{i}/{self.counts['total']}] error on {os.path.basename(v.abs_path)}",
                           i / max(self.counts["total"], 1))
                continue

            if not result.get("success"):
                self.counts["skipped"] += 1
                self._push(f"[{i}/{self.counts['total']}] skipped: {result.get('error', '?')}",
                           i / max(self.counts["total"], 1))
                continue

            segments = result.get("segments") or []
            text = str(result.get("text") or "").strip()
            if not text and not segments:
                # No speech at all — record an empty transcript so we don't
                # keep re-processing this file forever.
                payload = {"language": result.get("language"), "text": "",
                           "segments": [], "empty": True}
                self.store.set_transcript(v.id, payload, self._model_key)
                self.counts["empty"] += 1
            else:
                payload = {
                    "language": result.get("language"),
                    "text": text,
                    "segments": [
                        {"start": float(s.get("start", 0)), "end": float(s.get("end", 0)),
                         "text": str(s.get("text", "")).strip()}
                        for s in segments if isinstance(s, dict)
                    ],
                }
                self.store.set_transcript(v.id, payload, self._model_key)
                self.counts["done"] += 1

            self._push(
                f"[{i}/{self.counts['total']}] {os.path.basename(v.abs_path)}",
                i / max(self.counts["total"], 1),
            )

        self._push("Speech indexing complete", 1.0, "completed")
