#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Live screen OCR (Screen Transcribe).

Mirrors the live-audio pipeline: a stream of frames comes in, the model
runs over each one, deduplicated text events come out, everything is
persisted to disk incrementally.

Two input sources are supported:

  1. Browser frames - the client captures via getDisplayMedia, samples
     to JPEG at the chosen fps, and pushes binary messages over the WS.
  2. Local capture  - the server grabs frames directly from a chosen
     monitor / region using `mss`.

The smart layer prevents the obvious failure mode (spamming the log with
the same text every second) via:

  - frame-level perceptual hash diffing  (skip OCR on near-identical frames)
  - confidence floor                     (drop low-conf boxes)
  - rolling fuzzy dedup of emitted lines (handles scroll friction)
"""

from __future__ import annotations

import io
import json
import logging
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Engine cache
# ---------------------------------------------------------------------------

_OCR_CACHE: Dict[str, Any] = {}


def _get_ocr_engine(engine: str = "rapidocr"):
    """Lazy-load and cache the OCR engine. RapidOCR ships ONNX models;
    first call downloads them (~80 MB) and subsequent calls are instant."""
    if engine not in _OCR_CACHE:
        if engine == "rapidocr":
            from rapidocr_onnxruntime import RapidOCR
            logger.info("Loading RapidOCR ONNX models...")
            _OCR_CACHE[engine] = RapidOCR()
        else:
            raise ValueError(f"Unknown OCR engine: {engine!r}")
    return _OCR_CACHE[engine]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def enumerate_monitors() -> List[Dict[str, Any]]:
    """Return a list of monitors visible to mss, suitable for a picker UI."""
    try:
        import mss
    except ImportError:
        return []
    with mss.mss() as sct:
        out = []
        for i, m in enumerate(sct.monitors):
            # mss.monitors[0] is the virtual "all monitors" rectangle.
            label = "All monitors" if i == 0 else f"Monitor {i}"
            out.append({
                "index": i,
                "label": label,
                "left": int(m["left"]),
                "top": int(m["top"]),
                "width": int(m["width"]),
                "height": int(m["height"]),
            })
        return out


def _decode_image(frame: bytes):
    """Decode JPEG/PNG bytes to a PIL Image (RGB)."""
    from PIL import Image
    img = Image.open(io.BytesIO(frame))
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def _phash(img) -> "imagehash.ImageHash":
    import imagehash
    return imagehash.phash(img, hash_size=8)


def _line_similar(a: str, b: str, threshold: float) -> bool:
    """Cheap fuzzy match - good enough for OCR drift, OCR-vs-scroll, etc."""
    if not a or not b:
        return False
    if a == b:
        return True
    # Length-prefilter so we don't waste cycles on obvious non-matches
    ratio_len = min(len(a), len(b)) / max(len(a), len(b))
    if ratio_len < threshold - 0.1:
        return False
    return SequenceMatcher(None, a, b).ratio() >= threshold


_URL_RE = re.compile(r"https?://\S+", re.I)
_WS_RE = re.compile(r"\s+")


def _normalize_for_dedup(text: str) -> str:
    """Collapse whitespace and lowercase so exact-repeat detection is robust
    to OCR jitter that flips spacing but keeps the characters."""
    return _WS_RE.sub(" ", text).strip().lower()


def _is_garbage(text: str) -> bool:
    """Drop lines that aren't worth surfacing - browser/Office chrome misreads,
    URLs (which are visual noise in OCR), and single-glyph icons that
    confidently OCR to a single Chinese/symbol character."""
    s = text.strip()
    if not s:
        return True
    if len(s) < 3:
        # 1-2 character lines are almost always icon misreads like 'X', '十',
        # 'W', 'C'. Keep them only if they look like a real token (a digit
        # paired with a letter, e.g. 'v3').
        if not (any(c.isalpha() for c in s) and any(c.isdigit() for c in s)):
            return True
    # URL-dominated line
    url_chars = sum(len(m.group(0)) for m in _URL_RE.finditer(s))
    if url_chars and url_chars / len(s) >= 0.5:
        return True
    # Mostly non-alphanumeric: ribbon icons like '三V三三=田' or 'A☆'
    alnum = sum(c.isalnum() for c in s)
    if alnum / max(1, len(s)) < 0.4:
        return True
    return False


# ---------------------------------------------------------------------------
# Live screen OCR pipeline
# ---------------------------------------------------------------------------

@dataclass
class _EmittedLine:
    t: float
    text: str
    conf: float
    bbox: List[List[float]] = field(default_factory=list)


class LiveScreenOCR:
    """
    Frame-stream OCR with frame-diff, fuzzy dedup, and incremental persistence.

    Producer calls `process_frame(jpeg_bytes)` once per frame. The pipeline:
      1. Decode -> PIL image
      2. Crop to ROI if configured
      3. Perceptual hash; if too close to last frame, skip OCR entirely
      4. Run OCR (in caller's thread - callers should wrap in to_thread)
      5. Filter boxes below confidence floor
      6. Fuzzy-dedup line text against the rolling window
      7. Persist any new lines to .txt and .json
      8. Return {"new_lines": [...], "frame_skipped": bool, "frame_hash": str}
    """

    def __init__(
        self,
        session_id: str,
        transcripts_dir: str,
        engine: str = "rapidocr",
        language: Optional[str] = None,
        confidence_floor: float = 0.6,
        frame_diff_threshold: int = 4,
        dedup_similarity: float = 0.85,
        dedup_window: int = 80,
        max_repeats: int = 2,
        drop_garbage: bool = True,
        roi: Optional[Tuple[int, int, int, int]] = None,
    ):
        self.session_id = session_id
        self.transcripts_dir = transcripts_dir
        os.makedirs(transcripts_dir, exist_ok=True)

        self.engine_name = engine
        self.language = language
        self.confidence_floor = float(confidence_floor)
        self.frame_diff_threshold = int(frame_diff_threshold)
        self.dedup_similarity = float(dedup_similarity)
        self.dedup_window = int(dedup_window)
        self.max_repeats = max(1, int(max_repeats))
        self.drop_garbage = bool(drop_garbage)
        self.roi = roi  # (left, top, right, bottom) in image coords, or None

        self.start_time = time.time()
        self._engine = None
        self._last_phash = None

        # Rolling window of recent emitted text for fuzzy dedup
        self._recent_lines: List[str] = []
        # Session-wide exact-repeat counter (normalized text -> seen count).
        # Anything that hits self.max_repeats stops being emitted for the rest
        # of the session, which is what kills the per-page browser/Office
        # chrome ("File Home Insert...", URL bar, footer, etc.).
        self._seen_count: Dict[str, int] = defaultdict(int)
        self._all_lines: List[_EmittedLine] = []
        # Stats for filtered-out categories
        self.lines_dropped_garbage = 0
        self.lines_dropped_repeat = 0
        self.lines_dropped_fuzzy = 0
        self.lines_dropped_conf = 0

        self.txt_path = os.path.join(transcripts_dir, f"{session_id}.txt")
        self.json_path = os.path.join(transcripts_dir, f"{session_id}.json")
        self._txt_fh = open(self.txt_path, "w", encoding="utf-8")

        # Stats
        self.frames_received = 0
        self.frames_ocred = 0
        self.frames_skipped = 0

    # --- public API -------------------------------------------------------

    @property
    def device(self) -> str:
        # RapidOCR runs ONNX on CPU by default; no GPU on this box anyway.
        return "cpu"

    def ensure_engine(self) -> None:
        if self._engine is None:
            self._engine = _get_ocr_engine(self.engine_name)

    def process_frame(self, frame_bytes: bytes) -> Dict[str, Any]:
        """Run the full pipeline for a single frame. Safe to call from a
        thread (e.g. asyncio.to_thread)."""
        self.frames_received += 1
        try:
            img = _decode_image(frame_bytes)
        except Exception as e:
            logger.warning(f"frame decode failed: {e}")
            return {"new_lines": [], "frame_skipped": True, "reason": "decode_failed"}

        if self.roi is not None:
            try:
                img = img.crop(self.roi)
            except Exception:
                pass

        # Frame-diff via perceptual hash
        try:
            ph = _phash(img)
            if self._last_phash is not None:
                diff = ph - self._last_phash
                if diff <= self.frame_diff_threshold:
                    self.frames_skipped += 1
                    return {
                        "new_lines": [],
                        "frame_skipped": True,
                        "reason": "frame_unchanged",
                        "phash_diff": int(diff),
                    }
            self._last_phash = ph
        except Exception as e:
            logger.debug(f"phash failed (continuing without diff): {e}")

        self.ensure_engine()
        self.frames_ocred += 1

        # RapidOCR accepts numpy arrays directly
        import numpy as np
        arr = np.array(img)
        try:
            result, _elapse = self._engine(arr)
        except Exception as e:
            logger.warning(f"OCR call failed: {e}")
            return {"new_lines": [], "frame_skipped": True, "reason": "ocr_failed"}

        if not result:
            return {"new_lines": [], "frame_skipped": False}

        new_lines: List[Dict[str, Any]] = []
        now = time.time() - self.start_time
        for entry in result:
            if not entry or len(entry) < 3:
                continue
            bbox, text, conf = entry[0], entry[1], entry[2]
            text = (text or "").strip()
            try:
                conf = float(conf)
            except Exception:
                conf = 0.0
            if not text or conf < self.confidence_floor:
                self.lines_dropped_conf += 1
                continue
            if self.drop_garbage and _is_garbage(text):
                self.lines_dropped_garbage += 1
                continue
            norm = _normalize_for_dedup(text)
            # Exact-repeat cap - kills per-frame chrome (URL bar, ribbon,
            # tab title, page number) once we've seen it max_repeats times.
            if self._seen_count[norm] >= self.max_repeats:
                self._seen_count[norm] += 1  # still count it for stats
                self.lines_dropped_repeat += 1
                continue
            # Fuzzy match against the rolling window catches OCR drift
            # ("Editot" right after "Editor") and partial scroll re-emits.
            if any(_line_similar(prev, text, self.dedup_similarity)
                   for prev in self._recent_lines[-self.dedup_window:]):
                self.lines_dropped_fuzzy += 1
                continue
            self._seen_count[norm] += 1
            line = _EmittedLine(t=now, text=text, conf=conf, bbox=bbox if isinstance(bbox, list) else [])
            self._record(line)
            new_lines.append({"t": line.t, "text": line.text, "conf": line.conf, "bbox": line.bbox})

        return {"new_lines": new_lines, "frame_skipped": False}

    def flush(self) -> Dict[str, Any]:
        try:
            self._txt_fh.close()
        except Exception:
            pass
        self._dump_json()
        return {
            "complete": True,
            "stats": {
                "frames_received": self.frames_received,
                "frames_ocred": self.frames_ocred,
                "frames_skipped": self.frames_skipped,
                "lines_emitted": len(self._all_lines),
                "lines_dropped_conf": self.lines_dropped_conf,
                "lines_dropped_garbage": self.lines_dropped_garbage,
                "lines_dropped_repeat": self.lines_dropped_repeat,
                "lines_dropped_fuzzy": self.lines_dropped_fuzzy,
            },
        }

    def transcript_urls(self, base_url: str = "") -> Dict[str, str]:
        return {
            "txt": f"{base_url}/{self.session_id}.txt",
            "json": f"{base_url}/{self.session_id}.json",
        }

    # --- internals --------------------------------------------------------

    def _record(self, line: _EmittedLine) -> None:
        self._all_lines.append(line)
        self._recent_lines.append(line.text)
        if len(self._recent_lines) > self.dedup_window * 4:
            self._recent_lines = self._recent_lines[-self.dedup_window * 2:]
        ts = _format_clock(line.t)
        self._txt_fh.write(f"[{ts}] {line.text}\n")
        self._txt_fh.flush()
        # Defer the JSON dump until flush() - rewriting it per-line is wasteful

    def _dump_json(self) -> None:
        try:
            with open(self.json_path, "w", encoding="utf-8") as f:
                json.dump({
                    "session": self.session_id,
                    "engine": self.engine_name,
                    "lines": [
                        {"t": l.t, "text": l.text, "conf": l.conf, "bbox": l.bbox}
                        for l in self._all_lines
                    ],
                    "stats": {
                        "frames_received": self.frames_received,
                        "frames_ocred": self.frames_ocred,
                        "frames_skipped": self.frames_skipped,
                    },
                }, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"json dump failed: {e}")


def _format_clock(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# ---------------------------------------------------------------------------
# Server-side local capture (mss)
# ---------------------------------------------------------------------------

def capture_local_jpeg(monitor_index: int = 0, region: Optional[Tuple[int, int, int, int]] = None,
                       quality: int = 70) -> bytes:
    """Grab one frame from a monitor (or a sub-region) and return JPEG bytes.

    monitor_index: index into mss.monitors. 0 is the virtual "all monitors"
                   rectangle, 1+ are individual physical displays.
    region:        (left, top, width, height) relative to the chosen monitor's
                   top-left, or None for the whole monitor.
    quality:       JPEG quality 1-95.
    """
    import mss
    from PIL import Image

    with mss.mss() as sct:
        monitors = sct.monitors
        if not 0 <= monitor_index < len(monitors):
            raise ValueError(f"monitor_index {monitor_index} out of range (have {len(monitors)})")
        mon = dict(monitors[monitor_index])
        if region is not None:
            l, t, w, h = region
            mon = {
                "left": mon["left"] + int(l),
                "top": mon["top"] + int(t),
                "width": int(w),
                "height": int(h),
            }
        shot = sct.grab(mon)
        img = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        return buf.getvalue()
