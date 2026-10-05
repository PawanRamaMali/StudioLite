"""Face detection + embedding for the library.

Uses the OpenCV Zoo pair:
  - YuNet (`cv2.FaceDetectorYN`)  — ~230 KB, Apache 2.0, ~0.7 ms/frame CPU
  - SFace (`cv2.FaceRecognizerSF`) — ~37 MB, Apache 2.0, 128-d L2-normalized

Both are first-party in `opencv-python-headless`, so NO new Python deps.
The ONNX files aren't bundled; on first use we download them to
`<library_dir>/face_models/` from the OpenCV Zoo repo. If the download
fails (offline, firewall), we surface a clear "models not present" error
and the FaceIndexJob reports failed without touching any rows.

The embedding is stored packed via `library.embeddings.pack_embedding` so
the existing `unpack_embedding(blob, dim=128)` round-trips cleanly.
"""
from __future__ import annotations

import logging
import math
import os
import threading
import urllib.request
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger("studiolite.library.faces")


# Pinned filenames + URLs from the OpenCV Zoo (stable raw-paths on `main`).
# Both are tiny — downloading once lives alongside the library DB.
_YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/"
    "models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
_SFACE_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/"
    "models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
)
_YUNET_NAME = "face_detection_yunet_2023mar.onnx"
_SFACE_NAME = "face_recognition_sface_2021dec.onnx"

# SFace cosine-match threshold per OpenCV docs. 0.6 is the paper default
# for "same person"; we use it for the online centroid-assign step. The
# re-cluster path uses the tighter 0.5 threshold.
SFACE_COSINE_SAME_PERSON = 0.60

EMBED_DIM = 128
EMBED_MODEL = "sface-2021dec"


class FacesUnavailable(RuntimeError):
    """Raised when either model file isn't present and can't be downloaded.
    Caller should degrade cleanly (job marked failed with a clear message)."""


# ---------------------------------------------------------------------------
# Model files on disk
# ---------------------------------------------------------------------------

def models_dir(library_dir: str) -> str:
    return os.path.join(library_dir, "face_models")


def models_present(library_dir: str) -> bool:
    d = models_dir(library_dir)
    return (os.path.isfile(os.path.join(d, _YUNET_NAME)) and
            os.path.isfile(os.path.join(d, _SFACE_NAME)))


def ensure_models(library_dir: str, *, allow_download: bool = True,
                   timeout: float = 60.0) -> Tuple[str, str]:
    """Return (yunet_path, sface_path). Downloads from OpenCV Zoo on first
    use when `allow_download` is True and the file isn't present. Raises
    `FacesUnavailable` on failure."""
    d = models_dir(library_dir)
    os.makedirs(d, exist_ok=True)
    yunet_path = os.path.join(d, _YUNET_NAME)
    sface_path = os.path.join(d, _SFACE_NAME)

    def _fetch(url: str, dest: str) -> None:
        if os.path.isfile(dest) and os.path.getsize(dest) > 0:
            return
        if not allow_download:
            raise FacesUnavailable(
                f"Face model missing at {dest} and download disabled. "
                "Place the file manually or set STUDIOLITE_FACES_DOWNLOAD=1."
            )
        tmp = dest + ".tmp"
        try:
            logger.info("Downloading %s → %s", url, dest)
            urllib.request.urlretrieve(url, tmp)  # noqa: S310 — explicit opt-in
            os.replace(tmp, dest)
        except Exception as e:
            if os.path.exists(tmp):
                try: os.remove(tmp)
                except OSError: pass
            raise FacesUnavailable(
                f"Could not download face model from {url}: {e}. "
                "Download manually and drop it under <library>/face_models/."
            ) from e

    _fetch(_YUNET_URL, yunet_path)
    _fetch(_SFACE_URL, sface_path)
    return yunet_path, sface_path


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------

@dataclass
class FaceDetection:
    bbox: Tuple[float, float, float, float]   # x, y, w, h
    det_score: float
    embedding: np.ndarray                     # (128,) float32 L2-normalized
    aligned_bgr: np.ndarray                   # 112x112x3 BGR for thumbnail


class FaceRuntime:
    """Loads YuNet + SFace lazily. OpenCV is not fork-safe when torch is
    loaded in the same process; the detector sets thread count to 2 to
    reduce contention with concurrent CLIP / whisper work."""

    def __init__(self, library_dir: str, *, allow_download: bool = True):
        self._library_dir = library_dir
        self._allow_download = allow_download
        self._lock = threading.RLock()
        self._detector = None
        self._embedder = None
        self._size: Optional[Tuple[int, int]] = None  # last input size

    def _ensure(self):
        if self._detector is not None and self._embedder is not None:
            return
        import cv2  # local import so modules that never touch faces don't pay
        yunet_path, sface_path = ensure_models(
            self._library_dir, allow_download=self._allow_download,
        )
        # Keep OpenCV from grabbing every core — CLIP/whisper are busy elsewhere.
        try: cv2.setNumThreads(2)
        except Exception: pass
        with self._lock:
            if self._detector is None:
                self._detector = cv2.FaceDetectorYN.create(
                    yunet_path, "", (320, 320),
                    score_threshold=0.6, nms_threshold=0.3, top_k=25,
                )
            if self._embedder is None:
                self._embedder = cv2.FaceRecognizerSF.create(sface_path, "")

    def detect_and_embed(self, bgr: np.ndarray) -> List[FaceDetection]:
        """Return every detected face with its SFace embedding + aligned
        112×112 BGR crop (usable as a thumbnail)."""
        if bgr is None or bgr.size == 0:
            return []
        self._ensure()
        import cv2
        h, w = bgr.shape[:2]
        if (w, h) != self._size:
            self._detector.setInputSize((w, h))  # type: ignore[union-attr]
            self._size = (w, h)
        _, faces = self._detector.detect(bgr)   # type: ignore[union-attr]
        if faces is None:
            return []
        out: List[FaceDetection] = []
        for row in faces:
            # row = [x, y, w, h, right_eye_x, right_eye_y, left_eye_x, left_eye_y,
            #        nose_x, nose_y, right_mouth_x, right_mouth_y,
            #        left_mouth_x, left_mouth_y, confidence]
            try:
                aligned = self._embedder.alignCrop(bgr, row)   # type: ignore[union-attr]
                emb = self._embedder.feature(aligned)          # type: ignore[union-attr]
            except Exception as e:
                logger.debug("alignCrop/feature failed on one face: %s", e)
                continue
            emb = np.asarray(emb, dtype=np.float32).flatten()
            if emb.size != EMBED_DIM:
                continue
            n = float(np.linalg.norm(emb))
            if n < 1e-8:
                continue
            emb = emb / n
            out.append(FaceDetection(
                bbox=(float(row[0]), float(row[1]), float(row[2]), float(row[3])),
                det_score=float(row[14]),
                embedding=emb.astype(np.float32),
                aligned_bgr=aligned,
            ))
        return out


# ---------------------------------------------------------------------------
# Cosine helpers (shared with people.py)
# ---------------------------------------------------------------------------

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Both must be L2-normalized float32; returns a plain float."""
    return float(np.dot(a, b))


def running_mean_add(centroid: Optional[np.ndarray], count: int,
                     emb: np.ndarray) -> Tuple[np.ndarray, int]:
    """Add one embedding to a running-mean centroid, keep it unit-length.
    Returns (new_centroid, new_count)."""
    emb = emb.astype(np.float32)
    if centroid is None or count <= 0:
        return emb.copy(), 1
    new = (centroid * count + emb) / (count + 1)
    norm = float(np.linalg.norm(new))
    if norm > 1e-8:
        new = new / norm
    return new.astype(np.float32), count + 1
