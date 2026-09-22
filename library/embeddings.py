"""CLIP frame embeddings for semantic search + clustering.

Uses transformers + torch (already installed). Everything runs on CPU
by default; if CUDA is available and env var STUDIOLITE_EMBED_DEVICE
is not forced to "cpu", we use it.

Offline story: the CLIP model is loaded with `local_files_only=True` on
the first pass — if it isn't cached the user gets a clear "model not
downloaded" error, and the pipeline stays functional (search + clusters
just report empty). The launcher UI can offer a one-click download.

Frame sampling: reuse library.phash._extract_frames — same MJPEG-pipe
trick — so we don't shell out to ffmpeg twice per video.
"""
from __future__ import annotations

import io
import logging
import os
import subprocess
import threading
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from PIL import Image

logger = logging.getLogger("studiolite.library.embeddings")

# Small enough to run on CPU in reasonable time. If the user has a
# stronger GPU they can override via STUDIOLITE_EMBED_MODEL to use
# openai/clip-vit-large-patch14 or similar.
_MODEL_ID = os.environ.get("STUDIOLITE_EMBED_MODEL", "openai/clip-vit-base-patch32")
_DEVICE_OVERRIDE = os.environ.get("STUDIOLITE_EMBED_DEVICE", "").strip().lower()

FRAME_COUNT = 5
FRAME_WIDTH = 384
EMBED_DIM = 512  # base-patch32 is 512-d


class EmbeddingUnavailable(RuntimeError):
    """Raised when the CLIP model can't load — usually because the weights
    aren't in the local HuggingFace cache and we're running offline."""


@dataclass
class ClipRuntime:
    model: object
    processor: object
    device: str
    dim: int


_runtime_lock = threading.Lock()
_runtime: Optional[ClipRuntime] = None


def get_runtime() -> ClipRuntime:
    """Lazy-load the model on first use. Subsequent calls are ~free."""
    global _runtime
    if _runtime is not None:
        return _runtime
    with _runtime_lock:
        if _runtime is not None:
            return _runtime
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
        except Exception as e:
            raise EmbeddingUnavailable(
                f"Required libraries unavailable ({e}). Install torch + transformers."
            )
        device = _pick_device()
        try:
            processor = CLIPProcessor.from_pretrained(_MODEL_ID, local_files_only=True)
            model = CLIPModel.from_pretrained(_MODEL_ID, local_files_only=True)
        except Exception:
            # Retry allowing network — user may not have cached the model.
            try:
                processor = CLIPProcessor.from_pretrained(_MODEL_ID)
                model = CLIPModel.from_pretrained(_MODEL_ID)
            except Exception as e:
                raise EmbeddingUnavailable(
                    f"CLIP weights not cached and download failed ({e}). "
                    "Pre-download `openai/clip-vit-base-patch32` or set "
                    "STUDIOLITE_EMBED_MODEL to a locally-available checkpoint."
                )
        model.eval()
        model.to(device)
        _runtime = ClipRuntime(model=model, processor=processor, device=device,
                               dim=model.config.projection_dim)
        logger.info("CLIP loaded (%s) on %s dim=%d", _MODEL_ID, device, _runtime.dim)
        return _runtime


def _pick_device() -> str:
    try:
        import torch
        if _DEVICE_OVERRIDE in {"cpu", "cuda", "mps"}:
            return _DEVICE_OVERRIDE
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    except Exception:
        return "cpu"


# ---------------------------------------------------------------------------
# Frame extraction (shares the ffmpeg-pipe trick with phash.py)
# ---------------------------------------------------------------------------

def _extract_frames(path: str, duration_sec: Optional[float],
                    n: int = FRAME_COUNT, width: int = FRAME_WIDTH) -> List[Image.Image]:
    if not path or not os.path.isfile(path):
        return []
    if duration_sec and duration_sec > 0.5:
        rate = max(0.01, n / max(duration_sec, 1.0))
        vf = f"fps={rate},scale={width}:-2"
    else:
        vf = f"fps=1,scale={width}:-2"
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", path, "-vf", vf, "-vframes", str(n),
        "-f", "image2pipe", "-vcodec", "mjpeg", "-",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=180, check=False)
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return []
    if proc.returncode != 0 or not proc.stdout:
        return []
    buf = proc.stdout
    frames: List[Image.Image] = []
    i = 0
    while True:
        start = buf.find(b"\xff\xd8", i)
        if start < 0: break
        end = buf.find(b"\xff\xd9", start + 2)
        if end < 0: break
        end += 2
        try:
            img = Image.open(io.BytesIO(buf[start:end])).convert("RGB")
            img.load()
            frames.append(img)
        except Exception:
            pass
        i = end
    return frames


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def encode_video(path: str, duration_sec: Optional[float]) -> Optional[np.ndarray]:
    """Return one L2-normalized float32 embedding for the whole video, or None."""
    rt = get_runtime()
    frames = _extract_frames(path, duration_sec)
    if not frames:
        return None
    return _encode_images_pooled(frames, rt)


def encode_text(text: str) -> Optional[np.ndarray]:
    """Return one L2-normalized 512-d embedding for a text query."""
    import torch
    rt = get_runtime()
    inputs = rt.processor(text=[text], return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(rt.device) for k, v in inputs.items()}
    with torch.no_grad():
        emb = rt.model.get_text_features(**inputs)
    emb = emb / emb.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return emb.detach().cpu().numpy()[0].astype(np.float32)


def encode_texts_batch(texts: List[str]) -> np.ndarray:
    """L2-normalized text embeddings for a batch of prompts. Shape (N, dim)."""
    import torch
    rt = get_runtime()
    inputs = rt.processor(text=texts, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(rt.device) for k, v in inputs.items()}
    with torch.no_grad():
        emb = rt.model.get_text_features(**inputs)
    emb = emb / emb.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return emb.detach().cpu().numpy().astype(np.float32)


def _encode_images_pooled(frames: List[Image.Image], rt: ClipRuntime) -> np.ndarray:
    import torch
    with torch.no_grad():
        # Batched forward — a handful of frames fits easily even on CPU.
        inputs = rt.processor(images=frames, return_tensors="pt")
        inputs = {k: v.to(rt.device) for k, v in inputs.items()}
        feats = rt.model.get_image_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        pooled = feats.mean(dim=0)
        pooled = pooled / pooled.norm().clamp(min=1e-8)
    return pooled.detach().cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Similarity utilities
# ---------------------------------------------------------------------------

def cosine_similarity(matrix: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Given (N, D) L2-normalized rows and a normalized (D,) query, return
    (N,) similarities in [-1, 1]. Assumes L2 norm already applied."""
    if matrix.size == 0:
        return np.zeros((0,), dtype=np.float32)
    return matrix @ query.astype(np.float32)


def pack_embedding(arr: np.ndarray) -> bytes:
    """Serialize a 1-D float32 embedding for SQLite storage."""
    return np.asarray(arr, dtype=np.float32).tobytes()


def unpack_embedding(blob: bytes, dim: int = EMBED_DIM) -> Optional[np.ndarray]:
    if not blob:
        return None
    try:
        arr = np.frombuffer(blob, dtype=np.float32)
        if arr.size % dim != 0 and arr.size != dim:
            return None
        return arr[:dim] if arr.size == dim else arr.reshape(-1, dim)[0]
    except Exception:
        return None
