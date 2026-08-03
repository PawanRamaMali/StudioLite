"""
Multi-Model Hub for StudioLite.

Plugin system for supporting multiple video generation engines.
Provides a standardized interface and model registry.
"""

import os
import json
from typing import Optional
from pathlib import Path

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


# =============================================
# Model Registry
# =============================================

MODEL_REGISTRY = {
    # Built-in engines
    "wan_1.3b": {
        "name": "Wan 2.1 (1.3B)",
        "engine": "wan",
        "model": "1.3b",
        "type": "text2video",
        "vram_min": 8,
        "vram_recommended": 12,
        "speed": "medium",
        "quality": "good",
        "description": "Fast text-to-video, good quality for 8-12GB GPUs",
        "hf_id": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        "modes": ["text2video"],
        "max_frames": 81,
        "built_in": True,
    },
    "wan_14b": {
        "name": "Wan 2.1 (14B)",
        "engine": "wan",
        "model": "14b",
        "type": "text2video",
        "vram_min": 16,
        "vram_recommended": 24,
        "speed": "slow",
        "quality": "excellent",
        "description": "High-quality text and image-to-video, needs 16GB+ VRAM",
        "hf_id": "Wan-AI/Wan2.1-T2V-14B-Diffusers",
        "modes": ["text2video", "image2video"],
        "max_frames": 121,
        "built_in": True,
    },
    "wan_22_14b": {
        "name": "Wan 2.2 (14B)",
        "engine": "wan",
        "model": "2.2-14b",
        "type": "text2video",
        "vram_min": 16,
        "vram_recommended": 24,
        "speed": "slow",
        "quality": "excellent",
        "description": "Latest Wan model with improvements over 2.1",
        "hf_id": "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        "modes": ["text2video", "image2video", "video2video"],
        "max_frames": 121,
        "built_in": True,
    },
    "ltx_base": {
        "name": "LTX-Video (Base)",
        "engine": "ltx",
        "model": "base",
        "type": "text2video",
        "vram_min": 8,
        "vram_recommended": 12,
        "speed": "fast",
        "quality": "good",
        "description": "Fastest generation, good quality",
        "hf_id": "Lightricks/LTX-Video",
        "modes": ["text2video", "image2video"],
        "max_frames": 161,
        "built_in": True,
    },
    "cogvideox_2b": {
        "name": "CogVideoX (2B)",
        "engine": "cogvideox",
        "model": "2b",
        "type": "text2video",
        "vram_min": 8,
        "vram_recommended": 10,
        "speed": "fast",
        "quality": "good",
        "description": "Versatile, works well on 8GB GPUs",
        "hf_id": "THUDM/CogVideoX-2b",
        "modes": ["text2video", "video2video"],
        "max_frames": 49,
        "built_in": True,
    },
    "cogvideox_5b": {
        "name": "CogVideoX (5B)",
        "engine": "cogvideox",
        "model": "5b",
        "type": "text2video",
        "vram_min": 16,
        "vram_recommended": 18,
        "speed": "medium",
        "quality": "very good",
        "description": "Higher quality, needs 16GB+ VRAM",
        "hf_id": "THUDM/CogVideoX-5b",
        "modes": ["text2video", "video2video"],
        "max_frames": 81,
        "built_in": True,
    },
    "hunyuan_1.0": {
        "name": "HunyuanVideo (1.0)",
        "engine": "hunyuan",
        "model": "1.0",
        "type": "text2video",
        "vram_min": 24,
        "vram_recommended": 32,
        "speed": "slow",
        "quality": "excellent",
        "description": "Up to 1080p, needs 24GB+ VRAM",
        "hf_id": "tencent/HunyuanVideo",
        "modes": ["text2video"],
        "max_frames": 129,
        "built_in": True,
    },
    # Community / Third-party models (installable)
    "animatediff": {
        "name": "AnimateDiff",
        "engine": "animatediff",
        "model": "v3",
        "type": "text2video",
        "vram_min": 8,
        "vram_recommended": 12,
        "speed": "fast",
        "quality": "good",
        "description": "Animation-focused T2V/I2V, 8GB VRAM",
        "hf_id": "guoyww/animatediff-motion-adapter-v1-5-3",
        "modes": ["text2video", "image2video"],
        "max_frames": 32,
        "built_in": False,
        "install_note": "pip install diffusers[animatediff]",
    },
    "svd": {
        "name": "Stable Video Diffusion",
        "engine": "svd",
        "model": "xt",
        "type": "image2video",
        "vram_min": 8,
        "vram_recommended": 16,
        "speed": "medium",
        "quality": "very good",
        "description": "Strong Image-to-Video from Stability AI",
        "hf_id": "stabilityai/stable-video-diffusion-img2vid-xt",
        "modes": ["image2video"],
        "max_frames": 25,
        "built_in": False,
        "install_note": "Auto-downloads from HuggingFace on first use",
    },
    "mochi_1": {
        "name": "Mochi 1",
        "engine": "mochi",
        "model": "preview",
        "type": "text2video",
        "vram_min": 12,
        "vram_recommended": 24,
        "speed": "medium",
        "quality": "very good",
        "description": "High quality T2V from Genmo",
        "hf_id": "genmo/mochi-1-preview",
        "modes": ["text2video"],
        "max_frames": 84,
        "built_in": False,
        "install_note": "pip install mochi-preview",
    },
}


def get_all_models() -> dict:
    """Get the complete model registry."""
    return dict(MODEL_REGISTRY)


def get_built_in_models() -> dict:
    """Get only built-in models."""
    return {k: v for k, v in MODEL_REGISTRY.items() if v.get("built_in")}


def get_installable_models() -> dict:
    """Get third-party installable models."""
    return {k: v for k, v in MODEL_REGISTRY.items() if not v.get("built_in")}


def get_models_for_vram(vram_gb: float) -> dict:
    """Get models that fit within the given VRAM."""
    return {k: v for k, v in MODEL_REGISTRY.items() if v["vram_min"] <= vram_gb}


def get_models_by_mode(mode: str) -> dict:
    """Get models that support a specific mode (text2video, image2video, video2video)."""
    return {k: v for k, v in MODEL_REGISTRY.items() if mode in v.get("modes", [])}


def check_model_installed(model_key: str) -> bool:
    """Check if a model's weights are already downloaded."""
    model = MODEL_REGISTRY.get(model_key)
    if not model:
        return False

    hf_id = model.get("hf_id", "")
    if not hf_id:
        return False

    # Check HuggingFace cache
    try:
        from huggingface_hub import scan_cache_dir
        cache_info = scan_cache_dir()
        for repo in cache_info.repos:
            if repo.repo_id == hf_id:
                return True
    except Exception:
        pass

    # Check custom model directory
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    model_dir = os.path.join(hf_home, "hub", f"models--{hf_id.replace('/', '--')}")
    return os.path.exists(model_dir)


def get_model_status() -> list:
    """Get installation status of all models."""
    results = []
    for key, model in MODEL_REGISTRY.items():
        installed = check_model_installed(key)
        results.append({
            "key": key,
            "name": model["name"],
            "installed": installed,
            "vram_min": model["vram_min"],
            "engine": model["engine"],
            "built_in": model.get("built_in", False),
            "modes": model.get("modes", []),
            "quality": model.get("quality", ""),
            "speed": model.get("speed", ""),
        })
    return results


def _cache_size_bytes(hf_id: str) -> int:
    """Sum size of all files under the HF cache for this repo. Approximate."""
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    model_dir = os.path.join(hf_home, "hub", f"models--{hf_id.replace('/', '--')}")
    if not os.path.isdir(model_dir):
        return 0
    total = 0
    for root, _dirs, files in os.walk(model_dir):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


# Cache HfApi.model_info results in-process. Keys are hf_id strings, values are
# the summed byte count. Populated by estimate_download_size, read by the
# endpoint (for disk precheck) and by download_model (for real % progress).
_SIZE_CACHE: dict = {}


def estimate_download_size(model_key: str) -> int:
    """Return the total on-disk bytes a full download of this model will need.

    Hits HfApi.model_info(files_metadata=True) once per key (results cached in
    process) and sums the sibling file sizes. Returns 0 if the model has no
    hf_id, if huggingface_hub isn't importable, or if the metadata call fails
    (network, gated repo, rate limit) — callers should treat 0 as "unknown"
    and fall back to size-agnostic behavior rather than blocking on it.
    """
    model = MODEL_REGISTRY.get(model_key)
    if not model:
        return 0
    hf_id = model.get("hf_id", "")
    if not hf_id:
        return 0
    if hf_id in _SIZE_CACHE:
        return _SIZE_CACHE[hf_id]
    try:
        from huggingface_hub import HfApi
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        info = HfApi().model_info(hf_id, files_metadata=True, token=token)
        total = 0
        for sib in getattr(info, "siblings", []) or []:
            sz = getattr(sib, "size", None)
            if isinstance(sz, int) and sz > 0:
                total += sz
    except Exception:
        total = 0
    _SIZE_CACHE[hf_id] = total
    return total


def download_model(model_key: str, progress_callback=None) -> dict:
    """Download a model's weights via huggingface_hub.snapshot_download.

    Progress callback signature: ``cb(pct: int, message: str)``. When we know
    the total size (see estimate_download_size), the watcher thread emits a
    real percentage capped at 99 during transfer; the final 100 fires after
    snapshot_download returns. If size is unknown, falls back to a bytes-
    downloaded counter.

    Returns ``{"key", "hf_id", "path", "size_bytes"}`` on success.
    Raises RuntimeError with a human message on any failure.
    """
    model = MODEL_REGISTRY.get(model_key)
    if not model:
        raise RuntimeError(f"Unknown model: {model_key}")
    hf_id = model.get("hf_id", "")
    if not hf_id:
        raise RuntimeError(f"Model {model_key} has no hf_id — can't download.")
    if check_model_installed(model_key):
        return {
            "key": model_key,
            "hf_id": hf_id,
            "path": None,
            "size_bytes": _cache_size_bytes(hf_id),
            "already_installed": True,
        }

    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise RuntimeError(f"huggingface_hub not installed: {e}") from e

    total_bytes = estimate_download_size(model_key)  # 0 if unknown

    # HF snapshot_download blocks. To emit progress we start a watcher thread
    # that polls the on-disk cache size while the download runs.
    import threading
    stop_flag = threading.Event()

    def _watcher():
        last = -1
        while not stop_flag.wait(2.0):
            size = _cache_size_bytes(hf_id)
            if size == last:
                continue
            last = size
            if not progress_callback:
                continue
            if total_bytes > 0:
                # Cap at 99 so the final 100 comes from the post-download call.
                pct = min(99, int(size / total_bytes * 100))
                gb_done = size / (1024 * 1024 * 1024)
                gb_total = total_bytes / (1024 * 1024 * 1024)
                progress_callback(pct, f"Downloading… {gb_done:.2f} / {gb_total:.2f} GB")
            else:
                mb = size / (1024 * 1024)
                progress_callback(-1, f"Downloading… {mb:,.0f} MB so far")

    if progress_callback:
        if total_bytes > 0:
            gb = total_bytes / (1024 * 1024 * 1024)
            progress_callback(0, f"Starting download of {hf_id} ({gb:.2f} GB)…")
        else:
            progress_callback(0, f"Starting download of {hf_id}…")

    watcher = threading.Thread(target=_watcher, daemon=True)
    watcher.start()

    try:
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        path = snapshot_download(
            repo_id=hf_id,
            token=token,
            resume_download=True,
        )
    except Exception as e:
        stop_flag.set()
        raise RuntimeError(f"Download failed: {e}") from e
    finally:
        stop_flag.set()

    size = _cache_size_bytes(hf_id)
    if progress_callback:
        progress_callback(100, f"Downloaded {size / (1024*1024*1024):.2f} GB")

    return {
        "key": model_key,
        "hf_id": hf_id,
        "path": path,
        "size_bytes": size,
        "already_installed": False,
    }


def delete_model(model_key: str) -> dict:
    """Remove a model's weights from the HF cache. Uses the official
    scan_cache_dir + delete_revisions() API so shared blobs aren't nuked.
    Returns ``{"key", "hf_id", "freed_bytes"}``.
    """
    model = MODEL_REGISTRY.get(model_key)
    if not model:
        raise RuntimeError(f"Unknown model: {model_key}")
    hf_id = model.get("hf_id", "")
    if not hf_id:
        raise RuntimeError(f"Model {model_key} has no hf_id — can't delete.")

    try:
        from huggingface_hub import scan_cache_dir
    except ImportError as e:
        raise RuntimeError(f"huggingface_hub not installed: {e}") from e

    cache_info = scan_cache_dir()
    freed = 0
    revisions = []
    for repo in cache_info.repos:
        if repo.repo_id == hf_id:
            for rev in repo.revisions:
                revisions.append(rev.commit_hash)
                freed += rev.size_on_disk

    if not revisions:
        return {"key": model_key, "hf_id": hf_id, "freed_bytes": 0, "found": False}

    strategy = cache_info.delete_revisions(*revisions)
    strategy.execute()
    return {"key": model_key, "hf_id": hf_id, "freed_bytes": freed, "found": True}


def get_recommended_model(vram_gb: float, mode: str = "text2video") -> Optional[str]:
    """Get the best recommended model for the user's VRAM and use case."""
    candidates = get_models_for_vram(vram_gb)
    candidates = {k: v for k, v in candidates.items() if mode in v.get("modes", []) and v.get("built_in")}

    if not candidates:
        return None

    # Sort by quality rating
    quality_order = {"excellent": 4, "very good": 3, "good": 2, "fair": 1}
    sorted_models = sorted(
        candidates.items(),
        key=lambda x: quality_order.get(x[1].get("quality", ""), 0),
        reverse=True,
    )

    return sorted_models[0][0] if sorted_models else None
