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
