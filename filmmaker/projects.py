"""
Thin CRUD wrapper the API layer uses. Everything else touches Project
directly - this only owns the root directory + list helpers.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from .project import Project, ProjectConfig


def films_root(output_dir: str) -> str:
    root = os.path.join(output_dir, "films")
    os.makedirs(root, exist_ok=True)
    return root


def create(output_dir: str, brief: str, title: Optional[str], config_dict: Optional[Dict[str, Any]]) -> Project:
    cfg = _config_from_dict(config_dict or {})
    return Project.create(films_root(output_dir), brief=brief, title=title, config=cfg)


def load(output_dir: str, project_id: str) -> Project:
    return Project.load(films_root(output_dir), project_id)


def list_all(output_dir: str) -> List[Dict[str, Any]]:
    return Project.list_all(films_root(output_dir))


def _str_or(d: Dict[str, Any], key: str, default: str) -> str:
    v = d.get(key)
    return str(v) if v is not None else default


def _float_or(d: Dict[str, Any], key: str, default: float) -> float:
    v = d.get(key)
    if v is None:
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        raise ValueError(f"{key} must be a number, got {v!r}")


def _config_from_dict(d: Dict[str, Any]) -> ProjectConfig:
    return ProjectConfig(
        per_stage=d.get("per_stage") or {},
        llm_backend=_str_or(d, "llm_backend", "ollama"),
        llm_model=_str_or(d, "llm_model", "llama3.2"),
        llm_host=d.get("llm_host"),
        style=_str_or(d, "style", "stylized"),
        target_minutes=_float_or(d, "target_minutes", 2.0),
        quality=_str_or(d, "quality", "standard").lower(),
        sdxl_variant=_str_or(d, "sdxl_variant", "turbo").lower(),
        motion_backend=_str_or(d, "motion_backend", "auto").lower(),
        voice_backend=_str_or(d, "voice_backend", "piper").lower(),
        music_backend=_str_or(d, "music_backend", "musicgen").lower(),
        upscale_backend=_str_or(d, "upscale_backend", "none").lower(),
        lipsync_backend=_str_or(d, "lipsync_backend", "none").lower(),
    )
