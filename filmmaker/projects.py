"""
Thin CRUD wrapper the API layer uses. Everything else touches Project
directly — this only owns the root directory + list helpers.
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


def _config_from_dict(d: Dict[str, Any]) -> ProjectConfig:
    return ProjectConfig(
        per_stage=d.get("per_stage") or {},
        llm_backend=str(d.get("llm_backend", "ollama")),
        llm_model=str(d.get("llm_model", "llama3.2")),
        llm_host=d.get("llm_host"),
        style=str(d.get("style", "stylized")),
        target_minutes=float(d.get("target_minutes", 2.0)),
        quality=str(d.get("quality", "standard")).lower(),
        sdxl_variant=str(d.get("sdxl_variant", "turbo")).lower(),
    )
