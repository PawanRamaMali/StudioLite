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


# Fields a running project is allowed to swap after creation. Narrower than
# the full ProjectConfig surface: anything that would re-plan the whole film
# (per_stage overrides, custom knobs) stays off this list on purpose.
_PATCHABLE_FIELDS = frozenset({
    "lipsync_backend", "upscale_backend", "voice_backend",
    "music_backend", "motion_backend",
    "quality", "sdxl_variant", "style",
    "llm_backend", "llm_model", "llm_host",
    "target_minutes",
})

# Enum-ish fields the create path lowercases - we keep that invariant here
# so a case-mangled patch round-trips to the same canonical value.
_LOWERED_FIELDS = frozenset({
    "quality", "sdxl_variant", "motion_backend", "voice_backend",
    "music_backend", "upscale_backend", "lipsync_backend",
})

# Which stage a config swap directly invalidates. The caller still runs
# mark_downstream_stale on each one, so e.g. a lipsync_backend change
# cascades to continuity on its own.
_FIELD_AFFECTS_STAGE = {
    "lipsync_backend": "lipsync",
    "upscale_backend": "upscale",
    "voice_backend": "voice_actor",
    "music_backend": "composer",
    "motion_backend": "motion_shots",
    "sdxl_variant": "shots",
    "quality": "shots",
}


def apply_config_patch(project, patch: Dict[str, Any]) -> List[str]:
    """Merge a partial config dict into the project's meta and persist it.

    Returns the stage keys whose output no longer matches the new config,
    so the caller can mark them pending and cascade downstream. Raises
    ValueError on an unknown field or a bad value - the API layer maps
    that to 422.
    """
    if not isinstance(patch, dict):
        raise ValueError("patch must be an object")
    unknown = set(patch) - _PATCHABLE_FIELDS
    if unknown:
        raise ValueError(f"unknown config fields: {sorted(unknown)}")

    meta = project.meta
    cfg = meta.config
    affected: List[str] = []

    for key, raw in patch.items():
        if key == "target_minutes":
            if raw is None:
                continue
            try:
                new_val: Any = float(raw)
            except (TypeError, ValueError):
                raise ValueError(f"target_minutes must be a number, got {raw!r}")
        elif key == "llm_host":
            new_val = str(raw).strip() or None if raw else None
        elif raw is None:
            continue
        else:
            text = str(raw)
            new_val = text.lower() if key in _LOWERED_FIELDS else text

        if getattr(cfg, key) != new_val:
            setattr(cfg, key, new_val)
            stage = _FIELD_AFFECTS_STAGE.get(key)
            if stage and stage not in affected:
                affected.append(stage)

    project.save_meta(meta)
    return affected
