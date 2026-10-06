"""Read-only view of the model registry.

Reports which models are installed on disk plus any in-flight download
job that's targeting each one. Split off from ``api_server.py`` because
this is the piece a fresh UI hits first (Home / Settings) and it needs
to be fast and dependency-light.

Job state lives in ``api_server.jobs`` - imported lazily inside the
handler so this module can be imported before ``api_server`` finishes
initializing (fixes the router-mount order problem the extraction
otherwise creates).

Also exposes a wider "registry" view (``/registry``) that lists every
weight StudioLite touches - CLIP, Whisper, YuNet/SFace, SDXL, Wan,
MusicGen, AudioLDM, Real-ESRGAN, Piper, … - not just the video engines
``model_hub`` tracks. The registry is defined in ``models.registry``
and probed with a pure filesystem check; this handler just serializes
the result."""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException

logger = logging.getLogger("studiolite.api.models")

router = APIRouter(prefix="/api/v1/models", tags=["models"])


@router.get("/inventory")
async def model_inventory() -> Dict[str, Any]:
    """Live installation status of every model in the registry.

    Returns per-model: {key, name, installed, vram_min, engine, modes,
    quality, speed, built_in, active_job}. ``active_job`` is the job id
    of an in-flight download for that model, if any.
    """
    try:
        import model_hub as _mh
        models: List[Dict[str, Any]] = _mh.get_model_status()
    except Exception as e:  # noqa: BLE001
        logger.error("model_hub.get_model_status failed: %s", e)
        models = []

    # Lazy import to avoid a router → api_server → router circular import
    # at module load time; api_server's jobs store is set up before any
    # request lands here.
    from api_server import jobs, _jobs_lock

    with _jobs_lock:
        active = {
            j["params"].get("model_key"): jid
            for jid, j in jobs.items()
            if j.get("kind") == "model_download"
            and j["status"] in ("queued", "running")
            and j.get("params", {}).get("model_key")
        }

    for m in models:
        m["active_job"] = active.get(m["key"])

    installed = sum(1 for m in models if m.get("installed"))
    return {
        "models": models,
        "installed_count": installed,
        "total_count": len(models),
    }


# ---------------------------------------------------------------------------
# Broader cross-backend registry (CLIP, Whisper, YuNet, SDXL, Wan, …)
# ---------------------------------------------------------------------------


@router.get("/registry")
async def registry_inventory() -> Dict[str, Any]:
    """Unified inventory of every weight StudioLite can load.

    This is a disk-check only - never triggers a download, never hits the
    network. Each row carries the expected absolute path so the user can
    see exactly where to drop the file. Grouping (text / image / video /
    audio / face / other) happens client-side off the `kind` field.
    """
    try:
        from models.registry import probe_registry, registry_summary
    except Exception as e:  # noqa: BLE001
        logger.error("models.registry import failed: %s", e)
        raise HTTPException(status_code=500, detail=f"registry unavailable: {e}") from e

    rows = probe_registry()
    summary = registry_summary()
    return {
        "models": rows,
        "summary": summary,
    }


@router.post("/{model_id}/open-folder")
async def model_open_folder(model_id: str) -> Dict[str, Any]:
    """Return the parent directory for a model's expected path.

    The server does NOT shell out to a native file explorer - the UI
    simply shows the path, and the user opens it themselves. Returns 404
    when `model_id` isn't registered.
    """
    try:
        from models.registry import REGISTRY, expected_path_for
    except Exception as e:  # noqa: BLE001
        logger.error("models.registry import failed: %s", e)
        raise HTTPException(status_code=500, detail=f"registry unavailable: {e}") from e

    spec = REGISTRY.get(model_id)
    if spec is None:
        raise HTTPException(status_code=404, detail=f"unknown model: {model_id}")

    expected = expected_path_for(spec)
    # If the path points at a file, strip to its directory; if a dir, keep it.
    if os.path.isdir(expected):
        folder = expected
    else:
        folder = os.path.dirname(expected) or expected
    return {
        "model_id": model_id,
        "expected_path": expected,
        "folder": folder,
        "exists": os.path.isdir(folder),
    }
