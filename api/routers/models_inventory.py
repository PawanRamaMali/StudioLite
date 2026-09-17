"""Read-only view of the model registry.

Reports which models are installed on disk plus any in-flight download
job that's targeting each one. Split off from ``api_server.py`` because
this is the piece a fresh UI hits first (Home / Settings) and it needs
to be fast and dependency-light.

Job state lives in ``api_server.jobs`` — imported lazily inside the
handler so this module can be imported before ``api_server`` finishes
initializing (fixes the router-mount order problem the extraction
otherwise creates)."""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from fastapi import APIRouter

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
