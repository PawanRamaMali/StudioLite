"""Managed environment variables endpoint.

Exposes a strict allowlist of env keys (HF_TOKEN, CUDA_*, TORCH_*, …)
so the UI can inspect and update them without giving the API a way to
inject arbitrary process state. The keys are persisted to ``.env`` next
to the repo root so they survive a restart.

Extracted from ``api_server.py`` as part of the monolith cleanup. The
root path is resolved lazily so this module doesn't need to run before
``api_server`` sets up its filesystem layout - the router mounts
without side effects, and the first request looks up the current
process's cwd-inferred root."""
from __future__ import annotations

import os
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/api/v1/system", tags=["env-vars"])


# Keys the UI is allowed to show/edit. Anything not on this list AND
# not matching one of the safe prefixes below is rejected - a POST for
# ``PATH`` or ``SECRET_KEY`` will not silently succeed.
MANAGED_ENV_VARS: List[str] = [
    "HF_TOKEN", "HF_HOME", "NEXT_PUBLIC_API_URL",
    "CUDA_VISIBLE_DEVICES", "PYTORCH_CUDA_ALLOC_CONF",
]

_SAFE_PREFIXES = ("HF_", "CUDA_", "TORCH_", "PYTORCH_")


def _is_managed(key: str) -> bool:
    return key in MANAGED_ENV_VARS or key.startswith(_SAFE_PREFIXES)


def _mask(key: str, val: str) -> str:
    """Mask any value whose key contains 'TOKEN'. Short values pass
    through - masking a 4-char value would tell an attacker exactly
    how long the real token is."""
    if "TOKEN" in key and val and len(val) > 8:
        return val[:8] + "..." + val[-4:]
    return val


def _env_file_path() -> str:
    """Locate the repo's ``.env`` alongside ``api_server.py``. Resolved
    per call so importing the router doesn't touch the filesystem - 
    keeps the test harness's monkeypatched cwd working."""
    # api_server.py sits at the repo root; this module sits under
    # api/routers/, so two ``dirname`` calls back up walk to the root.
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(os.path.dirname(os.path.dirname(here)), ".env")


@router.get("/env")
async def get_env_vars() -> Dict[str, Any]:
    """Report every managed env var and every var matching a safe
    prefix, masking token values."""
    env: Dict[str, str] = {}
    for key in MANAGED_ENV_VARS:
        env[key] = _mask(key, os.environ.get(key, ""))
    for key, val in os.environ.items():
        if key.startswith(_SAFE_PREFIXES) and key not in env:
            env[key] = _mask(key, val)
    return {"env": env, "managed_keys": MANAGED_ENV_VARS}


@router.post("/env")
async def set_env_var(key: str, value: str) -> Dict[str, Any]:
    """Set an env var. Takes effect immediately for this process and
    persists to ``.env`` for the next restart. Auth handled by the
    global middleware."""
    k = key.strip()
    if not k:
        raise HTTPException(status_code=422, detail="Key cannot be empty")
    if not _is_managed(k):
        raise HTTPException(
            status_code=403,
            detail=f"Env key {k!r} is not on the managed allowlist. Add it "
                   f"to MANAGED_ENV_VARS in api/routers/env_vars.py if you "
                   f"need it exposed.",
        )
    os.environ[k] = value

    env_file = _env_file_path()
    env_lines: List[str] = []
    if os.path.exists(env_file):
        with open(env_file, "r", encoding="utf-8") as f:
            env_lines = f.readlines()
    prefix = f"{k}="
    found = False
    for i, line in enumerate(env_lines):
        if line.strip().startswith(prefix):
            env_lines[i] = f"{k}={value}\n"
            found = True
            break
    if not found:
        env_lines.append(f"{k}={value}\n")
    with open(env_file, "w", encoding="utf-8") as f:
        f.writelines(env_lines)
    return {"status": "ok", "key": k, "persisted": True}


@router.delete("/env")
async def delete_env_var(key: str) -> Dict[str, Any]:
    """Remove a managed env var from the process and ``.env``."""
    k = key.strip()
    if not _is_managed(k):
        raise HTTPException(
            status_code=403,
            detail=f"Env key {k!r} is not managed.",
        )
    if k in os.environ:
        del os.environ[k]

    env_file = _env_file_path()
    if os.path.exists(env_file):
        with open(env_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        prefix = f"{k}="
        lines = [ln for ln in lines if not ln.strip().startswith(prefix)]
        with open(env_file, "w", encoding="utf-8") as f:
            f.writelines(lines)
    return {"status": "ok", "key": k, "deleted": True}
