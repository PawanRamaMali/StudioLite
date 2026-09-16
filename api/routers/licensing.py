"""License status / install / deactivate endpoints.

Backed by ``filmmaker.licensing`` — this router is only the HTTP shell.
Extracted from ``api_server.py`` so the monolith shrinks and the
licensing surface is easy to find (and swap for a real activation
server later without touching unrelated code)."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from filmmaker import licensing

router = APIRouter(prefix="/api/v1/system", tags=["licensing"])


class LicenseInstallRequest(BaseModel):
    # Two-piece bundle: JSON payload the publisher signed + Ed25519 sig.
    payload: dict
    signature: str


@router.get("/license")
async def license_status() -> dict:
    """Report the current entitlement — tier, feature list, expiry,
    grace warning if any. Absence of a license reports tier=free and
    valid=False; the UI treats that as 'community edition' rather than
    an error."""
    check = licensing.check_entitlement()
    return {
        "valid": check.valid,
        "tier": check.tier,
        "features": check.features,
        "licensee": check.licensee,
        "expires_at": check.expires_at,
        "reason": check.reason,
        "warning": check.warning,
        "device_fingerprint": licensing.device_fingerprint(),
    }


@router.post("/license")
async def install_license(body: LicenseInstallRequest) -> dict:
    """Verify a caller-supplied license bundle and, if it checks out,
    persist it under ``.license``. Invalid bundles are rejected without
    touching disk so a botched install can't lock the user out."""
    check = licensing.install_license(body.model_dump())
    if not check.valid:
        raise HTTPException(status_code=400, detail=check.reason)
    return {
        "valid": True,
        "tier": check.tier,
        "features": check.features,
        "licensee": check.licensee,
        "expires_at": check.expires_at,
        "warning": check.warning,
    }


@router.delete("/license")
async def deactivate_license() -> dict:
    """Remove the on-disk license. Idempotent; missing file is fine."""
    licensing.deactivate_license()
    return {"deactivated": True}
