"""Opt-in local telemetry endpoints.

Backed by ``filmmaker.telemetry`` — this router is the HTTP shell. All
data is local; nothing gets shipped anywhere without the user zipping
the diagnostic bundle themselves."""
from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import Response
from pydantic import BaseModel

from filmmaker import telemetry

router = APIRouter(prefix="/api/v1/system/telemetry", tags=["telemetry"])


class TelemetryConsentRequest(BaseModel):
    consent: bool


@router.get("")
async def telemetry_state() -> dict:
    """Current consent + installation ID. Never returns event bodies."""
    return telemetry.get_state()


@router.post("/consent")
async def telemetry_set_consent(body: TelemetryConsentRequest) -> dict:
    """Flip local telemetry on or off. Auth via global middleware."""
    return telemetry.record_consent(body.consent)


@router.post("/reset-id")
async def telemetry_reset_id() -> dict:
    """Rotate the installation ID. Only useful when consent is on."""
    return telemetry.reset_installation_id()


@router.get("/events")
async def telemetry_events(limit: int = 100) -> dict:
    """Show the tail of the local events log so the user can see exactly
    what's being recorded before they consent to share a diagnostic
    bundle."""
    limit = max(1, min(1000, int(limit)))
    return {"events": telemetry.recent_events(limit=limit)}


@router.get("/bundle")
async def telemetry_bundle() -> Response:
    """Return a diagnostic zip the user can attach to a bug report. Never
    sent anywhere by us — this endpoint just packages what's already on
    disk with the same redaction the module documents."""
    data = telemetry.build_diagnostic_bundle()
    return Response(
        content=data,
        media_type="application/zip",
        headers={
            "Content-Disposition":
                "attachment; filename=studiolite-diagnostics.zip",
        },
    )
