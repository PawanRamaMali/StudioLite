"""Health / liveness probe.

Tiny on purpose — this is the endpoint a Docker HEALTHCHECK or a load
balancer hits. No dependencies, no side effects, no auth. Kept separate
from the ``/system/status`` endpoint (which does heavy work: probes GPU,
counts jobs) so a monitoring hit can't ever accidentally spin up a
CUDA context."""
from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(prefix="/api/v1", tags=["health"])


@router.get("/health")
async def health() -> dict:
    return {"status": "ok"}
