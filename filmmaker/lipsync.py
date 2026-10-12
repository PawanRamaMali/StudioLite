"""
Lip-sync post-processing backends.

The pipeline's final cut (titles + optional upscale) already has dialogue
mixed on top of generated motion. Lip sync is the step that re-paints the
mouth region of each shot to actually match the dialogue - a huge quality
lift when a shot features a named character speaking on camera.

Design:
  - A backend is a small object exposing `check_installed()` and `run()`.
  - Model stacks collide hard with the main `diffusers`/`torch 2.3+` stack
    (MuseTalk pins torch==2.0.1 + mmcv==2.0.1, LatentSync pins its own
    older torch, EchoMimicV3 pins tensorflow==2.15.0), so each backend
    takes its own isolated venv under ``.mp/lipsync-venv-<name>/`` and
    runs through a subprocess shim. Nothing in this module imports the
    heavy deps directly.
  - The default backend is `none`, which produces a skip-artifact so the
    stage is "done" without touching the final cut. Users opt in by
    setting ``config.lipsync_backend`` to one of the known names and
    accepting the one-time venv-bootstrap cost on first run.

This module provides the scaffold: the backend protocol, the `none`
backend, stubbed `musetalk` / `latentsync` / `echomimicv3` entries that
report `available=False` until their shim scripts land in follow-up
commits, and the `run_lipsync(project)` stage entrypoint the registry
calls.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol

from .project import Project

logger = logging.getLogger("studiolite.filmmaker.lipsync")


BackendName = str  # "none" | "musetalk" | "latentsync" | "echomimicv3"


@dataclass
class BackendStatus:
    """Whether a backend can run right now, and if not, why.

    `install_hint` is a short human-readable line the UI can surface
    verbatim ("Run `python -m filmmaker.lipsync install musetalk` to
    bootstrap the isolated venv"). The continuity stage will raise this
    as a warning when a project asked for a backend that isn't available."""

    name: BackendName
    available: bool
    reason: str = ""
    install_hint: str = ""


class LipSyncBackend(Protocol):
    """Minimal surface every backend implements.

    `run` receives the final video path (absolute) and the per-shot
    dialogue manifest the voice_actor stage already wrote. Returns a
    dict suitable for persisting as the stage artifact; must not raise
    on model/weight problems - set `rendered=False` with a `reason` so
    the pipeline still completes and continuity can flag it."""

    name: BackendName

    def check_installed(self, project: Project) -> BackendStatus: ...

    def run(self, project: Project, input_video: str,
            voice_lines: List[Dict[str, Any]],
            output_video: str) -> Dict[str, Any]: ...


# ---------------------------------------------------------------------------
# none - explicit opt-out (the default). Produces a skip artifact so the
# stage reports "done" without touching the final cut.

class NoneBackend:
    name: BackendName = "none"

    def check_installed(self, project: Project) -> BackendStatus:
        return BackendStatus(name="none", available=True)

    def run(self, project: Project, input_video: str,
            voice_lines: List[Dict[str, Any]],
            output_video: str) -> Dict[str, Any]:
        return {
            "backend": "none",
            "rendered": False,
            "skipped": "lipsync_backend=none",
            "input_path": _rel(project, input_video),
            "output_path": _rel(project, input_video),  # passthrough
        }


# ---------------------------------------------------------------------------
# musetalk / latentsync / echomimicv3 - stubs. Each returns
# `available=False` with a clear install hint until the shim scripts land
# in follow-up commits. `run()` returns a skip-artifact that keeps the
# pipeline moving and gives continuity a specific reason to flag.

class _StubBackend:
    """Shared behavior for backends whose shim isn't wired yet."""

    name: BackendName
    reason: str
    install_hint: str

    def check_installed(self, project: Project) -> BackendStatus:
        return BackendStatus(
            name=self.name, available=False,
            reason=self.reason, install_hint=self.install_hint,
        )

    def run(self, project: Project, input_video: str,
            voice_lines: List[Dict[str, Any]],
            output_video: str) -> Dict[str, Any]:
        status = self.check_installed(project)
        return {
            "backend": self.name,
            "rendered": False,
            "skipped": status.reason,
            "install_hint": status.install_hint,
            "input_path": _rel(project, input_video),
            "output_path": _rel(project, input_video),  # passthrough
        }


class MuseTalkBackend(_StubBackend):
    name = "musetalk"
    reason = "MuseTalk 1.5 shim not yet wired"
    install_hint = ("MuseTalk 1.5 needs its own venv (torch 2.0.1 + mmcv 2.0.1 "
                    "+ Python 3.10). Shim will be added in a follow-up commit.")


class LatentSyncBackend(_StubBackend):
    name = "latentsync"
    reason = "LatentSync 1.5 shim not yet wired"
    install_hint = ("LatentSync 1.5 needs its own venv (8 GB VRAM floor). "
                    "Shim will be added in a follow-up commit.")


class EchoMimicV3Backend(_StubBackend):
    name = "echomimicv3"
    reason = "EchoMimicV3 shim not yet wired"
    install_hint = ("EchoMimicV3 can run in-process (Apache-2.0, torch>=2.1.2). "
                    "In-process loader will be added in a follow-up commit.")


# ---------------------------------------------------------------------------
# Registry + stage entrypoint

_BACKENDS: Dict[BackendName, LipSyncBackend] = {
    "none":        NoneBackend(),
    "musetalk":    MuseTalkBackend(),
    "latentsync":  LatentSyncBackend(),
    "echomimicv3": EchoMimicV3Backend(),
}


def get_backend(name: str) -> LipSyncBackend:
    """Resolve a backend name to its instance. Unknown names fall back to
    `none` with a warning logged so a typo in config doesn't halt the
    pipeline; continuity picks up the skip and reports it."""
    key = (name or "none").lower()
    backend = _BACKENDS.get(key)
    if backend is None:
        logger.warning("Unknown lipsync backend %r; falling back to 'none'", name)
        return _BACKENDS["none"]
    return backend


def run_lipsync(project: Project) -> Dict[str, Any]:
    """Stage entrypoint. Resolves the configured backend, points it at the
    best available video (upscaled if present, else titled, else mixed),
    and persists its result.

    Never raises on backend-missing or weight-missing - the whole point
    is to let an optional post-processing stage be optional."""
    backend_name = (project.meta.config.lipsync_backend or "none").lower()
    backend = get_backend(backend_name)

    input_rel = _pick_input_video(project)
    if input_rel is None:
        return {
            "backend": backend_name,
            "rendered": False,
            "skipped": "no final video found (upscale/titles/mixer all missing)",
            "input_path": None,
            "output_path": None,
        }
    input_abs = os.path.join(project.dir, input_rel.replace("/", os.sep))

    # Load per-shot dialogue manifest from the voice_actor stage.
    voice_actor = project.read_artifact("voice_actor") or {}
    voice_lines: List[Dict[str, Any]] = list(voice_actor.get("lines") or [])

    # Output lands next to the input with a `_lipsync` suffix.
    stem, ext = os.path.splitext(input_abs)
    output_abs = f"{stem}_lipsync{ext or '.mp4'}"

    # Early exit for explicit opt-out and un-installed backends - no need
    # to prepare arguments or clean up on failure.
    if backend.name == "none":
        return backend.run(project, input_abs, voice_lines, output_abs)
    status = backend.check_installed(project)
    if not status.available:
        logger.info("Lipsync backend %s unavailable: %s", backend.name, status.reason)
        return backend.run(project, input_abs, voice_lines, output_abs)

    try:
        result = backend.run(project, input_abs, voice_lines, output_abs)
    except Exception as e:  # defensive - backends promise not to raise
        logger.exception("Lipsync backend %s raised unexpectedly", backend.name)
        return {
            "backend": backend.name,
            "rendered": False,
            "skipped": f"backend raised: {e}",
            "input_path": _rel(project, input_abs),
            "output_path": _rel(project, input_abs),
        }
    return result


# ---------------------------------------------------------------------------
# helpers

# Order matches the stage order: last writer wins. Upscale lands last on
# the happy path, titles when upscale is "none", mixer when titles failed.
_INPUT_CANDIDATE_STAGES = ("upscale", "titles", "mixer")


def _pick_input_video(project: Project) -> Optional[str]:
    for key in _INPUT_CANDIDATE_STAGES:
        art = project.read_artifact(key) or {}
        rel = art.get("output_path")
        if not rel:
            continue
        abs_path = os.path.join(project.dir, str(rel).replace("/", os.sep))
        if os.path.exists(abs_path):
            return str(rel)
    # Final fallback: a bare `final.mp4` at the project root.
    for candidate in ("final.mp4", "silent_cut.mp4"):
        abs_path = os.path.join(project.dir, candidate)
        if os.path.exists(abs_path):
            return candidate
    return None


def _rel(project: Project, abs_path: str) -> str:
    try:
        return os.path.relpath(abs_path, project.dir).replace(os.sep, "/")
    except ValueError:
        return abs_path
