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

import json
import logging
import os
import shutil
import subprocess
import tempfile
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


class MuseTalkBackend:
    """MuseTalk 1.5 via an isolated venv subprocess.

    MuseTalk hard-pins torch==2.0.1 + mmcv==2.0.1 on Python 3.10 / CUDA
    11.8, which collides with StudioLite's main torch 2.3+ / Python 3.11
    stack. We can't install it alongside; we shell out to a separate
    venv the user set up once.

    Setup is a one-time user action, not something StudioLite does on
    its own (a bad `pip install` sequence can brick a user's torch
    install, and MuseTalk's weights are ~2 GB). The user:
      1. Clones TMElyralab/MuseTalk under any path they like.
      2. Creates a Python 3.10 venv, pip-installs MuseTalk's pinned
         requirements, downloads weights per MuseTalk's README.
      3. Writes `.mp/lipsync-musetalk/config.json` under the StudioLite
         output dir pointing at that clone + venv + weights.

    `check_installed()` reads that config and verifies every path still
    exists. `run()` writes a per-shot MuseTalk inference-config YAML,
    calls MuseTalk's `scripts.inference` module with the venv's python,
    and moves the result into place."""

    name: BackendName = "musetalk"
    _CONFIG_SUBDIR = "lipsync-musetalk"
    _CONFIG_NAME = "config.json"

    def _config_path(self, project: Project) -> str:
        """Where we expect the user to drop their MuseTalk config.

        Resolved relative to the project's films-root parent so the file
        is shared across all films, not duplicated per-film."""
        root = getattr(project, "_root_dir", None) or os.path.dirname(project.dir)
        # films_root() nests `films/`; the config lives one level above
        # so it's siblings-with-models, not shared with film projects.
        base = os.path.dirname(root) if os.path.basename(root) == "films" else root
        return os.path.join(base, self._CONFIG_SUBDIR, self._CONFIG_NAME)

    def _load_config(self, project: Project) -> Optional[Dict[str, Any]]:
        path = self._config_path(project)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning("MuseTalk config at %s unreadable: %s", path, e)
            return None

    def check_installed(self, project: Project) -> BackendStatus:
        cfg_path = self._config_path(project)
        cfg = self._load_config(project)
        if cfg is None:
            return BackendStatus(
                name=self.name, available=False,
                reason=f"no config at {cfg_path}",
                install_hint=(
                    "Clone TMElyralab/MuseTalk, make a Python 3.10 venv, pip-install its "
                    "pinned requirements (torch==2.0.1, mmcv==2.0.1), and drop a "
                    f"config.json at {cfg_path} with keys: repo_dir, venv_python, "
                    "unet_model_path, unet_config, version (\"v15\"), ffmpeg_path."
                ),
            )
        required = ("repo_dir", "venv_python", "unet_model_path", "unet_config")
        missing_keys = [k for k in required if not cfg.get(k)]
        if missing_keys:
            return BackendStatus(
                name=self.name, available=False,
                reason=f"config missing keys: {', '.join(missing_keys)}",
                install_hint=f"Add the missing keys to {cfg_path}.",
            )
        repo_dir = cfg["repo_dir"]
        venv_py = cfg["venv_python"]
        if not os.path.isdir(repo_dir):
            return BackendStatus(name=self.name, available=False,
                                 reason=f"repo_dir does not exist: {repo_dir}",
                                 install_hint="Fix repo_dir in the config.")
        if not os.path.exists(venv_py):
            return BackendStatus(name=self.name, available=False,
                                 reason=f"venv_python does not exist: {venv_py}",
                                 install_hint="Fix venv_python in the config.")
        return BackendStatus(name=self.name, available=True)

    def run(self, project: Project, input_video: str,
            voice_lines: List[Dict[str, Any]],
            output_video: str) -> Dict[str, Any]:
        status = self.check_installed(project)
        if not status.available:
            return _passthrough(self.name, project, input_video, status)

        cfg = self._load_config(project) or {}
        audio_rel = self._collect_speech_track(project, voice_lines)
        if audio_rel is None:
            return _passthrough(
                self.name, project, input_video,
                BackendStatus(name=self.name, available=False,
                              reason="no dialogue wavs to drive lip sync"),
            )
        audio_abs = os.path.join(project.dir, audio_rel.replace("/", os.sep))

        # Write MuseTalk's inference config.
        work_dir = os.path.join(project.dir, "artifacts", "lipsync-musetalk")
        os.makedirs(work_dir, exist_ok=True)
        task_yaml = os.path.join(work_dir, "inference.yaml")
        bbox_shift = int(cfg.get("bbox_shift", 0))
        with open(task_yaml, "w", encoding="utf-8") as f:
            f.write("task_0:\n"
                    f"  video_path: {_posix(input_video)}\n"
                    f"  audio_path: {_posix(audio_abs)}\n"
                    f"  bbox_shift: {bbox_shift}\n")

        result_dir = os.path.join(work_dir, "results")
        os.makedirs(result_dir, exist_ok=True)
        cmd = [
            cfg["venv_python"], "-m", "scripts.inference",
            "--inference_config", task_yaml,
            "--result_dir", result_dir,
            "--unet_model_path", cfg["unet_model_path"],
            "--unet_config", cfg["unet_config"],
            "--version", cfg.get("version", "v15"),
        ]
        if cfg.get("ffmpeg_path"):
            cmd += ["--ffmpeg_path", cfg["ffmpeg_path"]]
        extra = cfg.get("extra_args") or []
        if isinstance(extra, list):
            cmd += [str(x) for x in extra]

        logger.info("MuseTalk inference: cd %s && %s", cfg["repo_dir"], " ".join(cmd))
        try:
            proc = subprocess.run(cmd, cwd=cfg["repo_dir"], capture_output=True,
                                   text=True, check=False,
                                   timeout=int(cfg.get("timeout_sec", 1800)))
        except subprocess.TimeoutExpired as e:
            return _passthrough(self.name, project, input_video,
                                BackendStatus(name=self.name, available=False,
                                              reason=f"MuseTalk timed out: {e}"))
        if proc.returncode != 0:
            # Keep stderr tail so the UI can show why the shim failed.
            tail = (proc.stderr or "").strip().splitlines()[-6:]
            return _passthrough(self.name, project, input_video,
                                BackendStatus(name=self.name, available=False,
                                              reason=f"MuseTalk exited {proc.returncode}: " + " | ".join(tail)))

        # MuseTalk writes `<result_dir>/task_0.mp4` on the happy path.
        produced = os.path.join(result_dir, "task_0.mp4")
        if not os.path.exists(produced):
            # Fall back to the newest mp4 in result_dir.
            candidates = sorted(
                (os.path.join(result_dir, n) for n in os.listdir(result_dir)
                 if n.lower().endswith(".mp4")),
                key=lambda p: os.path.getmtime(p),
                reverse=True,
            )
            produced = candidates[0] if candidates else ""
        if not produced or not os.path.exists(produced):
            return _passthrough(self.name, project, input_video,
                                BackendStatus(name=self.name, available=False,
                                              reason="MuseTalk produced no output file"))

        try:
            shutil.copy2(produced, output_video)
        except Exception as e:
            return _passthrough(self.name, project, input_video,
                                BackendStatus(name=self.name, available=False,
                                              reason=f"could not copy MuseTalk output: {e}"))

        return {
            "backend": self.name,
            "rendered": True,
            "input_path": _rel(project, input_video),
            "output_path": _rel(project, output_video),
            "audio_path": _rel(project, audio_abs),
            "line_count": len(voice_lines),
            "repo_dir": cfg["repo_dir"],
            "version": cfg.get("version", "v15"),
        }

    def _collect_speech_track(self, project: Project,
                               voice_lines: List[Dict[str, Any]]) -> Optional[str]:
        """Pick the best speech-only audio to drive MuseTalk.

        Preference order:
          1. The mixer stage's speech-only intermediate, if persisted
             (artifacts/mixer_speech.wav; mixer stage may write it).
          2. A concat of per-line voice_actor wavs in scene/shot order.

        Returns a project-relative POSIX path, or None if nothing's
        available (no dialogue in this film)."""
        mixer_speech = os.path.join(project.dir, "artifacts", "mixer_speech.wav")
        if os.path.exists(mixer_speech):
            return _rel(project, mixer_speech)

        usable = [l for l in voice_lines if l.get("wav") and l.get("synthesized")]
        if not usable:
            return None
        out_rel = "artifacts/lipsync-musetalk/speech.wav"
        out_abs = os.path.join(project.dir, out_rel.replace("/", os.sep))
        os.makedirs(os.path.dirname(out_abs), exist_ok=True)

        # ffmpeg concat demuxer: write a listfile referencing each wav.
        with tempfile.NamedTemporaryFile("w", suffix=".txt",
                                           delete=False, encoding="utf-8") as lf:
            for line in usable:
                wav_rel = str(line["wav"])
                wav_abs = os.path.join(project.dir, wav_rel.replace("/", os.sep))
                if os.path.exists(wav_abs):
                    lf.write(f"file '{wav_abs.replace(chr(92), '/')}'\n")
            listfile = lf.name
        try:
            cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
                   "-i", listfile, "-c", "copy", out_abs]
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if proc.returncode != 0 or not os.path.exists(out_abs):
                logger.warning("ffmpeg concat for MuseTalk speech failed: %s", proc.stderr)
                return None
            return out_rel
        finally:
            try:
                os.unlink(listfile)
            except OSError:
                pass


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


def _posix(path: str) -> str:
    """YAML-safe path: forward slashes, no surprise backslashes."""
    return path.replace("\\", "/")


def _passthrough(backend_name: BackendName, project: Project,
                  input_video: str, status: BackendStatus) -> Dict[str, Any]:
    """Shape of a skip-artifact that still points at the untouched final
    video so continuity and the UI have something to render."""
    out = {
        "backend": backend_name,
        "rendered": False,
        "skipped": status.reason,
        "input_path": _rel(project, input_video),
        "output_path": _rel(project, input_video),
    }
    if status.install_hint:
        out["install_hint"] = status.install_hint
    return out
