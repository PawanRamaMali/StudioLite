"""
On-disk project model.

Every film is a directory under `.mp/films/<project_id>/` holding
`project.json`, `state.json`, an `artifacts/` tree, and rendered media.
The Project class is the only code that touches those files — everything
else in the pipeline reads/writes via its helpers.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from .stages import STAGE_KEYS, StageKey, StageStatus, downstream_of


@dataclass
class ProjectConfig:
    # Free-form dict of per-agent overrides. Each entry may hold model/host
    # info; keys are stage keys. Additive; missing = defaults.
    per_stage: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # LLM backend. Only "ollama" is wired in T1; "gemini"/"groq"/"hf" are
    # accepted for forward compatibility.
    llm_backend: str = "ollama"
    llm_model: str = "llama3.2"
    llm_host: Optional[str] = None  # e.g. http://localhost:11434
    # Visual style — passed to Cinematographer and Shot Generator prompts.
    style: str = "stylized"          # "stylized" | "photoreal"
    # Target runtime hint the Producer and Breakdown agents anchor on.
    target_minutes: float = 2.0
    # SDXL inference-step preset. Trades wall-clock for image quality.
    # Values: "draft" (fast iteration), "standard" (default),
    # "high" (finishing look), "ultra" (final render).
    quality: str = "standard"
    # SDXL variant. "turbo" is fast (12 steps, distilled) but faces come
    # out flat and hands often mangle. "base" uses SDXL 1.0 (30-50 steps,
    # much higher fidelity, slower). "flux" is not wired yet.
    sdxl_variant: str = "turbo"
    # Motion backend for motion_shots. "auto" tries SVD -> Wan -> Ken Burns.
    # "animatediff" is faster (~30s per 16 frames) but style drifts from
    # the SDXL keyframe. "svd" locks composition to the keyframe but is
    # slow on 12GB (5-10 min per clip). "wan" is text-only T2V. "kenburns"
    # skips motion entirely and pans on the stills.
    motion_backend: str = "auto"
    # Voice backend for voice_actor. "piper" (default) is fast, offline,
    # reads at a flat register. "xtts" uses Coqui XTTS-v2 which supports
    # voice cloning from a 6s reference clip and emotion, at the cost of
    # slower synth (~5-10x Piper) and a ~2GB model download on first use.
    voice_backend: str = "piper"


@dataclass
class ProjectMeta:
    id: str
    title: str
    brief: str                       # user's original one-paragraph pitch
    config: ProjectConfig
    created_at: float
    updated_at: float


@dataclass
class ProjectState:
    """
    Mutable per-run state. `stage_status` covers every stage in STAGES.
    `current_stage` is the stage the orchestrator is (or last was) working on.
    `gates` maps stage → whether a review gate is enabled; blank = default.
    """
    stage_status: Dict[str, StageStatus]
    current_stage: Optional[str] = None
    is_paused: bool = False
    pause_requested: bool = False
    last_error: Optional[str] = None
    gates: Dict[str, bool] = field(default_factory=dict)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None


class Project:
    """
    Wraps a project directory. Cheap to construct — reads project.json
    and state.json each time you touch them (small files, no cache), so
    a background orchestrator writing on one side and the API reading on
    the other stay coherent without an in-process lock across processes.
    Within one process a per-project RLock guards write bursts.
    """

    _locks: Dict[str, threading.RLock] = {}
    _locks_guard = threading.Lock()

    def __init__(self, root_dir: str, project_id: str):
        self.project_id = project_id
        self.dir = os.path.join(root_dir, project_id)
        self.artifacts_dir = os.path.join(self.dir, "artifacts")
        self.logs_dir = os.path.join(self.dir, "logs")
        self.shots_dir = os.path.join(self.artifacts_dir, "shots")
        self.meta_path = os.path.join(self.dir, "project.json")
        self.state_path = os.path.join(self.dir, "state.json")
        self.events_path = os.path.join(self.logs_dir, "events.jsonl")

    # ---- lifecycle -------------------------------------------------------

    @classmethod
    def create(cls, root_dir: str, brief: str, title: Optional[str], config: ProjectConfig) -> "Project":
        pid = f"film-{int(time.time())}-{uuid.uuid4().hex[:6]}"
        proj = cls(root_dir, pid)
        os.makedirs(proj.artifacts_dir, exist_ok=True)
        os.makedirs(proj.logs_dir, exist_ok=True)
        os.makedirs(proj.shots_dir, exist_ok=True)
        now = time.time()
        meta = ProjectMeta(
            id=pid,
            title=title or "Untitled Film",
            brief=brief.strip(),
            config=config,
            created_at=now,
            updated_at=now,
        )
        state = ProjectState(
            stage_status={k: "pending" for k in STAGE_KEYS},
        )
        proj._write_json(proj.meta_path, _meta_to_json(meta))
        proj._write_json(proj.state_path, _state_to_json(state))
        return proj

    @classmethod
    def load(cls, root_dir: str, project_id: str) -> "Project":
        proj = cls(root_dir, project_id)
        if not os.path.exists(proj.meta_path):
            raise FileNotFoundError(f"Project not found: {project_id}")
        return proj

    @classmethod
    def list_all(cls, root_dir: str) -> List[Dict[str, Any]]:
        if not os.path.isdir(root_dir):
            return []
        out: List[Dict[str, Any]] = []
        for name in os.listdir(root_dir):
            meta_path = os.path.join(root_dir, name, "project.json")
            if not os.path.exists(meta_path):
                continue
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                out.append({
                    "id": meta.get("id", name),
                    "title": meta.get("title", "Untitled Film"),
                    "created_at": meta.get("created_at", 0),
                    "updated_at": meta.get("updated_at", 0),
                    "brief": meta.get("brief", ""),
                })
            except Exception:
                continue
        out.sort(key=lambda p: p.get("updated_at", 0), reverse=True)
        return out

    def delete(self) -> None:
        import shutil
        shutil.rmtree(self.dir, ignore_errors=True)

    # ---- meta / state ----------------------------------------------------

    @property
    def meta(self) -> ProjectMeta:
        return _meta_from_json(self._read_json(self.meta_path))

    def save_meta(self, meta: ProjectMeta) -> None:
        meta.updated_at = time.time()
        self._write_json(self.meta_path, _meta_to_json(meta))

    @property
    def state(self) -> ProjectState:
        raw = self._read_json(self.state_path)
        return _state_from_json(raw)

    def save_state(self, state: ProjectState) -> None:
        self._write_json(self.state_path, _state_to_json(state))
        self._bump_updated()

    def update_state(self, **changes: Any) -> ProjectState:
        with self._lock():
            state = self.state
            for k, v in changes.items():
                setattr(state, k, v)
            self.save_state(state)
            return state

    def set_stage_status(self, key: StageKey, status: StageStatus) -> ProjectState:
        with self._lock():
            state = self.state
            state.stage_status[key] = status
            self.save_state(state)
            return state

    def mark_downstream_stale(self, key: StageKey) -> None:
        """When an upstream artifact is edited, mark downstream stages stale."""
        with self._lock():
            state = self.state
            for down_key in downstream_of(key):
                # Preserve failed/pending — only nudge completed ones.
                if state.stage_status.get(down_key) in ("done", "needs_review"):
                    state.stage_status[down_key] = "stale"
            self.save_state(state)

    # ---- artifacts -------------------------------------------------------

    def artifact_path(self, key: StageKey) -> str:
        return os.path.join(self.artifacts_dir, f"{key}.json")

    def has_artifact(self, key: StageKey) -> bool:
        return os.path.exists(self.artifact_path(key))

    def read_artifact(self, key: StageKey) -> Optional[Dict[str, Any]]:
        p = self.artifact_path(key)
        if not os.path.exists(p):
            return None
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)

    def write_artifact(self, key: StageKey, data: Dict[str, Any]) -> None:
        with self._lock():
            self._write_json(self.artifact_path(key), data)
            self._bump_updated()

    # ---- events ----------------------------------------------------------

    def append_event(self, event: Dict[str, Any]) -> None:
        event = {"ts": time.time(), **event}
        with self._lock():
            with open(self.events_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")

    def read_events(self, tail: int = 200) -> List[Dict[str, Any]]:
        if not os.path.exists(self.events_path):
            return []
        with open(self.events_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        out = []
        for line in lines[-tail:]:
            try:
                out.append(json.loads(line))
            except Exception:
                continue
        return out

    # ---- internals -------------------------------------------------------

    def _lock(self) -> threading.RLock:
        with Project._locks_guard:
            lock = Project._locks.get(self.project_id)
            if lock is None:
                lock = threading.RLock()
                Project._locks[self.project_id] = lock
            return lock

    def _bump_updated(self) -> None:
        try:
            meta = self.meta
            meta.updated_at = time.time()
            self._write_json(self.meta_path, _meta_to_json(meta))
        except Exception:
            pass  # meta write is best-effort

    @staticmethod
    def _read_json(path: str) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _write_json(path: str, data: Dict[str, Any]) -> None:
        # Atomic-ish: write to tmp then replace, so a crashed process never
        # leaves half a JSON file behind for the next reader.
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)


# ---------- (de)serialization helpers, kept out of the class for clarity ----

def _meta_to_json(meta: ProjectMeta) -> Dict[str, Any]:
    d = asdict(meta)
    return d


def _meta_from_json(d: Dict[str, Any]) -> ProjectMeta:
    cfg_raw = d.get("config") or {}
    cfg = ProjectConfig(
        per_stage=cfg_raw.get("per_stage") or {},
        llm_backend=cfg_raw.get("llm_backend", "ollama"),
        llm_model=cfg_raw.get("llm_model", "llama3.2"),
        llm_host=cfg_raw.get("llm_host"),
        style=cfg_raw.get("style", "stylized"),
        target_minutes=float(cfg_raw.get("target_minutes", 2.0)),
        quality=str(cfg_raw.get("quality", "standard")).lower(),
        sdxl_variant=str(cfg_raw.get("sdxl_variant", "turbo")).lower(),
        motion_backend=str(cfg_raw.get("motion_backend", "auto")).lower(),
        voice_backend=str(cfg_raw.get("voice_backend", "piper")).lower(),
    )
    return ProjectMeta(
        id=d["id"],
        title=d.get("title", "Untitled Film"),
        brief=d.get("brief", ""),
        config=cfg,
        created_at=float(d.get("created_at", 0)),
        updated_at=float(d.get("updated_at", 0)),
    )


def _state_to_json(state: ProjectState) -> Dict[str, Any]:
    return asdict(state)


def _state_from_json(d: Dict[str, Any]) -> ProjectState:
    return ProjectState(
        stage_status=dict(d.get("stage_status") or {}),
        current_stage=d.get("current_stage"),
        is_paused=bool(d.get("is_paused", False)),
        pause_requested=bool(d.get("pause_requested", False)),
        last_error=d.get("last_error"),
        gates=dict(d.get("gates") or {}),
        started_at=d.get("started_at"),
        finished_at=d.get("finished_at"),
    )
