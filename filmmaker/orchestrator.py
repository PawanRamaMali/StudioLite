"""
Pipeline orchestrator.

The orchestrator drives a Project's stages in order, checkpointing state
after every completed stage so a paused/crashed run can pick up cleanly.
Pause is cooperative: it's honored between stages, not mid-stage. This is
enough for T1 because the longest-running T1 stages are FFmpeg (which we
just wait for) and the LLM calls (which are already atomic).

The runner is asyncio-based so it plays nicely with FastAPI. A single-item
run task per project id is tracked in _runs; starting a new one on an
already-running project is a no-op.
"""

from __future__ import annotations

import asyncio
import logging
import time
import traceback
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from .project import Project
from .registry import get_runner
from .stages import STAGES, StageKey, StageSpec

logger = logging.getLogger("studiolite.filmmaker.orch")


EventListener = Callable[[Dict], None]


@dataclass
class _RunHandle:
    task: asyncio.Task
    listeners: List[EventListener] = field(default_factory=list)


class RunManager:
    """Process-wide registry of active runs. One run per project."""

    def __init__(self) -> None:
        self._runs: Dict[str, _RunHandle] = {}
        self._lock = asyncio.Lock()

    async def start(self, project: Project, only_stale: bool = False) -> bool:
        """Kick off (or resume) the pipeline for the given project.

        Returns False if a run is already active. When `only_stale` is True,
        already-`done` stages are left alone — used after an artifact edit
        to re-run just the affected downstream chain.
        """
        async with self._lock:
            if project.project_id in self._runs and not self._runs[project.project_id].task.done():
                return False
            # Clear any leftover completed handle before starting the new one.
            self._runs.pop(project.project_id, None)
            task = asyncio.create_task(self._run(project, only_stale=only_stale))
            self._runs[project.project_id] = _RunHandle(task=task)
        return True

    def is_running(self, project_id: str) -> bool:
        h = self._runs.get(project_id)
        return h is not None and not h.task.done()

    def request_pause(self, project: Project) -> bool:
        if not self.is_running(project.project_id):
            return False
        project.update_state(pause_requested=True)
        return True

    def subscribe(self, project_id: str, cb: EventListener) -> Callable[[], None]:
        h = self._runs.get(project_id)
        if h is None:
            # Allow subscribing to a non-running project (e.g. right after start()).
            h = _RunHandle(task=asyncio.get_event_loop().create_future())  # type: ignore[arg-type]
            self._runs[project_id] = h
        h.listeners.append(cb)

        def _unsub() -> None:
            try:
                h.listeners.remove(cb)
            except ValueError:
                pass
        return _unsub

    # ---- internal --------------------------------------------------------

    def _broadcast(self, project_id: str, event: Dict) -> None:
        h = self._runs.get(project_id)
        if not h:
            return
        for cb in list(h.listeners):
            try:
                cb(event)
            except Exception:
                logger.exception("event listener raised; dropping it")
                try:
                    h.listeners.remove(cb)
                except ValueError:
                    pass

    async def _run(self, project: Project, *, only_stale: bool) -> None:
        state = project.update_state(
            is_paused=False, pause_requested=False,
            started_at=time.time(), last_error=None,
        )
        self._emit(project, {"type": "run_started", "project_id": project.project_id})

        try:
            for spec in STAGES:
                # Pause check between stages.
                if project.state.pause_requested:
                    project.update_state(is_paused=True, pause_requested=False)
                    self._emit(project, {"type": "paused", "stage": spec.key})
                    return

                status = project.state.stage_status.get(spec.key, "pending")
                if only_stale and status == "done":
                    continue  # skip already-done stages during a re-run

                # Skip already-done unless we're resuming from a stale one.
                if status == "done" and not only_stale:
                    self._emit(project, {"type": "stage_skipped", "stage": spec.key, "reason": "already done"})
                    continue

                await self._run_one(project, spec)

                # Gate check after successful stage.
                gate_on = project.state.gates.get(spec.key, spec.gated_by_default)
                if gate_on:
                    project.set_stage_status(spec.key, "needs_review")
                    project.update_state(is_paused=True)
                    self._emit(project, {"type": "gate_hit", "stage": spec.key})
                    return

            # All stages complete
            project.update_state(finished_at=time.time())
            self._emit(project, {"type": "run_finished"})
        except Exception as e:
            tb = traceback.format_exc()
            logger.exception("orchestrator failed on project %s", project.project_id)
            project.update_state(last_error=str(e), is_paused=True)
            self._emit(project, {"type": "run_failed", "error": str(e), "traceback": tb[-2000:]})

    async def _run_one(self, project: Project, spec: StageSpec) -> None:
        project.set_stage_status(spec.key, "running")
        project.update_state(current_stage=spec.key)
        self._emit(project, {"type": "stage_started", "stage": spec.key, "label": spec.label})
        started = time.time()

        runner = get_runner(spec.key)
        try:
            data = await asyncio.to_thread(runner, project)
            project.write_artifact(spec.key, data)
            project.set_stage_status(spec.key, "done")
            self._emit(project, {
                "type": "stage_completed",
                "stage": spec.key,
                "elapsed": round(time.time() - started, 2),
            })
        except Exception as e:
            project.set_stage_status(spec.key, "failed")
            project.update_state(last_error=f"{spec.key}: {e}", is_paused=True)
            self._emit(project, {
                "type": "stage_failed",
                "stage": spec.key,
                "error": str(e),
            })
            raise

    def _emit(self, project: Project, event: Dict) -> None:
        project.append_event(event)
        self._broadcast(project.project_id, event)


# Global singleton — one manager per FastAPI process.
manager = RunManager()
