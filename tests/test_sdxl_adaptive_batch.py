"""Adaptive-batch-size loop for the Shot Generator stage.

Doesn't run real SDXL — patches the batch renderer to a mock that OOMs
at size 4 and succeeds at size 2, then confirms the loop halves and
completes without erroring out."""
from __future__ import annotations

import os
import tempfile
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import pytest


class _FakeProject:
    """Just enough of `filmmaker.project.Project` for run_shots to do its job."""

    def __init__(self, tmp: str):
        self.dir = tmp
        self.shots_dir = os.path.join(tmp, "shots")
        os.makedirs(self.shots_dir, exist_ok=True)
        self.events: List[Dict[str, Any]] = []
        self._artifacts: Dict[str, Dict[str, Any]] = {}
        self.meta = SimpleNamespace(config=SimpleNamespace(
            style="stylized", quality="standard",
            sdxl_variant="turbo",
        ))

    def read_artifact(self, key: str):
        return self._artifacts.get(key)

    def write_artifact(self, key: str, data):
        self._artifacts[key] = data

    def append_event(self, ev: Dict[str, Any]):
        self.events.append(ev)


def _mock_render_batch_that_oomses_above(threshold: int):
    """Returns a callable that raises CUDA-OOM if the batch is > `threshold`
    and writes 1-byte PNGs otherwise. Enough to make the loop exercise its
    OOM-halving path."""
    def _mock(prompts, out_paths, *, steps=None, ref_images=None):
        if len(prompts) > threshold:
            raise RuntimeError("CUDA out of memory. Tried to allocate 512.00 MiB.")
        for p in out_paths:
            with open(p, "wb") as f:
                f.write(b"\x89PNG\r\n\x1a\nfake")
    return _mock


def _seed_artifacts(project: _FakeProject, num_shots: int = 6):
    project.write_artifact("breakdown", {
        "scenes": [{"id": "s01", "summary": "test scene",
                    "location": "studio", "time_of_day": "day",
                    "mood": "curious", "characters": []}],
    })
    project.write_artifact("storyboard", {
        "scenes": {"s01": [
            {"id": f"sh{i+1}", "description": f"shot {i+1}",
             "subject": "a robot", "action": "waves",
             "dialogue": ""} for i in range(num_shots)
        ]},
    })
    project.write_artifact("cinematographer", {
        "scenes": {"s01": {
            f"sh{i+1}": {"framing": "MS", "lens_mm": 35,
                         "camera_move": "static", "lighting": "soft",
                         "palette": "warm", "duration_sec": 4}
            for i in range(num_shots)
        }},
    })


def test_batch_halves_on_oom_and_completes():
    """4-shot render with a mock that OOMs at batch>=3.
    Loop should try 4 → OOM → retry with 2 → succeed on the retry, finish
    the remaining 2 in the second batch."""
    from filmmaker import agents

    with tempfile.TemporaryDirectory() as tmp:
        p = _FakeProject(tmp)
        _seed_artifacts(p, num_shots=4)

        mock_batch = _mock_render_batch_that_oomses_above(threshold=2)
        with patch.object(agents, "_load_sdxl_batch_renderer",
                          return_value=mock_batch), \
             patch.object(agents, "_load_sdxl_renderer",
                          return_value=None):
            os.environ["STUDIOLITE_SDXL_BATCH"] = "4"
            try:
                result = agents.run_shots(p)
            finally:
                os.environ.pop("STUDIOLITE_SDXL_BATCH", None)

    # Every shot got rendered (via the mock batch calls, not the placeholder).
    assert len(result["shots"]) == 4
    assert all(s["rendered"] for s in result["shots"]), \
        f"unrendered shots: {[s for s in result['shots'] if not s['rendered']]}"
    # At least one 'shots_batch' event should record the shrink.
    shrinks = [e for e in p.events if e.get("type") == "shots_batch" and e.get("shrunk")]
    assert shrinks, f"expected a shrink event, got {p.events}"
    # And a start-of-run size event with the initial 4.
    starts = [e for e in p.events if e.get("type") == "shots_batch" and e.get("size") == 4]
    assert starts, f"expected an initial size=4 event, got {p.events}"


def test_env_override_starts_at_specified_size():
    """STUDIOLITE_SDXL_BATCH=1 should force per-shot mode from the start."""
    from filmmaker import agents

    with tempfile.TemporaryDirectory() as tmp:
        p = _FakeProject(tmp)
        _seed_artifacts(p, num_shots=3)

        # per-shot renderer writes tiny PNGs
        def _single(prompt, out_path, *, steps=None, ref_image=None):
            with open(out_path, "wb") as f:
                f.write(b"\x89PNG\r\n\x1a\nsingle")

        with patch.object(agents, "_load_sdxl_batch_renderer",
                          return_value=None), \
             patch.object(agents, "_load_sdxl_renderer",
                          return_value=_single):
            os.environ["STUDIOLITE_SDXL_BATCH"] = "1"
            try:
                result = agents.run_shots(p)
            finally:
                os.environ.pop("STUDIOLITE_SDXL_BATCH", None)

    assert len(result["shots"]) == 3
    assert all(s["rendered"] for s in result["shots"])
    starts = [e for e in p.events if e.get("type") == "shots_batch" and e.get("size") == 1]
    assert starts, f"expected initial batch=1, got {p.events}"


def test_non_oom_error_falls_back_without_shrinking():
    """Non-OOM error from batch renderer should switch to per-shot for that
    chunk but leave the batch-size alone."""
    from filmmaker import agents

    with tempfile.TemporaryDirectory() as tmp:
        p = _FakeProject(tmp)
        _seed_artifacts(p, num_shots=2)

        def _boom(prompts, out_paths, *, steps=None, ref_images=None):
            raise ValueError("shape mismatch, definitely not oom")

        def _single(prompt, out_path, *, steps=None, ref_image=None):
            with open(out_path, "wb") as f:
                f.write(b"ok")

        with patch.object(agents, "_load_sdxl_batch_renderer",
                          return_value=_boom), \
             patch.object(agents, "_load_sdxl_renderer",
                          return_value=_single):
            result = agents.run_shots(p)

    assert len(result["shots"]) == 2
    assert all(s["rendered"] for s in result["shots"])
    shrinks = [e for e in p.events if e.get("type") == "shots_batch" and e.get("shrunk")]
    assert not shrinks, f"unexpected shrink on non-OOM: {shrinks}"
