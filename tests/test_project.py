"""Unit tests for filmmaker/project.py - persistence, staleness
propagation, and config parsing."""
from __future__ import annotations

import json
import os

import pytest

from filmmaker.project import Project, ProjectConfig, _meta_from_json
from filmmaker import stages


def _make_project(tmp_path, brief="A test brief.", **cfg_overrides):
    """Small helper that mints a fresh project rooted at `tmp_path`.
    Each test gets its own root so parallel-mode pytest doesn't step on
    itself and the tearDown is `tmp_path` itself."""
    cfg = ProjectConfig(**cfg_overrides)
    return Project.create(str(tmp_path), brief=brief, title=None, config=cfg)


def test_create_writes_meta_and_state(tmp_path):
    p = _make_project(tmp_path)
    assert os.path.exists(p.meta_path), "meta file not written"
    assert os.path.exists(p.state_path), "state file not written"
    meta = json.loads(open(p.meta_path, encoding="utf-8").read())
    assert meta["brief"] == "A test brief."
    state = json.loads(open(p.state_path, encoding="utf-8").read())
    # Every stage should start pending - the orchestrator flips these.
    assert set(state["stage_status"].values()) == {"pending"}


def test_config_backends_default_to_conservative_paths(tmp_path):
    p = _make_project(tmp_path)
    cfg = p.meta.config
    # These defaults matter for a fresh install: nothing that would
    # trigger a big model download or open a network port.
    assert cfg.voice_backend == "piper"
    assert cfg.motion_backend == "auto"
    assert cfg.sdxl_variant == "turbo"
    assert cfg.upscale_backend == "none"


def test_stage_status_write_survives_reload(tmp_path):
    p = _make_project(tmp_path)
    p.set_stage_status("producer", "done")
    # Re-open via a fresh handle to prove the JSON was actually flushed.
    p2 = Project.load(str(tmp_path), p.project_id)
    assert p2.state.stage_status["producer"] == "done"


def test_mark_downstream_stale_only_nudges_done_stages(tmp_path):
    """Editing producer should make screenwriter etc. stale iff they
    had completed. Pending/failed stages must not be silently overwritten."""
    p = _make_project(tmp_path)
    p.set_stage_status("producer", "done")
    p.set_stage_status("screenwriter", "done")
    p.set_stage_status("story_editor", "failed")  # deliberately not done
    p.mark_downstream_stale("producer")
    st = p.state
    assert st.stage_status["screenwriter"] == "stale"
    # failed stays failed - mark_downstream_stale must not clobber it
    assert st.stage_status["story_editor"] == "failed"


def test_meta_from_json_fills_missing_backends(tmp_path):
    """_meta_from_json is what disk load runs; it needs to tolerate the
    older shape (before we added music_backend / upscale_backend)."""
    raw = {
        "id": "film-abc",
        "title": "T",
        "brief": "b",
        "config": {"llm_backend": "ollama", "llm_model": "qwen2.5:14b"},
        "created_at": 1.0,
        "updated_at": 2.0,
    }
    meta = _meta_from_json(raw)
    assert meta.id == "film-abc"
    assert meta.config.music_backend == "musicgen"
    assert meta.config.upscale_backend == "none"


def test_read_events_returns_empty_when_never_written(tmp_path):
    p = _make_project(tmp_path)
    assert p.read_events() == []


def test_append_event_reads_back_in_order(tmp_path):
    p = _make_project(tmp_path)
    p.append_event({"type": "a"})
    p.append_event({"type": "b"})
    events = p.read_events()
    assert [e["type"] for e in events] == ["a", "b"]
    assert all("ts" in e for e in events), "timestamp should be stamped in"


def test_downstream_of_producer_covers_expected_span(tmp_path):
    """Cross-check: mark_downstream_stale should hit exactly the same
    stages that downstream_of returns."""
    down = stages.downstream_of("producer")
    p = _make_project(tmp_path)
    for k in stages.STAGE_KEYS:
        p.set_stage_status(k, "done")
    p.mark_downstream_stale("producer")
    st = p.state
    # Producer itself is not downstream of producer; leave it alone.
    assert st.stage_status["producer"] == "done"
    for k in down:
        assert st.stage_status[k] == "stale", f"{k} should be stale"
