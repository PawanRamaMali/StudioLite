"""Unit tests for filmmaker.projects.apply_config_patch plus a round-trip
test through the HTTP PATCH endpoint. The helper is where validation and
the stale-stage mapping live; the endpoint test just confirms the wire-up
and the running-run guard."""
from __future__ import annotations

import importlib
import sys

import pytest

from filmmaker import projects as fp
from filmmaker.project import Project, ProjectConfig


def _mkproj(tmp_path, **cfg_overrides):
    cfg = ProjectConfig(**cfg_overrides)
    return Project.create(str(tmp_path), brief="A test brief.", title=None, config=cfg)


def test_patch_backend_marks_affected_stage(tmp_path):
    p = _mkproj(tmp_path)
    affected = fp.apply_config_patch(p, {"lipsync_backend": "musetalk"})
    assert affected == ["lipsync"]
    assert p.meta.config.lipsync_backend == "musetalk"


def test_patch_lowercases_enum_fields(tmp_path):
    p = _mkproj(tmp_path)
    fp.apply_config_patch(p, {"voice_backend": "XTTS"})
    assert p.meta.config.voice_backend == "xtts"


def test_patch_rejects_unknown_field(tmp_path):
    p = _mkproj(tmp_path)
    with pytest.raises(ValueError, match="unknown config fields"):
        fp.apply_config_patch(p, {"not_a_field": "x"})


def test_patch_rejects_bad_target_minutes(tmp_path):
    p = _mkproj(tmp_path)
    with pytest.raises(ValueError, match="target_minutes"):
        fp.apply_config_patch(p, {"target_minutes": "soon"})


def test_patch_noop_when_value_unchanged(tmp_path):
    p = _mkproj(tmp_path)
    assert fp.apply_config_patch(p, {"lipsync_backend": "none"}) == []
    assert fp.apply_config_patch(p, {"upscale_backend": "none"}) == []


def test_patch_llm_host_blank_clears(tmp_path):
    p = _mkproj(tmp_path)
    fp.apply_config_patch(p, {"llm_host": "http://localhost:11434"})
    assert p.meta.config.llm_host == "http://localhost:11434"
    fp.apply_config_patch(p, {"llm_host": ""})
    assert p.meta.config.llm_host is None


def test_patch_quality_affects_shots(tmp_path):
    p = _mkproj(tmp_path)
    affected = fp.apply_config_patch(p, {"quality": "high"})
    assert affected == ["shots"]
    assert p.meta.config.quality == "high"


def test_patch_multiple_fields_collects_each_affected_stage(tmp_path):
    p = _mkproj(tmp_path)
    affected = fp.apply_config_patch(p, {
        "lipsync_backend": "musetalk",
        "upscale_backend": "realesrgan_x2",
        "style": "photoreal",  # style is patchable but doesn't map to a stage
    })
    assert set(affected) == {"lipsync", "upscale"}
    assert p.meta.config.style == "photoreal"


def test_patch_persists_to_disk(tmp_path):
    p = _mkproj(tmp_path)
    fp.apply_config_patch(p, {"voice_backend": "xtts"})
    reopened = Project.load(str(tmp_path), p.project_id)
    assert reopened.meta.config.voice_backend == "xtts"


# ---- HTTP endpoint ---------------------------------------------------------

def _fresh_api(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("STUDIOLITE_AUTH", "off")
    for name in list(sys.modules):
        if name == "api_server" or name.startswith("api_server."):
            del sys.modules[name]
    api = importlib.import_module("api_server")
    from fastapi.testclient import TestClient
    return api, TestClient(api.app)


def test_endpoint_patches_and_marks_stale(monkeypatch, tmp_path):
    api, client = _fresh_api(monkeypatch, tmp_path)
    create = client.post("/api/v1/films", json={"brief": "A quick test brief"})
    assert create.status_code == 200, create.text
    pid = create.json()["project"]["id"]
    # Mark lipsync as done so the patch has something to reset.
    proj = api._fp_load_for_test(pid) if hasattr(api, "_fp_load_for_test") else None
    if proj is None:
        from filmmaker import projects as _fp
        proj = _fp.load(api.OUTPUT_DIR, pid)
    proj.set_stage_status("lipsync", "done")

    r = client.patch(f"/api/v1/films/{pid}/config",
                     json={"patch": {"lipsync_backend": "musetalk"}})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["affected_stages"] == ["lipsync"]
    assert body["project"]["config"]["lipsync_backend"] == "musetalk"
    assert body["state"]["stage_status"]["lipsync"] == "pending"


def test_endpoint_rejects_unknown_field_with_422(monkeypatch, tmp_path):
    api, client = _fresh_api(monkeypatch, tmp_path)
    create = client.post("/api/v1/films", json={"brief": "A quick test brief"})
    pid = create.json()["project"]["id"]
    r = client.patch(f"/api/v1/films/{pid}/config",
                     json={"patch": {"not_a_field": "x"}})
    assert r.status_code == 422
    assert "unknown config fields" in r.json()["detail"]


def test_endpoint_404_for_missing_project(monkeypatch, tmp_path):
    api, client = _fresh_api(monkeypatch, tmp_path)
    r = client.patch("/api/v1/films/does-not-exist/config",
                     json={"patch": {"voice_backend": "xtts"}})
    assert r.status_code == 404
