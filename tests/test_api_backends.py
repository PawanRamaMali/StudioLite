"""Tests for the GET /api/v1/backends endpoint.

The endpoint is a thin JSON view of filmmaker.backend_registry plus a
soft report of which cloud providers are configured. These tests pin
response shape and the stage-filter behaviour so a schema drift in the
registry forces a conscious test update."""
from __future__ import annotations

import importlib
import os
import sys

import pytest


def _fresh_api(monkeypatch, tmp_path, *, auth: str = "off"):
    """Reload api_server with auth off by default - these are public reads
    and we want to test the response shape, not the auth layer."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("STUDIOLITE_AUTH", auth)
    for name in list(sys.modules):
        if name == "api_server" or name.startswith("api_server."):
            del sys.modules[name]
    api = importlib.import_module("api_server")
    from fastapi.testclient import TestClient
    return api, TestClient(api.app)


def test_backends_response_shape(monkeypatch, tmp_path):
    _api, client = _fresh_api(monkeypatch, tmp_path)
    r = client.get("/api/v1/backends")
    assert r.status_code == 200
    body = r.json()
    assert "backends" in body
    assert "cloud_providers" in body
    assert isinstance(body["backends"], list)
    assert body["backends"], "registry should have at least one backend"
    first = body["backends"][0]
    for key in ("stage", "name", "license", "available",
                 "min_vram_gb_fp16", "cloud_burst", "base_model",
                 "quant_variants", "source_url", "notes"):
        assert key in first, f"missing key in backend spec: {key}"


def test_backends_stage_filter_narrows_response(monkeypatch, tmp_path):
    _api, client = _fresh_api(monkeypatch, tmp_path)
    r = client.get("/api/v1/backends", params={"stage": "upscale"})
    assert r.status_code == 200
    body = r.json()
    stages = {b["stage"] for b in body["backends"]}
    assert stages == {"upscale"}
    names = {b["name"] for b in body["backends"]}
    assert {"none", "realesrgan_x2", "realesrgan_x4", "seedvr2"} <= names


def test_backends_unknown_stage_returns_empty_list(monkeypatch, tmp_path):
    _api, client = _fresh_api(monkeypatch, tmp_path)
    r = client.get("/api/v1/backends", params={"stage": "nonexistent"})
    assert r.status_code == 200
    assert r.json()["backends"] == []


def test_backends_roadmap_entries_are_flagged(monkeypatch, tmp_path):
    _api, client = _fresh_api(monkeypatch, tmp_path)
    r = client.get("/api/v1/backends", params={"stage": "upscale"})
    body = r.json()
    seedvr2 = next(b for b in body["backends"] if b["name"] == "seedvr2")
    assert seedvr2["available"] == "roadmap"
    assert seedvr2["license"] == "Apache-2.0"
    # Quant ladder is serialised in full.
    formats = {q["format"] for q in seedvr2["quant_variants"]}
    assert {"gguf_q3_k_m", "gguf_q4_k_m", "gguf_q5_k_m",
            "gguf_q6_k", "gguf_q8_0"} <= formats


def test_backends_cloud_providers_without_env_are_all_false(monkeypatch, tmp_path):
    monkeypatch.delenv("REPLICATE_API_TOKEN", raising=False)
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    _api, client = _fresh_api(monkeypatch, tmp_path)
    body = client.get("/api/v1/backends").json()
    cp = body["cloud_providers"]
    # dry-run is always available; the others require env.
    assert cp.get("replicate") is False
    assert cp.get("runpod") is False
    assert cp.get("dry-run") is True


def test_backends_cloud_provider_lights_up_when_token_set(monkeypatch, tmp_path):
    monkeypatch.setenv("REPLICATE_API_TOKEN", "token")
    _api, client = _fresh_api(monkeypatch, tmp_path)
    body = client.get("/api/v1/backends").json()
    assert body["cloud_providers"].get("replicate") is True


# ---------------------------------------------------------------------------
# GET /api/v1/films/{id}/vbench

def _make_project(api, title: str = "Beacon"):
    """Create a project via the module's own helpers so paths line up with
    its OUTPUT_DIR."""
    return api.filmmaker_projects.create(
        api.OUTPUT_DIR, brief="A keeper opens a letter.",
        title=title, config_dict={
            "llm_backend": "ollama", "llm_model": "mock",
            "style": "stylized", "target_minutes": 1.0,
        },
    )


def test_vbench_endpoint_404_for_missing_project(monkeypatch, tmp_path):
    _api, client = _fresh_api(monkeypatch, tmp_path)
    r = client.get("/api/v1/films/does-not-exist/vbench")
    assert r.status_code == 404


def test_vbench_endpoint_returns_stored_scorecard(monkeypatch, tmp_path):
    """When artifacts/vbench2.json already exists, the endpoint returns
    it verbatim without re-running the harness."""
    api, client = _fresh_api(monkeypatch, tmp_path)
    # Import projects as an attribute of the test module so _make_project
    # does not depend on api_server's internals too tightly.
    from filmmaker import projects as _fp
    api.filmmaker_projects = _fp  # type: ignore[attr-defined]
    proj = _make_project(api)

    import os as _os
    import json as _json
    stored_dir = _os.path.join(proj.dir, "artifacts")
    _os.makedirs(stored_dir, exist_ok=True)
    stored = _os.path.join(stored_dir, "vbench2.json")
    canned = {"clip_count": 7, "per_clip": [], "aggregate": [],
              "overall": 0.4242}
    with open(stored, "w", encoding="utf-8") as f:
        _json.dump(canned, f)

    r = client.get(f"/api/v1/films/{proj.project_id}/vbench")
    assert r.status_code == 200
    assert r.json() == canned


def test_vbench_endpoint_refresh_overrides_stored(monkeypatch, tmp_path):
    """?refresh=true re-runs score_project and overwrites the file.

    We register one constant scorer so the harness math is predictable
    without ffmpeg installed."""
    api, client = _fresh_api(monkeypatch, tmp_path)
    from filmmaker import projects as _fp, vbench2 as _vb
    api.filmmaker_projects = _fp  # type: ignore[attr-defined]
    proj = _make_project(api)

    import os as _os
    import json as _json
    stored = _os.path.join(proj.dir, "artifacts", "vbench2.json")
    _os.makedirs(_os.path.dirname(stored), exist_ok=True)
    with open(stored, "w", encoding="utf-8") as f:
        _json.dump({"clip_count": 99}, f)

    # Add a scorer that returns a known value, and a motion_shots artifact
    # pointing at one clip that exists on disk.
    _vb.register_scorer(_vb.ScorerSpec(
        name="probe", dimension="physics", capability="mechanics",
        fn=lambda _p: 0.75,
    ))
    clip = _os.path.join(proj.dir, "clip.mp4")
    with open(clip, "wb") as f:
        f.write(b"stub")
    proj.write_artifact("motion_shots",
                         {"shots": [{"output_path": "clip.mp4"}]})

    try:
        r = client.get(f"/api/v1/films/{proj.project_id}/vbench?refresh=true")
        assert r.status_code == 200
        body = r.json()
        assert body["clip_count"] == 1
        phys = next(d for d in body["aggregate"] if d["dimension"] == "physics")
        assert phys["score"] == pytest.approx(0.75)
    finally:
        # Clean up the probe scorer so later tests see a stable registry.
        _vb._SCORERS[:] = [s for s in _vb._SCORERS if s.name != "probe"]


# ---------------------------------------------------------------------------
# GET /api/v1/vbench/corpus

def test_vbench_corpus_without_env_returns_missing(monkeypatch, tmp_path):
    monkeypatch.delenv("VBENCH2_PROMPT_DIR", raising=False)
    _api, client = _fresh_api(monkeypatch, tmp_path)
    r = client.get("/api/v1/vbench/corpus")
    assert r.status_code == 200
    body = r.json()
    assert body["corpus_dir"] is None
    assert body["exists"] is False
    assert body["total_prompts"] == 0
    assert body["capabilities"] == []
    # Every paper capability counts as missing when the dir isn't set.
    assert "mechanics" in body["missing_capabilities"]


def test_vbench_corpus_reads_env_and_reports_counts(monkeypatch, tmp_path):
    """With VBENCH2_PROMPT_DIR set, the endpoint lists every JSON file it
    found, enriches each with the paper's dimension (or known=False for
    unknown filenames), and reports the still-missing capability names."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "mechanics.json").write_text(
        '["ball drops", "water pours"]', encoding="utf-8")
    (corpus / "composition.json").write_text(
        '{"prompts": ["wide shot"]}', encoding="utf-8")
    (corpus / "typo_cap.json").write_text('["nope"]', encoding="utf-8")
    monkeypatch.setenv("VBENCH2_PROMPT_DIR", str(corpus))

    _api, client = _fresh_api(monkeypatch, tmp_path)
    r = client.get("/api/v1/vbench/corpus")
    assert r.status_code == 200
    body = r.json()
    assert body["exists"] is True
    assert body["total_prompts"] == 4
    by_cap = {r["capability"]: r for r in body["capabilities"]}
    assert by_cap["mechanics"]["dimension"] == "physics"
    assert by_cap["mechanics"]["known"] is True
    assert by_cap["mechanics"]["prompt_count"] == 2
    assert by_cap["typo_cap"]["known"] is False
    assert "mechanics" not in body["missing_capabilities"]


def test_vbench_corpus_dir_query_overrides_env(monkeypatch, tmp_path):
    """?dir=... lets the caller inspect any directory, including a path
    other than the one the server was launched with."""
    monkeypatch.delenv("VBENCH2_PROMPT_DIR", raising=False)
    corpus = tmp_path / "elsewhere"
    corpus.mkdir()
    (corpus / "material.json").write_text(
        '["cloth tears"]', encoding="utf-8")

    _api, client = _fresh_api(monkeypatch, tmp_path)
    r = client.get(f"/api/v1/vbench/corpus?dir={corpus}")
    assert r.status_code == 200
    body = r.json()
    assert body["exists"] is True
    assert body["total_prompts"] == 1
    assert body["capabilities"][0]["capability"] == "material"


# ---------------------------------------------------------------------------
# POST /api/v1/lipsync/eval

def test_lipsync_eval_404_for_missing_project(monkeypatch, tmp_path):
    _api, client = _fresh_api(monkeypatch, tmp_path)
    r = client.post("/api/v1/lipsync/eval",
                     json={"project_id": "nope", "corpus": [], "backends": []})
    assert r.status_code == 404


def test_lipsync_eval_422_without_reference(monkeypatch, tmp_path):
    api, client = _fresh_api(monkeypatch, tmp_path)
    from filmmaker import projects as _fp
    api.filmmaker_projects = _fp  # type: ignore[attr-defined]
    proj = _make_project(api)
    r = client.post("/api/v1/lipsync/eval", json={
        "project_id": proj.project_id,
        "corpus": [{"audio": "a.wav"}],
        "backends": ["musetalk"],
    })
    assert r.status_code == 422


def test_lipsync_eval_matrix_shape(monkeypatch, tmp_path):
    """With all three backends uninstalled (no config.json for any),
    the matrix still returns a well-shaped scorecard - rendered_count
    is zero and every heavy metric aggregates to None."""
    api, client = _fresh_api(monkeypatch, tmp_path)
    from filmmaker import projects as _fp
    api.filmmaker_projects = _fp  # type: ignore[attr-defined]
    proj = _make_project(api)
    # Reference exists on disk so the matrix can at least call ffprobe.
    import os as _os
    ref = _os.path.join(proj.dir, "ref.mp4")
    with open(ref, "wb") as f:
        f.write(b"stub")

    r = client.post("/api/v1/lipsync/eval", json={
        "project_id": proj.project_id,
        "corpus": [{"reference": "ref.mp4", "audio": None}],
        "backends": ["musetalk", "latentsync", "echomimicv3"],
    })
    assert r.status_code == 200
    body = r.json()
    assert body["corpus_count"] == 1
    names = {row["backend"] for row in body["backends"]}
    assert names == {"musetalk", "latentsync", "echomimicv3"}
    for row in body["backends"]:
        assert row["rendered_count"] == 0
        assert row["metrics"]["syncnet_confidence"] is None
