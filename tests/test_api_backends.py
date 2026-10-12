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
