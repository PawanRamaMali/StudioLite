"""Extracted env-vars router: mirrors the behavior the inline endpoints
used to have, so a regression here catches a mount-order or extraction
mistake before the UI sees it."""
from __future__ import annotations

import importlib
import os
import sys

import pytest


def _client(monkeypatch, tmp_path, *, auth="off"):
    from fastapi.testclient import TestClient
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("STUDIOLITE_AUTH", auth)
    for name in list(sys.modules):
        if name == "api_server":
            del sys.modules[name]
    api = importlib.import_module("api_server")
    return api, TestClient(api.app)


class TestGet:
    def test_returns_managed_keys(self, monkeypatch, tmp_path):
        _api, client = _client(monkeypatch, tmp_path)
        r = client.get("/api/v1/system/env")
        assert r.status_code == 200
        body = r.json()
        assert "managed_keys" in body
        assert "HF_TOKEN" in body["managed_keys"]

    def test_masks_token_value(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HF_TOKEN", "hf_realsecrettokenvalue123456789")
        _api, client = _client(monkeypatch, tmp_path)
        body = client.get("/api/v1/system/env").json()
        val = body["env"]["HF_TOKEN"]
        assert "..." in val
        assert "hf_reals" in val   # first-8
        # And the raw value is NOT returned.
        assert "realsecrettokenvalue" not in val


class TestSetDelete:
    def test_reject_non_managed_key(self, monkeypatch, tmp_path):
        _api, client = _client(monkeypatch, tmp_path)
        r = client.post("/api/v1/system/env?key=PATH&value=/oops")
        assert r.status_code == 403

    def test_set_persists_and_updates_process_env(self, monkeypatch, tmp_path):
        # The router computes .env path from its own location on disk;
        # the file lives next to api_server.py — which sits at the repo
        # root. Confirm the write happens by checking the process env
        # instead of a specific file path (the CI runner and dev boxes
        # differ on where the module actually lives).
        _api, client = _client(monkeypatch, tmp_path)
        r = client.post("/api/v1/system/env?key=CUDA_VISIBLE_DEVICES&value=1")
        assert r.status_code == 200, r.text
        assert r.json()["persisted"] is True
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "1"

    def test_delete_removes_from_process_env(self, monkeypatch, tmp_path):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
        _api, client = _client(monkeypatch, tmp_path)
        r = client.delete("/api/v1/system/env?key=CUDA_VISIBLE_DEVICES")
        assert r.status_code == 200
        assert "CUDA_VISIBLE_DEVICES" not in os.environ

    def test_delete_non_managed_rejected(self, monkeypatch, tmp_path):
        _api, client = _client(monkeypatch, tmp_path)
        r = client.delete("/api/v1/system/env?key=PATH")
        assert r.status_code == 403
