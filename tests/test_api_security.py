"""Integration tests for the FastAPI security layer.

Each test spins up a fresh isolated api_server import with STUDIOLITE_AUTH
in a known state and a temp .auth file. This is verbose but the
alternative - an import-time singleton - makes it too easy for one test
to leak state into another."""
from __future__ import annotations

import importlib
import io
import os
import sys

import pytest


def _fresh_api(monkeypatch, tmp_path, *, auth: str = "on"):
    """Reload api_server against a scratch cwd/env. Returns (module, client)."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("STUDIOLITE_AUTH", auth)
    # Wipe any cached copy so the module reads our env fresh.
    for name in list(sys.modules):
        if name == "api_server" or name.startswith("api_server."):
            del sys.modules[name]
    api = importlib.import_module("api_server")
    from fastapi.testclient import TestClient
    return api, TestClient(api.app)


class TestAuthMiddleware:
    def test_auth_status_is_public(self, monkeypatch, tmp_path):
        api, client = _fresh_api(monkeypatch, tmp_path, auth="on")
        r = client.get("/api/v1/system/auth-status")
        assert r.status_code == 200
        body = r.json()
        assert body["auth_enabled"] is True
        assert body["header"] == "X-StudioLite-Token"

    def test_health_is_public(self, monkeypatch, tmp_path):
        api, client = _fresh_api(monkeypatch, tmp_path, auth="on")
        r = client.get("/api/v1/health")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}

    def test_write_endpoint_401_without_token(self, monkeypatch, tmp_path):
        api, client = _fresh_api(monkeypatch, tmp_path, auth="on")
        r = client.post(
            "/api/v1/system/env",
            params={"key": "HF_TOKEN", "value": "test"},
        )
        assert r.status_code == 401

    def test_write_endpoint_accepts_valid_token(self, monkeypatch, tmp_path):
        api, client = _fresh_api(monkeypatch, tmp_path, auth="on")
        # Load the token the module minted into .auth
        assert os.path.exists(api.AUTH_FILE)
        token = open(api.AUTH_FILE, encoding="utf-8").read().strip()
        r = client.post(
            "/api/v1/system/env",
            params={"key": "HF_TOKEN", "value": "hf_dummy"},
            headers={"X-StudioLite-Token": token},
        )
        assert r.status_code == 200

    def test_env_key_off_allowlist_is_403_even_with_token(self, monkeypatch, tmp_path):
        api, client = _fresh_api(monkeypatch, tmp_path, auth="on")
        token = open(api.AUTH_FILE, encoding="utf-8").read().strip()
        r = client.post(
            "/api/v1/system/env",
            params={"key": "PATH", "value": "/etc"},
            headers={"X-StudioLite-Token": token},
        )
        assert r.status_code == 403

    def test_auth_off_bypasses_middleware(self, monkeypatch, tmp_path):
        api, client = _fresh_api(monkeypatch, tmp_path, auth="off")
        r = client.post(
            "/api/v1/system/env",
            params={"key": "HF_TOKEN", "value": "hf_dummy"},
        )
        # With auth off the middleware never intervenes; the request
        # succeeds (or fails on downstream validation, not on auth).
        assert r.status_code != 401


class TestWebsocketAuth:
    """The HTTP middleware never sees ws handshakes, so each ws handler
    checks `?token=` itself. The film stream route is the probe: with auth
    passed, an unknown project is accepted and then closed with 4404."""

    _URL = "/api/v1/films/no-such-project/stream"

    def _close_code(self, client, url):
        from starlette.websockets import WebSocketDisconnect
        with pytest.raises(WebSocketDisconnect) as exc:
            with client.websocket_connect(url) as ws:
                ws.receive_json()
        return exc.value.code

    def test_ws_rejected_without_token(self, monkeypatch, tmp_path):
        api, client = _fresh_api(monkeypatch, tmp_path, auth="on")
        assert self._close_code(client, self._URL) == 1008

    def test_ws_rejected_with_wrong_token(self, monkeypatch, tmp_path):
        api, client = _fresh_api(monkeypatch, tmp_path, auth="on")
        assert self._close_code(client, self._URL + "?token=nope") == 1008

    def test_ws_accepts_valid_token(self, monkeypatch, tmp_path):
        api, client = _fresh_api(monkeypatch, tmp_path, auth="on")
        token = open(api.AUTH_FILE, encoding="utf-8").read().strip()
        assert self._close_code(client, f"{self._URL}?token={token}") == 4404

    def test_ws_open_when_auth_off(self, monkeypatch, tmp_path):
        api, client = _fresh_api(monkeypatch, tmp_path, auth="off")
        assert self._close_code(client, self._URL) == 4404

    @pytest.mark.parametrize("path", ["/api/v1/transcribe/live", "/api/v1/screen/live"])
    def test_live_ws_routes_reject_without_token(self, monkeypatch, tmp_path, path):
        api, client = _fresh_api(monkeypatch, tmp_path, auth="on")
        assert self._close_code(client, path) == 1008


class TestUploadHardening:
    def test_upload_rejects_wrong_extension(self, monkeypatch, tmp_path):
        api, client = _fresh_api(monkeypatch, tmp_path, auth="off")
        r = client.post(
            "/api/v1/edit/upload",
            files={"file": ("mine.txt", b"hello", "text/plain")},
        )
        assert r.status_code == 415

    def test_upload_rejects_wrong_magic_bytes(self, monkeypatch, tmp_path):
        api, client = _fresh_api(monkeypatch, tmp_path, auth="off")
        # File claims .png but the content is plain text - magic-byte
        # check should reject it and delete the partial write.
        r = client.post(
            "/api/v1/images/upload",
            files={"file": ("hack.png", b"NOT AN IMAGE", "image/png")},
        )
        assert r.status_code == 415

    def test_sanitize_filename_strips_traversal(self, monkeypatch, tmp_path):
        api, _ = _fresh_api(monkeypatch, tmp_path, auth="off")
        safe = api._sanitize_filename("../../etc/passwd")
        assert ".." not in safe
        assert "/" not in safe and "\\" not in safe

    def test_sanitize_filename_drops_nulls_and_controls(self, monkeypatch, tmp_path):
        api, _ = _fresh_api(monkeypatch, tmp_path, auth="off")
        safe = api._sanitize_filename("hello\x00world.png")
        assert "\x00" not in safe

    def test_sanitize_filename_falls_back_to_default(self, monkeypatch, tmp_path):
        api, _ = _fresh_api(monkeypatch, tmp_path, auth="off")
        # Every char stripped → default applies.
        safe = api._sanitize_filename("!!!!!!")
        assert safe  # non-empty
        assert "!" not in safe


class TestListJobs:
    def test_empty_jobs_returns_shape(self, monkeypatch, tmp_path):
        api, client = _fresh_api(monkeypatch, tmp_path, auth="off")
        # jobs dict is process-global; wipe any leftover from earlier tests
        api.jobs.clear()
        r = client.get("/api/v1/jobs")
        assert r.status_code == 200
        body = r.json()
        assert body["jobs"] == []
        assert body["total"] == 0
