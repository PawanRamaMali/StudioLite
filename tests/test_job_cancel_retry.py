"""Coverage for the universal cancel / retry surface added on top of the
job registry."""
from __future__ import annotations

import importlib
import sys
import threading
import time

import pytest


def _fresh_api(monkeypatch, tmp_path, *, auth: str = "off"):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("STUDIOLITE_AUTH", auth)
    for name in list(sys.modules):
        if name == "api_server" or name.startswith("api_server."):
            del sys.modules[name]
    api = importlib.import_module("api_server")
    from fastapi.testclient import TestClient
    return api, TestClient(api.app)


def _slow_runner(api, job_id, params):
    """Test runner: idles in a tight loop, honoring _should_cancel."""
    api._update_job(job_id, status="running", message="working")
    for i in range(200):
        if api._should_cancel(job_id):
            api._update_job(job_id, status="cancelled", message="stopped")
            return
        time.sleep(0.01)
    api._update_job(job_id, status="completed", message="done", progress=1.0)


class TestCancel:
    def test_cancel_flips_running_to_cancelling(self, monkeypatch, tmp_path):
        api, client = _fresh_api(monkeypatch, tmp_path)
        api.jobs.clear()
        api._cancel_events.clear()
        job_id = api._create_job("test_slow")
        t = threading.Thread(target=_slow_runner, args=(api, job_id, {}), daemon=True)
        t.start()
        # give the runner a beat to hit the loop
        time.sleep(0.05)
        r = client.post(f"/api/v1/jobs/{job_id}/cancel")
        assert r.status_code == 200
        t.join(timeout=2)
        with api._jobs_lock:
            job = api.jobs[job_id]
        assert job["status"] == "cancelled"

    def test_cancel_of_finished_job_is_noop(self, monkeypatch, tmp_path):
        api, client = _fresh_api(monkeypatch, tmp_path)
        api.jobs.clear()
        api._cancel_events.clear()
        job_id = api._create_job("done_kind")
        api._update_job(job_id, status="completed", progress=1.0)
        r = client.post(f"/api/v1/jobs/{job_id}/cancel")
        assert r.status_code == 200
        assert r.json()["status"] == "completed"

    def test_cancel_missing_job_returns_404(self, monkeypatch, tmp_path):
        api, client = _fresh_api(monkeypatch, tmp_path)
        r = client.post("/api/v1/jobs/does-not-exist/cancel")
        assert r.status_code == 404


class TestRetry:
    def test_retry_unregistered_kind_returns_400(self, monkeypatch, tmp_path):
        api, client = _fresh_api(monkeypatch, tmp_path)
        api.jobs.clear()
        job_id = api._create_job("test_slow")
        api._update_job(job_id, status="failed", error="boom")
        r = client.post(f"/api/v1/jobs/{job_id}/retry")
        assert r.status_code == 400

    def test_retry_still_running_returns_400(self, monkeypatch, tmp_path):
        api, client = _fresh_api(monkeypatch, tmp_path)
        api.jobs.clear()
        job_id = api._create_job("test_slow")
        api._update_job(job_id, status="running")
        r = client.post(f"/api/v1/jobs/{job_id}/retry")
        assert r.status_code == 400

    def test_retry_registered_kind_starts_new_job(self, monkeypatch, tmp_path):
        api, client = _fresh_api(monkeypatch, tmp_path)
        api.jobs.clear()
        api._cancel_events.clear()

        # Register a runner that instantly succeeds so the retry test
        # doesn't have to wait on real work.
        def _instant(job_id, params):
            api._update_job(job_id, status="completed", progress=1.0)

        api.JOB_KIND_RUNNERS["instant"] = _instant

        job_id = api._create_job("instant", {"seed": 42})
        api._update_job(job_id, status="failed")
        r = client.post(f"/api/v1/jobs/{job_id}/retry")
        assert r.status_code == 200
        body = r.json()
        assert body["new_job_id"] != job_id
        # Give the runner a beat to run
        time.sleep(0.05)
        with api._jobs_lock:
            new_job = api.jobs[body["new_job_id"]]
        assert new_job["status"] == "completed"
        assert new_job["params"]["seed"] == 42


class TestSubprocessTracking:
    def test_register_and_clear(self, monkeypatch, tmp_path):
        api, _ = _fresh_api(monkeypatch, tmp_path)

        class _FakeProc:
            def __init__(self):
                self._alive = True
            def poll(self):
                return None if self._alive else 0
            def terminate(self):
                self._alive = False

        p1, p2 = _FakeProc(), _FakeProc()
        api._register_subprocess("j1", p1)
        api._register_subprocess("j1", p2)
        killed = api._terminate_job_subprocesses("j1")
        assert killed == 2
        assert not p1._alive and not p2._alive
        api._clear_subprocesses("j1")
        assert "j1" not in api._job_subprocesses

    def test_terminate_ignores_already_finished(self, monkeypatch, tmp_path):
        api, _ = _fresh_api(monkeypatch, tmp_path)

        class _FinishedProc:
            def poll(self):
                return 0  # already exited
            def terminate(self):  # pragma: no cover - should never be called
                raise AssertionError("should not terminate a finished proc")

        api._register_subprocess("j2", _FinishedProc())
        killed = api._terminate_job_subprocesses("j2")
        assert killed == 0
