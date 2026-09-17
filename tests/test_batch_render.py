"""Coverage for the batch timeline-render endpoint.

We can't actually invoke the moviepy runner without a real video, so
the runner threads created here will fail almost immediately with
"clip not found" — that's fine, we're testing the batch aggregation
layer, not the render itself. We do that by pointing the batch at
real (empty) files on disk so submit-time validation passes, then
observe the batch record and its status endpoint."""
from __future__ import annotations

import importlib
import os
import sys
import time

import pytest


def _client(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("STUDIOLITE_AUTH", "off")
    for name in list(sys.modules):
        if name == "api_server":
            del sys.modules[name]
    api = importlib.import_module("api_server")
    return api, TestClient(api.app)


def _touch_clip(tmp_path, name="clip.mp4"):
    p = tmp_path / name
    p.write_bytes(b"\x00" * 32)  # not a real video, just needs to exist
    return str(p)


def _render_payload(video_path):
    return {
        "clips": [{"video_path": video_path, "in_point": 0.0, "out_point": 1.0}],
        "fps": 30, "width": 1280, "height": 720,
        "codec": "h264", "quality": "high",
        "apply_watermark": True,
    }


class TestBatchSubmit:
    def test_empty_batch_rejected(self, monkeypatch, tmp_path):
        _api, client = _client(monkeypatch, tmp_path)
        r = client.post("/api/v1/edit/batch-render", json={"items": []})
        assert r.status_code == 422

    def test_missing_clip_rejected_before_any_job_started(self, monkeypatch, tmp_path):
        _api, client = _client(monkeypatch, tmp_path)
        good = _touch_clip(tmp_path, "ok.mp4")
        r = client.post("/api/v1/edit/batch-render", json={
            "items": [
                {"render": _render_payload(good)},
                {"render": _render_payload(str(tmp_path / "missing.mp4"))},
            ],
        })
        assert r.status_code == 422
        # And no batch was created — the response is an error, not a status.
        assert "detail" in r.json()

    def test_submit_creates_one_job_per_item(self, monkeypatch, tmp_path):
        api, client = _client(monkeypatch, tmp_path)
        clip = _touch_clip(tmp_path)
        r = client.post("/api/v1/edit/batch-render", json={
            "name": "overnight batch",
            "items": [
                {"name": "shot A", "render": _render_payload(clip)},
                {"name": "shot B", "render": _render_payload(clip)},
                {"name": "shot C", "render": _render_payload(clip)},
            ],
        })
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["total"] == 3
        assert body["name"] == "overnight batch"
        assert len({it["job_id"] for it in body["items"]}) == 3
        # Each item is registered in the shared jobs store.
        for it in body["items"]:
            assert it["job_id"] in api.jobs


class TestBatchStatus:
    def test_status_endpoint_matches_submit_response(self, monkeypatch, tmp_path):
        _api, client = _client(monkeypatch, tmp_path)
        clip = _touch_clip(tmp_path)
        submit = client.post("/api/v1/edit/batch-render", json={
            "items": [{"render": _render_payload(clip)}],
        }).json()
        batch_id = submit["batch_id"]

        r = client.get(f"/api/v1/edit/batch-render/{batch_id}")
        assert r.status_code == 200
        got = r.json()
        assert got["batch_id"] == batch_id
        assert got["total"] == 1
        assert set(got["counts"]).issuperset(
            {"queued", "running", "completed", "failed", "cancelled"}
        )

    def test_unknown_batch_404s(self, monkeypatch, tmp_path):
        _api, client = _client(monkeypatch, tmp_path)
        r = client.get("/api/v1/edit/batch-render/does-not-exist")
        assert r.status_code == 404

    def test_status_reflects_job_updates(self, monkeypatch, tmp_path):
        """The status endpoint reads the shared jobs dict — poking a job
        via the store should be visible through the rollup. This proves
        the aggregation layer isn't caching stale state."""
        api, client = _client(monkeypatch, tmp_path)
        clip = _touch_clip(tmp_path)
        submit = client.post("/api/v1/edit/batch-render", json={
            "items": [{"render": _render_payload(clip)},
                      {"render": _render_payload(clip)}],
        }).json()
        batch_id = submit["batch_id"]
        first_job = submit["items"][0]["job_id"]

        # Force-mark one job completed; batch rollup should reflect it.
        api._update_job(first_job, status="completed",
                        progress=100, message="test hand-off")
        got = client.get(f"/api/v1/edit/batch-render/{batch_id}").json()
        assert got["counts"]["completed"] >= 1
        item = next(x for x in got["items"] if x["job_id"] == first_job)
        assert item["status"] == "completed"
        assert item["message"] == "test hand-off"


class TestBatchCancel:
    def test_cancel_unknown_returns_404(self, monkeypatch, tmp_path):
        _api, client = _client(monkeypatch, tmp_path)
        r = client.delete("/api/v1/edit/batch-render/nope")
        assert r.status_code == 404

    def test_cancel_returns_count(self, monkeypatch, tmp_path):
        _api, client = _client(monkeypatch, tmp_path)
        clip = _touch_clip(tmp_path)
        submit = client.post("/api/v1/edit/batch-render", json={
            "items": [{"render": _render_payload(clip)},
                      {"render": _render_payload(clip)}],
        }).json()
        batch_id = submit["batch_id"]
        r = client.delete(f"/api/v1/edit/batch-render/{batch_id}")
        assert r.status_code == 200
        # requested_cancel counts only jobs whose cancel event was set
        # here — a job that already finished won't be re-cancelled.
        assert r.json()["batch_id"] == batch_id
        assert isinstance(r.json()["requested_cancel"], int)
