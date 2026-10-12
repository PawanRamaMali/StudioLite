"""Tests for the cloud-burst dispatcher.

Everything is driven through an injected transport so none of these tests
touch the network. The transport is a tiny stub that returns scripted
replies in sequence."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import pytest

from filmmaker import cloud_burst as cb


def _reply(status: int, body: bytes = b"", headers: Optional[Dict[str, str]] = None):
    return cb.TransportReply(status=status, body=body,
                              headers=headers or {})


class _Stub:
    """Return a scripted list of replies, in order, per URL match."""

    def __init__(self, script: List):
        # Each entry is (method, url_contains, reply).
        self.script = list(script)
        self.calls: List = []

    def __call__(self, method, url, headers, body):
        self.calls.append((method, url, dict(headers), body))
        for i, (m, u, r) in enumerate(self.script):
            if m == method and u in url:
                self.script.pop(i)
                return r
        return _reply(599, b"no scripted reply")


# ---------------------------------------------------------------------------
# Dry-run and smoke tests of the dispatch entry point

def test_dry_run_writes_marker_without_network(tmp_path: Path, monkeypatch):
    """STUDIOLITE_CLOUD_DRY_RUN=1 routes to the dry-run provider."""
    monkeypatch.setenv("STUDIOLITE_CLOUD_DRY_RUN", "1")
    monkeypatch.delenv("STUDIOLITE_CLOUD_PROVIDER", raising=False)
    out = tmp_path / "burst.txt"
    req = cb.BurstRequest(stage="upscale", backend_name="seedvr2",
                           inputs={"video": "in.mp4"}, output_path=str(out),
                           deadline_seconds=10)
    # Transport MUST not be called in dry-run mode.
    def _should_not_be_called(*args, **kwargs):
        raise AssertionError("transport called in dry-run mode")
    result = cb.dispatch(req, transport=_should_not_be_called)
    assert result.rendered is True
    assert result.provider == "dry-run"
    assert out.exists()
    assert "stage: upscale" in out.read_text(encoding="utf-8")


def test_no_provider_configured_returns_no_render(tmp_path: Path, monkeypatch):
    """No env, no override, no configured provider: dispatcher reports
    'no cloud provider configured' - never raises."""
    monkeypatch.delenv("STUDIOLITE_CLOUD_PROVIDER", raising=False)
    monkeypatch.delenv("STUDIOLITE_CLOUD_DRY_RUN", raising=False)
    monkeypatch.delenv("REPLICATE_API_TOKEN", raising=False)
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    req = cb.BurstRequest(stage="x", backend_name="y", inputs={},
                           output_path=str(tmp_path / "nope.bin"))
    result = cb.dispatch(req)
    assert result.rendered is False
    assert "no cloud provider configured" in result.reason


def test_unknown_provider_is_a_soft_error(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("STUDIOLITE_CLOUD_PROVIDER", "pigeon")
    req = cb.BurstRequest(stage="x", backend_name="y", inputs={},
                           output_path=str(tmp_path / "nope.bin"))
    result = cb.dispatch(req)
    assert result.rendered is False
    assert "unknown provider" in result.reason


def test_provider_exception_is_wrapped_not_raised(tmp_path: Path, monkeypatch):
    """A buggy provider must not take the pipeline down."""
    class Boom:
        name = "boom"
        def is_configured(self): return True
        def dispatch(self, request, transport):
            raise RuntimeError("oops")
    cb.register_provider("boom", Boom())
    try:
        monkeypatch.setenv("STUDIOLITE_CLOUD_PROVIDER", "boom")
        req = cb.BurstRequest(stage="x", backend_name="y", inputs={},
                               output_path=str(tmp_path / "nope.bin"))
        result = cb.dispatch(req)
        assert result.rendered is False
        assert "provider raised" in result.reason
    finally:
        # Clean up the test-only provider so later tests see the defaults.
        cb._PROVIDERS.pop("boom", None)


# ---------------------------------------------------------------------------
# Replicate happy path and failure modes

def test_replicate_success(tmp_path: Path, monkeypatch):
    """POST creates a prediction; one poll returns succeeded; output URL
    download writes the artifact to disk."""
    monkeypatch.setenv("REPLICATE_API_TOKEN", "test-token")
    monkeypatch.setenv("REPLICATE_SLUG_SEEDVR2", "owner/model:v1")
    monkeypatch.setenv("STUDIOLITE_CLOUD_PROVIDER", "replicate")
    out = tmp_path / "out.mp4"
    payload_body = b"\x00\x01\x02rendered-bytes"

    stub = _Stub([
        ("POST", "api.replicate.com/v1/predictions",
         _reply(201, json.dumps({"urls": {"get": "https://api.replicate.com/v1/p/abc"},
                                 "status": "starting"}).encode("utf-8"))),
        ("GET", "/v1/p/abc",
         _reply(200, json.dumps({"status": "succeeded",
                                 "output": "https://cdn.replicate.com/out.mp4"}).encode("utf-8"))),
        ("GET", "cdn.replicate.com/out.mp4", _reply(200, payload_body)),
    ])

    # Short deadline to make the test fast; the stub never actually sleeps.
    monkeypatch.setattr(cb.time, "sleep", lambda _s: None)

    req = cb.BurstRequest(stage="upscale", backend_name="seedvr2",
                           inputs={"video_url": "https://example/in.mp4"},
                           output_path=str(out), deadline_seconds=10)
    result = cb.dispatch(req, transport=stub)
    assert result.rendered is True, result.reason
    assert result.provider == "replicate"
    assert out.read_bytes() == payload_body
    # First call carries the token and the slug.
    first_method, first_url, first_headers, first_body = stub.calls[0]
    assert first_headers["Authorization"] == "Bearer test-token"
    assert b"owner/model:v1" in (first_body or b"")


def test_replicate_missing_slug(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("REPLICATE_API_TOKEN", "test-token")
    monkeypatch.delenv("REPLICATE_SLUG_SEEDVR2", raising=False)
    monkeypatch.setenv("STUDIOLITE_CLOUD_PROVIDER", "replicate")
    # The provider reads env at construction time; rebuild it so this test
    # sees the deleted var.
    cb._PROVIDERS["replicate"] = cb.ReplicateProvider()
    req = cb.BurstRequest(stage="upscale", backend_name="seedvr2",
                           inputs={}, output_path=str(tmp_path / "x.mp4"))
    result = cb.dispatch(req, transport=_Stub([]))
    assert result.rendered is False
    assert "REPLICATE_SLUG_SEEDVR2" in result.reason


def test_replicate_failed_status(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("REPLICATE_API_TOKEN", "test-token")
    monkeypatch.setenv("REPLICATE_SLUG_SEEDVR2", "owner/model:v1")
    monkeypatch.setenv("STUDIOLITE_CLOUD_PROVIDER", "replicate")
    cb._PROVIDERS["replicate"] = cb.ReplicateProvider()
    monkeypatch.setattr(cb.time, "sleep", lambda _s: None)

    stub = _Stub([
        ("POST", "api.replicate.com/v1/predictions",
         _reply(201, json.dumps({"urls": {"get": "https://api/p/abc"},
                                 "status": "starting"}).encode("utf-8"))),
        ("GET", "/p/abc",
         _reply(200, json.dumps({"status": "failed",
                                 "error": "nvidia smoked"}).encode("utf-8"))),
    ])
    req = cb.BurstRequest(stage="upscale", backend_name="seedvr2",
                           inputs={}, output_path=str(tmp_path / "x.mp4"),
                           deadline_seconds=10)
    result = cb.dispatch(req, transport=stub)
    assert result.rendered is False
    assert "failed" in result.reason


def test_replicate_deadline_exceeded(tmp_path: Path, monkeypatch):
    """Status never resolves inside the deadline -> rendered=False."""
    monkeypatch.setenv("REPLICATE_API_TOKEN", "test-token")
    monkeypatch.setenv("REPLICATE_SLUG_SEEDVR2", "owner/model:v1")
    monkeypatch.setenv("STUDIOLITE_CLOUD_PROVIDER", "replicate")
    cb._PROVIDERS["replicate"] = cb.ReplicateProvider()

    # Fast-forward time inside the poll loop so the deadline trips without
    # the test actually waiting.
    virtual_now = [0.0]
    monkeypatch.setattr(cb.time, "monotonic", lambda: virtual_now[0])
    def _tick(_s):
        virtual_now[0] += 10
    monkeypatch.setattr(cb.time, "sleep", _tick)

    stub = _Stub([
        ("POST", "api.replicate.com/v1/predictions",
         _reply(201, json.dumps({"urls": {"get": "https://api/p/abc"},
                                 "status": "starting"}).encode("utf-8"))),
        # Replies forever: whenever polled, still processing.
    ])
    # Replace scripted list with an on-demand responder for the GET.
    original_call = stub.__call__
    def _call(method, url, headers, body):
        if method == "GET":
            return _reply(200, json.dumps({"status": "processing"}).encode("utf-8"))
        return original_call(method, url, headers, body)
    stub.__call__ = _call  # type: ignore[assignment]

    req = cb.BurstRequest(stage="upscale", backend_name="seedvr2",
                           inputs={}, output_path=str(tmp_path / "x.mp4"),
                           deadline_seconds=5)
    result = cb.dispatch(req, transport=stub)
    assert result.rendered is False
    assert "deadline" in result.reason


# ---------------------------------------------------------------------------
# RunPod happy path (async branch - deadline > 300s)

def test_runpod_async_success(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("RUNPOD_API_KEY", "test-key")
    monkeypatch.setenv("RUNPOD_ENDPOINT_SEEDVR2", "endpoint-abc")
    monkeypatch.setenv("STUDIOLITE_CLOUD_PROVIDER", "runpod")
    cb._PROVIDERS["runpod"] = cb.RunPodProvider()
    monkeypatch.setattr(cb.time, "sleep", lambda _s: None)

    out = tmp_path / "out.wav"
    body = b"rendered-audio"

    stub = _Stub([
        ("POST", "runpod.ai/v2/endpoint-abc/run",
         _reply(200, json.dumps({"id": "job-xyz", "status": "IN_QUEUE"}).encode("utf-8"))),
        ("GET", "/status/job-xyz",
         _reply(200, json.dumps({"status": "COMPLETED",
                                 "output": {"url": "https://cdn/out.wav"}}).encode("utf-8"))),
        ("GET", "cdn/out.wav", _reply(200, body)),
    ])
    req = cb.BurstRequest(stage="composer", backend_name="seedvr2",
                           inputs={"audio_url": "x"},
                           output_path=str(out), deadline_seconds=600)
    result = cb.dispatch(req, transport=stub)
    assert result.rendered is True, result.reason
    assert result.provider == "runpod"
    assert out.read_bytes() == body


# ---------------------------------------------------------------------------
# providers_configured reports env state honestly

def test_providers_configured_tracks_env(monkeypatch):
    monkeypatch.delenv("REPLICATE_API_TOKEN", raising=False)
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    cb._PROVIDERS["replicate"] = cb.ReplicateProvider()
    cb._PROVIDERS["runpod"] = cb.RunPodProvider()
    reported = cb.providers_configured()
    assert reported["replicate"] is False
    assert reported["runpod"] is False
    assert reported["dry-run"] is True
    monkeypatch.setenv("REPLICATE_API_TOKEN", "t")
    cb._PROVIDERS["replicate"] = cb.ReplicateProvider()
    assert cb.providers_configured()["replicate"] is True
