"""Tests for the one-off upscale API (/api/v1/edit/upscale).

Covers:
- UpscaleRequest keeps backend/preset/scale as a Pydantic model.
- _run_upscale dispatches to lanczos / realesrgan / seedvr2 by preset.
- An explicit `backend` field overrides the preset.
- When a real backend returns {ok: False}, the job ends failed with
  that error / install hint on it, not a silent lanczos fallback.
- The standalone upscale_file_realesrgan / upscale_file_seedvr2 helpers
  return {ok:False, error} on invalid input instead of raising, so an
  API thread never crashes the server.
"""
from __future__ import annotations

import os
import pytest


def _mkvideo(tmp_path):
    """A zero-byte stand-in file; the real backends are monkeypatched."""
    path = os.path.join(tmp_path, "src.mp4")
    with open(path, "wb") as f:
        f.write(b"\x00" * 16)
    return path


def _run_and_fetch_job(req, api_server, monkeypatch, tmp_path):
    """Create a job, run _run_upscale synchronously, return the job dict."""
    monkeypatch.setattr(api_server, "OUTPUT_DIR", str(tmp_path), raising=True)
    job_id = api_server._create_job("upscale", req.model_dump())
    api_server._run_upscale(job_id, req)
    return api_server.jobs[job_id]


def test_upscale_request_defaults():
    from api_server import UpscaleRequest
    req = UpscaleRequest(video_path="x.mp4")
    assert req.scale == 2
    assert req.preset == "quality_2x"
    assert req.backend == "auto"


def test_run_upscale_quality_2x_hits_realesrgan(tmp_path, monkeypatch):
    import api_server
    src = _mkvideo(tmp_path)

    seen = {}
    def fake(src_abs, out_abs, *, scale):
        seen["args"] = (src_abs, out_abs, scale)
        # Pretend the backend wrote the output.
        with open(out_abs, "wb") as f:
            f.write(b"\x00" * 2048)
        return {"ok": True, "backend": f"realesrgan_x{scale}", "scale": scale,
                "output_path": out_abs}
    monkeypatch.setattr(
        "filmmaker.agents.upscale_file_realesrgan", fake, raising=True
    )

    req = api_server.UpscaleRequest(video_path=src, preset="quality_2x")
    job = _run_and_fetch_job(req, api_server, monkeypatch, tmp_path)
    assert job["status"] == "completed", job
    assert seen["args"][2] == 2
    assert job["result"]["backend"] == "realesrgan_x2"
    assert job["result"]["scale"] == 2


def test_run_upscale_ultra_4x_passes_scale_4(tmp_path, monkeypatch):
    import api_server
    src = _mkvideo(tmp_path)

    seen = {}
    def fake(src_abs, out_abs, *, scale):
        seen["scale"] = scale
        with open(out_abs, "wb") as f:
            f.write(b"\x00" * 2048)
        return {"ok": True, "backend": f"realesrgan_x{scale}", "scale": scale,
                "output_path": out_abs}
    monkeypatch.setattr(
        "filmmaker.agents.upscale_file_realesrgan", fake, raising=True
    )

    # ultra_4x preset should imply scale 4 even if the request still
    # carries the default scale of 2.
    req = api_server.UpscaleRequest(video_path=src, preset="ultra_4x")
    job = _run_and_fetch_job(req, api_server, monkeypatch, tmp_path)
    assert job["status"] == "completed", job
    assert seen["scale"] == 4


def test_run_upscale_seedvr2_preset_hits_seedvr2(tmp_path, monkeypatch):
    import api_server
    src = _mkvideo(tmp_path)

    seen = {}
    def fake(src_abs, out_abs):
        seen["args"] = (src_abs, out_abs)
        with open(out_abs, "wb") as f:
            f.write(b"\x00" * 2048)
        return {"ok": True, "backend": "seedvr2", "variant": "3b",
                "output_path": out_abs}
    monkeypatch.setattr(
        "filmmaker.agents.upscale_file_seedvr2", fake, raising=True
    )

    req = api_server.UpscaleRequest(video_path=src, preset="seedvr2_4k")
    job = _run_and_fetch_job(req, api_server, monkeypatch, tmp_path)
    assert job["status"] == "completed", job
    assert job["result"]["backend"] == "seedvr2"
    assert job["result"]["variant"] == "3b"


def test_run_upscale_explicit_backend_overrides_preset(tmp_path, monkeypatch):
    """A client that sends backend='seedvr2' overrides a preset whose
    default backend is realesrgan."""
    import api_server
    src = _mkvideo(tmp_path)
    called = {"realesrgan": 0, "seedvr2": 0}

    def fake_r(src_abs, out_abs, *, scale):
        called["realesrgan"] += 1
        with open(out_abs, "wb") as f:
            f.write(b"\x00" * 2048)
        return {"ok": True, "backend": "realesrgan_x2", "scale": 2,
                "output_path": out_abs}
    def fake_s(src_abs, out_abs):
        called["seedvr2"] += 1
        with open(out_abs, "wb") as f:
            f.write(b"\x00" * 2048)
        return {"ok": True, "backend": "seedvr2", "variant": "3b",
                "output_path": out_abs}
    monkeypatch.setattr(
        "filmmaker.agents.upscale_file_realesrgan", fake_r, raising=True
    )
    monkeypatch.setattr(
        "filmmaker.agents.upscale_file_seedvr2", fake_s, raising=True
    )

    req = api_server.UpscaleRequest(video_path=src, preset="quality_2x",
                                     backend="seedvr2")
    job = _run_and_fetch_job(req, api_server, monkeypatch, tmp_path)
    assert job["status"] == "completed"
    assert called == {"realesrgan": 0, "seedvr2": 1}


def test_run_upscale_realesrgan_failure_surfaces_error(tmp_path, monkeypatch):
    import api_server
    src = _mkvideo(tmp_path)

    def fake(src_abs, out_abs, *, scale):
        return {"ok": False, "error": "weights unavailable: 429"}
    monkeypatch.setattr(
        "filmmaker.agents.upscale_file_realesrgan", fake, raising=True
    )

    req = api_server.UpscaleRequest(video_path=src, preset="quality_2x")
    job = _run_and_fetch_job(req, api_server, monkeypatch, tmp_path)
    assert job["status"] == "failed", job
    assert "429" in (job["error"] or "")


def test_run_upscale_seedvr2_missing_surfaces_install_hint(tmp_path, monkeypatch):
    import api_server
    src = _mkvideo(tmp_path)

    def fake(src_abs, out_abs):
        return {"ok": False, "error": "SeedVR2 not installed locally",
                "install_hint": "Drop config.json at ~/.studiolite/upscale-seedvr2/config.json"}
    monkeypatch.setattr(
        "filmmaker.agents.upscale_file_seedvr2", fake, raising=True
    )

    req = api_server.UpscaleRequest(video_path=src, preset="seedvr2_4k")
    job = _run_and_fetch_job(req, api_server, monkeypatch, tmp_path)
    assert job["status"] == "failed", job
    # The install hint should be surfaced, not a bare "not installed".
    assert "config.json" in (job["error"] or "")


def test_run_upscale_missing_source_fails_cleanly(tmp_path, monkeypatch):
    import api_server
    req = api_server.UpscaleRequest(
        video_path=os.path.join(tmp_path, "does_not_exist.mp4"),
        preset="quality_2x",
    )
    job = _run_and_fetch_job(req, api_server, monkeypatch, tmp_path)
    assert job["status"] == "failed"
    assert "not found" in (job["error"] or "").lower()


def test_upscale_file_realesrgan_rejects_invalid_scale(tmp_path):
    from filmmaker.agents import upscale_file_realesrgan
    src = _mkvideo(tmp_path)
    out = os.path.join(tmp_path, "out.mp4")
    result = upscale_file_realesrgan(src, out, scale=3)
    assert result["ok"] is False
    assert "scale" in result["error"].lower()


def test_upscale_file_realesrgan_missing_source(tmp_path):
    from filmmaker.agents import upscale_file_realesrgan
    out = os.path.join(tmp_path, "out.mp4")
    result = upscale_file_realesrgan(
        os.path.join(tmp_path, "ghost.mp4"), out, scale=2
    )
    assert result["ok"] is False
    assert "not found" in result["error"].lower()


def test_upscale_file_seedvr2_without_config_returns_install_hint(tmp_path, monkeypatch):
    """No config file -> we return {ok:False, install_hint:...} without
    reaching the subprocess. Point the config at a path we know doesn't
    exist so no personal install leaks into the test."""
    from filmmaker.agents import upscale_file_seedvr2
    src = _mkvideo(tmp_path)
    out = os.path.join(tmp_path, "out.mp4")
    missing_cfg = os.path.join(tmp_path, "no-seedvr2", "config.json")
    result = upscale_file_seedvr2(src, out, config_path=missing_cfg)
    assert result["ok"] is False
    assert "install_hint" in result
    assert "config.json" in result["install_hint"]
