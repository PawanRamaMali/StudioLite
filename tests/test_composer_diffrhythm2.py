"""Tests for the DiffRhythm 2 composer branch.

DiffRhythm 2 is GPU-only in practice. These tests exercise the three
resolution paths of the helper function - local install, cloud burst, and
skip-with-install-hint - without touching the network or needing any
model weights.

We hit _render_diffrhythm2 directly rather than through run_composer so
the tests do not need to stage producer/editor artifacts and an LLM
response for the overall composer stage."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest


def _mkproj(tmp: Path, music_backend: str = "diffrhythm2"):
    from filmmaker import projects
    return projects.create(
        str(tmp),
        brief="A keeper opens a letter.",
        title="Beacon",
        config_dict={
            "llm_backend": "ollama", "llm_model": "mock",
            "style": "stylized", "target_minutes": 1.0,
            "music_backend": music_backend,
        },
    )


def test_diffrhythm2_without_local_or_cloud_returns_install_hint(tmp_path: Path, monkeypatch):
    """No config.json, no cloud provider -> helper returns (False, reason,
    extra with install_hint). The caller writes silence."""
    monkeypatch.delenv("STUDIOLITE_CLOUD_PROVIDER", raising=False)
    monkeypatch.delenv("STUDIOLITE_CLOUD_DRY_RUN", raising=False)
    monkeypatch.delenv("REPLICATE_API_TOKEN", raising=False)
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)

    proj = _mkproj(tmp_path)
    out = os.path.join(proj.artifacts_dir, "score.wav")
    from filmmaker import agents
    ok, err, extra = agents._render_diffrhythm2(
        proj, "cinematic drama 80 bpm", out, duration_sec=30,
    )
    assert ok is False
    assert err
    assert "install_hint" in extra
    assert "REPLICATE_SLUG_DIFFRHYTHM2" in extra["install_hint"]


def test_diffrhythm2_cloud_burst_dry_run_writes_output(tmp_path: Path, monkeypatch):
    """STUDIOLITE_CLOUD_DRY_RUN=1 routes to the dry-run provider; the
    helper reports success and the output file lands on disk."""
    monkeypatch.setenv("STUDIOLITE_CLOUD_DRY_RUN", "1")

    proj = _mkproj(tmp_path)
    out = os.path.join(proj.artifacts_dir, "score.wav")
    from filmmaker import agents
    ok, err, extra = agents._render_diffrhythm2(
        proj, "cinematic drama 80 bpm", out, duration_sec=30,
    )
    assert ok is True, err
    assert err is None
    assert extra.get("mode") == "cloud"
    assert extra.get("provider") == "dry-run"
    assert os.path.exists(out)


def test_diffrhythm2_local_runs_subprocess_on_valid_config(tmp_path: Path, monkeypatch):
    """Config pointing at real paths and a stubbed subprocess that writes
    a plausible wav -> helper reports mode=local success."""
    proj = _mkproj(tmp_path)

    from filmmaker import agents
    cfg_path = agents._diffrhythm2_config_path(proj)
    repo_dir = tmp_path / "diffrhythm2-repo"
    repo_dir.mkdir()
    venv_py = repo_dir / "python.exe"
    venv_py.write_bytes(b"stub")
    weights = repo_dir / "ckpt.safetensors"
    weights.write_bytes(b"stub")

    os.makedirs(os.path.dirname(cfg_path), exist_ok=True)
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump({
            "repo_dir": str(repo_dir), "venv_python": str(venv_py),
            "model_path": str(weights),
        }, f)

    import subprocess as _sp
    def fake_run(cmd, cwd=None, capture_output=None, text=None, check=None,
                  timeout=None):
        for i, token in enumerate(cmd):
            if token == "--output" and i + 1 < len(cmd):
                with open(cmd[i + 1], "wb") as f:
                    f.write(b"x" * 2048)
        return _sp.CompletedProcess(args=cmd, returncode=0,
                                      stdout="rendered 30s\n", stderr="")

    monkeypatch.setattr(agents.subprocess, "run", fake_run)
    out = os.path.join(proj.artifacts_dir, "score.wav")
    ok, err, extra = agents._render_diffrhythm2(
        proj, "cinematic drama", out, duration_sec=30,
    )
    assert ok is True, err
    assert err is None
    assert extra.get("mode") == "local"
    assert os.path.exists(out)


def test_diffrhythm2_local_subprocess_failure_falls_through(tmp_path: Path, monkeypatch):
    """When the subprocess exits non-zero and no cloud is configured, the
    helper falls through to the skip-with-install-hint path."""
    monkeypatch.delenv("STUDIOLITE_CLOUD_PROVIDER", raising=False)
    monkeypatch.delenv("STUDIOLITE_CLOUD_DRY_RUN", raising=False)
    monkeypatch.delenv("REPLICATE_API_TOKEN", raising=False)
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)

    proj = _mkproj(tmp_path)
    from filmmaker import agents
    cfg_path = agents._diffrhythm2_config_path(proj)
    repo_dir = tmp_path / "diffrhythm2-repo"
    repo_dir.mkdir()
    venv_py = repo_dir / "python.exe"
    venv_py.write_bytes(b"stub")
    weights = repo_dir / "ckpt.safetensors"
    weights.write_bytes(b"stub")
    os.makedirs(os.path.dirname(cfg_path), exist_ok=True)
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump({
            "repo_dir": str(repo_dir), "venv_python": str(venv_py),
            "model_path": str(weights),
        }, f)

    import subprocess as _sp
    def fake_run(cmd, **kwargs):
        return _sp.CompletedProcess(args=cmd, returncode=2,
                                      stdout="", stderr="oom")
    monkeypatch.setattr(agents.subprocess, "run", fake_run)

    out = os.path.join(proj.artifacts_dir, "score.wav")
    ok, err, extra = agents._render_diffrhythm2(
        proj, "cinematic drama", out, duration_sec=30,
    )
    assert ok is False
    assert err
    assert "install_hint" in extra


def test_backend_label_lowercased_in_registry():
    """Smoke that the composer backend name used here is the one the
    registry already knows about. Guards against typos drifting the two
    sources of truth apart."""
    from filmmaker import backend_registry as br
    assert br.get("composer", "diffrhythm2") is not None
