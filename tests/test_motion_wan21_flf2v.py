"""Tests for the Wan2.1-FLF2V transition helper.

Wan2.1-Fun-14B-InP via city96's GGUF ladder is the roadmap transition
backend (6.02 GB Q2_K up to 32.8 GB bf16). ``_render_transition_flf2v``
takes two keyframes and produces a short morph clip. These tests
exercise the three resolution paths without a GPU or the network."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest


def _mkproj(tmp: Path):
    from filmmaker import projects
    return projects.create(
        str(tmp),
        brief="A keeper opens a letter.",
        title="Beacon",
        config_dict={
            "llm_backend": "ollama", "llm_model": "mock",
            "style": "stylized", "target_minutes": 1.0,
        },
    )


def _two_frames(proj) -> tuple[str, str, str]:
    a = os.path.join(proj.dir, "frame_a.png")
    b = os.path.join(proj.dir, "frame_b.png")
    out = os.path.join(proj.dir, "transition.mp4")
    for p in (a, b):
        with open(p, "wb") as f:
            f.write(b"stub-image")
    return a, b, out


def test_flf2v_without_local_or_cloud_returns_install_hint(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("STUDIOLITE_CLOUD_PROVIDER", raising=False)
    monkeypatch.delenv("STUDIOLITE_CLOUD_DRY_RUN", raising=False)
    monkeypatch.delenv("REPLICATE_API_TOKEN", raising=False)
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)

    proj = _mkproj(tmp_path)
    a, b, out = _two_frames(proj)
    from filmmaker import agents

    r = agents._render_transition_flf2v(proj, a, b, out, duration_sec=2)
    assert r["backend"] == "wan21_flf2v"
    assert r["mode"] == "skipped"
    assert "install_hint" in r
    assert "REPLICATE_SLUG_WAN21_FLF2V" in r["install_hint"]


def test_flf2v_cloud_dry_run_writes_output(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("STUDIOLITE_CLOUD_DRY_RUN", "1")

    proj = _mkproj(tmp_path)
    a, b, out = _two_frames(proj)
    from filmmaker import agents

    r = agents._render_transition_flf2v(proj, a, b, out, duration_sec=2)
    assert r["backend"] == "wan21_flf2v"
    assert r["mode"] == "cloud"
    assert r["provider"] == "dry-run"
    assert os.path.exists(r["output_path"])


def test_flf2v_local_config_missing_paths_falls_through(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("STUDIOLITE_CLOUD_PROVIDER", raising=False)
    monkeypatch.delenv("STUDIOLITE_CLOUD_DRY_RUN", raising=False)
    monkeypatch.delenv("REPLICATE_API_TOKEN", raising=False)
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)

    proj = _mkproj(tmp_path)
    a, b, out = _two_frames(proj)
    from filmmaker import agents

    cfg_path = agents._flf2v_config_path(proj)
    os.makedirs(os.path.dirname(cfg_path), exist_ok=True)
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump({
            "repo_dir": str(tmp_path / "no-such-dir"),
            "venv_python": str(tmp_path / "nope.exe"),
            "model_path": str(tmp_path / "weights.gguf"),
        }, f)

    r = agents._render_transition_flf2v(proj, a, b, out)
    assert r["mode"] == "skipped"
    assert "install_hint" in r


def test_flf2v_local_runs_subprocess_on_valid_config(tmp_path: Path, monkeypatch):
    proj = _mkproj(tmp_path)
    a, b, out = _two_frames(proj)

    from filmmaker import agents
    cfg_path = agents._flf2v_config_path(proj)
    repo_dir = tmp_path / "flf2v-repo"
    repo_dir.mkdir()
    venv_py = repo_dir / "python.exe"
    venv_py.write_bytes(b"stub")
    weights = repo_dir / "weights.gguf"
    weights.write_bytes(b"stub")

    os.makedirs(os.path.dirname(cfg_path), exist_ok=True)
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump({
            "repo_dir": str(repo_dir),
            "venv_python": str(venv_py),
            "model_path": str(weights),
        }, f)

    import subprocess as _sp

    def fake_run(cmd, cwd=None, capture_output=None, text=None, check=None,
                 timeout=None):
        for i, token in enumerate(cmd):
            if token == "--output" and i + 1 < len(cmd):
                dest = cmd[i + 1]
                with open(dest, "wb") as f:
                    f.write(b"x" * 2048)
        return _sp.CompletedProcess(args=cmd, returncode=0,
                                     stdout="ok\n", stderr="")

    monkeypatch.setattr(agents.subprocess, "run", fake_run)
    r = agents._render_transition_flf2v(proj, a, b, out, duration_sec=3,
                                         prompt="smooth morph")
    assert r["backend"] == "wan21_flf2v"
    assert r["mode"] == "local"
    assert os.path.exists(out)
    assert os.path.getsize(out) > 1024


def test_flf2v_config_path_sits_outside_film(tmp_path: Path):
    proj = _mkproj(tmp_path)
    from filmmaker import agents
    cfg = agents._flf2v_config_path(proj)
    assert os.path.basename(cfg) == "config.json"
    assert os.sep + "motion-wan21-flf2v" + os.sep in cfg
    assert not cfg.startswith(proj.dir + os.sep)
