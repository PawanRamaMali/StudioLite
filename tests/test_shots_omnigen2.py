"""Tests for the OmniGen2 character-keyframe helper.

OmniGen2 is GPU-only (~20 GB VRAM) and research-registered as a roadmap
backend for the shots stage. These tests exercise the three resolution
paths of ``_render_shots_omnigen2`` - local install, cloud burst, and
skip-with-install-hint - without touching the network or a GPU.

The helper is called directly rather than through run_shots so the test
does not drag in SDXL and torch; wiring into run_shots is a follow-up
once the research-confirmed numbers land in CI."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

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


def _prompts() -> List[Dict[str, Any]]:
    return [
        {"scene_id": "s1", "shot_id": "sh1", "prompt": "a lighthouse at dusk"},
        {"scene_id": "s1", "shot_id": "sh2", "prompt": "the keeper opens a letter"},
    ]


def test_omnigen2_without_local_or_cloud_returns_install_hint(tmp_path: Path, monkeypatch):
    """No config.json, no cloud provider -> skip artifact. Never raises."""
    monkeypatch.delenv("STUDIOLITE_CLOUD_PROVIDER", raising=False)
    monkeypatch.delenv("STUDIOLITE_CLOUD_DRY_RUN", raising=False)
    monkeypatch.delenv("REPLICATE_API_TOKEN", raising=False)
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)

    proj = _mkproj(tmp_path)
    out_dir = os.path.join(proj.dir, "omnigen2-shots")
    from filmmaker import agents

    r = agents._render_shots_omnigen2(proj, _prompts(), out_dir)
    assert r["backend"] == "omnigen2"
    assert r["mode"] == "skipped"
    assert "install_hint" in r
    assert "REPLICATE_SLUG_OMNIGEN2" in r["install_hint"]
    assert r["shots"] == []


def test_omnigen2_cloud_dry_run_marks_each_shot(tmp_path: Path, monkeypatch):
    """DRY_RUN writes a marker file at output_path (the out_dir itself).
    The helper also looks for one PNG per prompt there - the dry-run
    provider does not actually render those, so the shots list stays
    empty but mode is cloud and the provider is reported."""
    monkeypatch.setenv("STUDIOLITE_CLOUD_DRY_RUN", "1")

    proj = _mkproj(tmp_path)
    out_dir = os.path.join(proj.dir, "omnigen2-shots")
    from filmmaker import agents

    r = agents._render_shots_omnigen2(proj, _prompts(), out_dir)
    assert r["backend"] == "omnigen2"
    assert r["mode"] == "cloud"
    assert r["provider"] == "dry-run"


def test_omnigen2_local_config_missing_paths_falls_through(tmp_path: Path, monkeypatch):
    """A config.json pointing at non-existent paths must not crash; it
    falls through to the cloud dispatcher (also absent here, so skip)."""
    monkeypatch.delenv("STUDIOLITE_CLOUD_PROVIDER", raising=False)
    monkeypatch.delenv("STUDIOLITE_CLOUD_DRY_RUN", raising=False)
    monkeypatch.delenv("REPLICATE_API_TOKEN", raising=False)
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)

    proj = _mkproj(tmp_path)
    out_dir = os.path.join(proj.dir, "omnigen2-shots")
    from filmmaker import agents

    cfg_path = agents._omnigen2_config_path(proj)
    os.makedirs(os.path.dirname(cfg_path), exist_ok=True)
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump({
            "repo_dir": str(tmp_path / "does-not-exist"),
            "venv_python": str(tmp_path / "nope.exe"),
            "model_path": str(tmp_path / "weights.safetensors"),
        }, f)

    r = agents._render_shots_omnigen2(proj, _prompts(), out_dir)
    assert r["backend"] == "omnigen2"
    assert r["mode"] == "skipped"
    assert "install_hint" in r


def test_omnigen2_local_runs_subprocess_and_collects_pngs(tmp_path: Path, monkeypatch):
    """A valid config + a subprocess that writes one PNG per prompt
    produces mode=local with every shot resolved."""
    proj = _mkproj(tmp_path)
    out_dir = os.path.join(proj.dir, "omnigen2-shots")

    from filmmaker import agents
    cfg_path = agents._omnigen2_config_path(proj)
    repo_dir = tmp_path / "omnigen2-repo"
    repo_dir.mkdir()
    venv_py = repo_dir / "python.exe"
    venv_py.write_bytes(b"stub")
    weights = repo_dir / "model.safetensors"
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
        # The script reads the prompts manifest and writes one PNG per entry.
        try:
            manifest_idx = cmd.index("--prompts")
            output_idx = cmd.index("--output_dir")
            with open(cmd[manifest_idx + 1], "r", encoding="utf-8") as m:
                manifest = json.load(m)
            for p in manifest["prompts"]:
                name = f"{p['scene_id']}_{p['shot_id']}.png"
                with open(os.path.join(cmd[output_idx + 1], name), "wb") as f:
                    f.write(b"x" * 2048)
        except Exception:
            pass
        return _sp.CompletedProcess(args=cmd, returncode=0,
                                     stdout="ok\n", stderr="")

    monkeypatch.setattr(agents.subprocess, "run", fake_run)
    r = agents._render_shots_omnigen2(proj, _prompts(), out_dir)
    assert r["backend"] == "omnigen2"
    assert r["mode"] == "local"
    rendered = {(s["scene_id"], s["shot_id"]) for s in r["shots"]}
    assert rendered == {("s1", "sh1"), ("s1", "sh2")}


def test_omnigen2_config_path_sits_outside_film(tmp_path: Path):
    """config.json lives next to the films/ tree, not inside a specific
    film - deleting a film must not blow away the backend install."""
    proj = _mkproj(tmp_path)
    from filmmaker import agents
    cfg = agents._omnigen2_config_path(proj)
    assert os.path.basename(cfg) == "config.json"
    assert os.sep + "shots-omnigen2" + os.sep in cfg
    # Not inside the project directory.
    assert not cfg.startswith(proj.dir + os.sep)
