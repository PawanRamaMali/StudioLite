"""Tests for the SeedVR2 upscale branch of run_upscale.

SeedVR2 is GPU-only in practice (3B/7B DiT; GGUF ladder via ComfyUI-GGUF).
These tests exercise the three resolution paths - local install, cloud
burst, and skip-with-install-hint - without touching the network or
needing any model weights."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

import pytest


def _mkproj(tmp: Path, backend: str = "seedvr2"):
    from filmmaker import projects
    return projects.create(
        str(tmp),
        brief="A keeper opens a letter.",
        title="Beacon",
        config_dict={
            "llm_backend": "ollama", "llm_model": "mock",
            "style": "stylized", "target_minutes": 1.0,
            "upscale_backend": backend,
        },
    )


def _write_final(proj) -> str:
    final = os.path.join(proj.dir, "final.mp4")
    with open(final, "wb") as f:
        f.write(b"stub-final")
    return final


def test_seedvr2_without_local_or_cloud_returns_install_hint(tmp_path: Path, monkeypatch):
    """No config.json, no cloud provider -> we return a skip artifact that
    tells the user exactly which paths exist. Never raises."""
    monkeypatch.delenv("STUDIOLITE_CLOUD_PROVIDER", raising=False)
    monkeypatch.delenv("STUDIOLITE_CLOUD_DRY_RUN", raising=False)
    monkeypatch.delenv("REPLICATE_API_TOKEN", raising=False)
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)

    proj = _mkproj(tmp_path)
    _write_final(proj)
    from filmmaker import agents
    result = agents.run_upscale(proj)
    assert result["backend"] == "seedvr2"
    assert "skipped" in result
    assert "install_hint" in result
    assert "REPLICATE_SLUG_SEEDVR2" in result["install_hint"]
    # Output path falls back to the untouched final so continuity still
    # has a video to inspect.
    assert result["output_path"] == "final.mp4"


def test_seedvr2_cloud_burst_dry_run_writes_output(tmp_path: Path, monkeypatch):
    """With STUDIOLITE_CLOUD_DRY_RUN=1, the dispatcher writes a marker at
    the expected output path and run_upscale picks that up as success."""
    monkeypatch.setenv("STUDIOLITE_CLOUD_DRY_RUN", "1")

    proj = _mkproj(tmp_path)
    _write_final(proj)
    from filmmaker import agents
    result = agents.run_upscale(proj)
    assert result["backend"] == "seedvr2"
    assert result.get("mode") == "cloud"
    assert result.get("provider") == "dry-run"
    assert result["output_path"] == "final_upscaled.mp4"
    out_abs = os.path.join(proj.dir, "final_upscaled.mp4")
    assert os.path.exists(out_abs)


def test_seedvr2_local_config_missing_paths_falls_through(tmp_path: Path, monkeypatch):
    """A config.json that lies about where repo_dir/venv_python live must
    not crash the stage - it falls through to the cloud dispatcher (which
    is also unavailable here, so we end on the skip-artifact)."""
    monkeypatch.delenv("STUDIOLITE_CLOUD_PROVIDER", raising=False)
    monkeypatch.delenv("STUDIOLITE_CLOUD_DRY_RUN", raising=False)
    monkeypatch.delenv("REPLICATE_API_TOKEN", raising=False)
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)

    proj = _mkproj(tmp_path)
    _write_final(proj)

    from filmmaker import agents
    cfg_path = agents._seedvr2_config_path(proj)
    os.makedirs(os.path.dirname(cfg_path), exist_ok=True)
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump({
            "repo_dir": str(tmp_path / "does-not-exist"),
            "venv_python": str(tmp_path / "nope.exe"),
            "model_path": str(tmp_path / "weights.safetensors"),
            "variant": "3b",
        }, f)

    result = agents.run_upscale(proj)
    # No cloud provider either -> skip, with install hint.
    assert result["backend"] == "seedvr2"
    assert "skipped" in result
    assert "install_hint" in result


def test_seedvr2_local_runs_subprocess_on_valid_config(tmp_path: Path, monkeypatch):
    """When the config paths all exist and the subprocess writes a plausible
    output file, run_upscale reports mode=local and the output path."""
    proj = _mkproj(tmp_path)
    src = _write_final(proj)

    from filmmaker import agents
    cfg_path = agents._seedvr2_config_path(proj)
    repo_dir = tmp_path / "seedvr2-repo"
    repo_dir.mkdir()
    venv_py = repo_dir / "python.exe"
    venv_py.write_bytes(b"stub")
    weights = repo_dir / "weights.safetensors"
    weights.write_bytes(b"stub")

    os.makedirs(os.path.dirname(cfg_path), exist_ok=True)
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump({
            "repo_dir": str(repo_dir),
            "venv_python": str(venv_py),
            "model_path": str(weights),
            "variant": "3b",
        }, f)

    # Stub subprocess.run so the test does not actually execute the fake
    # python.exe. The stub writes 2 KiB to the output path to pass the
    # size check in _run_seedvr2_local.
    import subprocess as _sp

    def fake_run(cmd, cwd=None, capture_output=None, text=None, check=None,
                  timeout=None):
        # Find --output path in the cmd and write a stub file there.
        for i, token in enumerate(cmd):
            if token == "--output" and i + 1 < len(cmd):
                dest = cmd[i + 1]
                with open(dest, "wb") as f:
                    f.write(b"x" * 2048)
        return _sp.CompletedProcess(args=cmd, returncode=0,
                                      stdout="ok\n", stderr="")

    monkeypatch.setattr(agents.subprocess, "run", fake_run)
    result = agents.run_upscale(proj)
    assert result["backend"] == "seedvr2"
    assert result.get("mode") == "local"
    assert result["output_path"] == "final_upscaled.mp4"
    assert result.get("variant") == "3b"
    assert os.path.exists(os.path.join(proj.dir, "final_upscaled.mp4"))


def test_seedvr2_without_final_mp4_short_circuits(tmp_path: Path):
    """No final.mp4 on disk -> early return, matching the existing
    Real-ESRGAN behaviour. Never reaches the new seedvr2 branch."""
    proj = _mkproj(tmp_path)
    # No final.mp4 written.
    from filmmaker import agents
    result = agents.run_upscale(proj)
    assert result["backend"] == "seedvr2"
    assert result.get("skipped") == "final.mp4 not found"
