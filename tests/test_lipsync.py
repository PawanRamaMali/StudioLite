"""Tests for the lipsync stage scaffold.

The stage is a thin dispatcher: it picks a backend from
`config.lipsync_backend`, points it at the best available final video,
and persists what the backend returns. Backends promise not to raise -
they emit a skip-artifact when their models aren't installed - so the
dispatcher's job is to pass arguments correctly and to never halt the
pipeline.

These tests cover only the scaffold - the actual MuseTalk / LatentSync /
EchoMimicV3 shims land in follow-up commits and get their own tests."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict


def _mkproj(tmp: Path, lipsync_backend: str = "none"):
    from filmmaker import projects
    return projects.create(
        str(tmp),
        brief="A keeper opens a letter.",
        title="Beacon",
        config_dict={
            "llm_backend": "ollama", "llm_model": "mock",
            "style": "stylized", "target_minutes": 1.0,
            "lipsync_backend": lipsync_backend,
        },
    )


def _write_fake_final(proj) -> str:
    """Drop a dummy mp4 at `final.mp4` so the input-picker finds something.
    Returns the relative path."""
    rel = "final.mp4"
    abs_path = os.path.join(proj.dir, rel)
    with open(abs_path, "wb") as f:
        f.write(b"stub-mp4")
    proj.write_artifact("titles", {"output_path": rel,
                                     "title": "Beacon", "cast_line_count": 0})
    return rel


def test_lipsync_none_backend_skips_cleanly(tmp_path: Path):
    """Default `lipsync_backend=none` must return a skip artifact, point
    output_path at the untouched input, and never raise."""
    from filmmaker.lipsync import run_lipsync
    proj = _mkproj(tmp_path, lipsync_backend="none")
    _write_fake_final(proj)

    result = run_lipsync(proj)
    assert result["backend"] == "none"
    assert result["rendered"] is False
    assert result["skipped"] == "lipsync_backend=none"
    assert result["output_path"] == result["input_path"]
    # Passthrough points at the actual final video, not a lipsync suffix.
    assert result["output_path"].endswith("final.mp4")


def test_lipsync_missing_final_video_reports_skip(tmp_path: Path):
    """When no final cut exists anywhere (upscale/titles/mixer all empty,
    no final.mp4 on disk), the stage still returns an artifact rather
    than raising. input_path/output_path are None."""
    from filmmaker.lipsync import run_lipsync
    proj = _mkproj(tmp_path, lipsync_backend="none")
    # No artifacts, no files on disk.
    result = run_lipsync(proj)
    assert result["rendered"] is False
    assert result["input_path"] is None
    assert result["output_path"] is None
    assert "no final video" in result["skipped"]


def test_lipsync_musetalk_backend_no_config_file(tmp_path: Path):
    """With musetalk selected but no config.json dropped at the expected
    path, the backend reports available=False naming the config path, and
    run() returns a passthrough citing the install hint."""
    from filmmaker.lipsync import run_lipsync, get_backend
    proj = _mkproj(tmp_path, lipsync_backend="musetalk")
    _write_fake_final(proj)

    status = get_backend("musetalk").check_installed(proj)
    assert status.name == "musetalk"
    assert status.available is False
    assert "config" in status.reason.lower()
    assert "lipsync-musetalk" in status.install_hint

    result = run_lipsync(proj)
    assert result["backend"] == "musetalk"
    assert result["rendered"] is False
    assert "config" in result["skipped"].lower()
    assert "install_hint" in result
    # Still passes through - final.mp4 isn't lipsync'd but downstream
    # continuity should see output_path pointing at the real file.
    assert result["output_path"].endswith("final.mp4")


def test_lipsync_unknown_backend_falls_back_to_none(tmp_path: Path):
    """A typo'd backend name must NOT halt the pipeline. The dispatcher
    falls back to the `none` backend and logs a warning."""
    from filmmaker.lipsync import run_lipsync
    proj = _mkproj(tmp_path, lipsync_backend="not-a-real-backend")
    _write_fake_final(proj)

    result = run_lipsync(proj)
    # get_backend fell back to NoneBackend, so the result reads as the
    # none-skip artifact.
    assert result["backend"] == "none"
    assert result["rendered"] is False


def test_lipsync_prefers_upscale_output_over_titles(tmp_path: Path):
    """When both upscale and titles produced output_paths, the picker
    should take the upscaled version (last writer wins) and feed that
    into the backend."""
    from filmmaker.lipsync import run_lipsync, _pick_input_video
    proj = _mkproj(tmp_path, lipsync_backend="none")

    # Write both files.
    titles_rel = "final.mp4"
    upscale_rel = "final_upscaled.mp4"
    for rel in (titles_rel, upscale_rel):
        with open(os.path.join(proj.dir, rel), "wb") as f:
            f.write(b"stub")
    proj.write_artifact("titles", {"output_path": titles_rel})
    proj.write_artifact("upscale", {"output_path": upscale_rel,
                                      "backend": "realesrgan_x2", "scale": 2})

    picked = _pick_input_video(proj)
    assert picked == upscale_rel

    result = run_lipsync(proj)
    assert result["input_path"] == upscale_rel


def _write_musetalk_config(tmp_path: Path, project_output_dir: Path,
                            overrides=None) -> Path:
    """Drop a `.mp/lipsync-musetalk/config.json` next to the project's
    films root. The backend resolves it from the project's _root_dir."""
    import json as _json
    cfg_dir = project_output_dir / "lipsync-musetalk"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    repo_dir = tmp_path / "musetalk_repo"
    repo_dir.mkdir()
    venv_py = tmp_path / "venv_py.exe"
    venv_py.write_bytes(b"stub")
    cfg = {
        "repo_dir": str(repo_dir),
        "venv_python": str(venv_py),
        "unet_model_path": "models/musetalkV15/unet.pth",
        "unet_config": "models/musetalkV15/musetalk.json",
        "version": "v15",
        "ffmpeg_path": "ffmpeg",
    }
    if overrides:
        cfg.update(overrides)
    cfg_path = cfg_dir / "config.json"
    cfg_path.write_text(_json.dumps(cfg), encoding="utf-8")
    return cfg_path


def test_musetalk_check_installed_missing_config(tmp_path: Path):
    """No config.json at all → available=False with a clear install_hint
    naming the expected path, so the UI can tell the user where to drop
    the file."""
    from filmmaker.lipsync import MuseTalkBackend
    proj = _mkproj(tmp_path, lipsync_backend="musetalk")
    status = MuseTalkBackend().check_installed(proj)
    assert status.available is False
    assert "config" in status.reason.lower()
    assert "lipsync-musetalk" in status.install_hint


def test_musetalk_check_installed_bad_repo_dir(tmp_path: Path):
    """Config exists but points at a repo_dir that isn't on disk →
    available=False with the actual path in the reason."""
    from filmmaker.lipsync import MuseTalkBackend
    proj = _mkproj(tmp_path, lipsync_backend="musetalk")
    _write_musetalk_config(tmp_path, tmp_path,
                            overrides={"repo_dir": str(tmp_path / "ghost")})
    status = MuseTalkBackend().check_installed(proj)
    assert status.available is False
    assert "repo_dir" in status.reason
    assert "ghost" in status.reason


def test_musetalk_check_installed_happy(tmp_path: Path):
    """Config valid + repo + venv_python all on disk → available=True."""
    from filmmaker.lipsync import MuseTalkBackend
    proj = _mkproj(tmp_path, lipsync_backend="musetalk")
    _write_musetalk_config(tmp_path, tmp_path)
    status = MuseTalkBackend().check_installed(proj)
    assert status.available is True
    assert status.reason == ""


def test_musetalk_run_no_dialogue_passthrough(tmp_path: Path):
    """Backend is installed but the film has no voice_actor lines →
    passthrough with 'no dialogue' reason, no subprocess call."""
    from filmmaker.lipsync import run_lipsync
    proj = _mkproj(tmp_path, lipsync_backend="musetalk")
    _write_musetalk_config(tmp_path, tmp_path)
    _write_fake_final(proj)
    result = run_lipsync(proj)
    assert result["backend"] == "musetalk"
    assert result["rendered"] is False
    assert "no dialogue" in result["skipped"].lower()


def test_musetalk_run_subprocess_success(tmp_path: Path, monkeypatch):
    """Happy path: speech track already exists (mixer_speech.wav), the
    subprocess call succeeds, and MuseTalk's result file is copied to
    the output path. No real ffmpeg or MuseTalk is invoked."""
    import subprocess as _sp
    from filmmaker.lipsync import run_lipsync

    proj = _mkproj(tmp_path, lipsync_backend="musetalk")
    _write_musetalk_config(tmp_path, tmp_path)
    final_rel = _write_fake_final(proj)

    # Give the backend a pre-rendered mixer-speech so it skips the
    # ffmpeg-concat path (that path is covered separately by the
    # voice_actor branch, and real ffmpeg isn't guaranteed on CI).
    speech_abs = os.path.join(proj.dir, "artifacts", "mixer_speech.wav")
    os.makedirs(os.path.dirname(speech_abs), exist_ok=True)
    with open(speech_abs, "wb") as f:
        f.write(b"stub-wav")

    # Register one voice_actor line so collect_speech_track has something
    # to count even though we're using the mixer path.
    proj.write_artifact("voice_actor", {"lines": [
        {"scene_id": "s01", "shot_id": "sh1", "speaker": "MORGAN",
         "text": "Hello.", "wav": "voice/s01_sh1.wav",
         "synthesized": True, "error": None},
    ]})

    calls: Dict[str, Any] = {}

    def _fake_run(cmd, cwd=None, capture_output=False, text=False,
                   check=False, timeout=None):
        calls["cmd"] = cmd
        calls["cwd"] = cwd
        # MuseTalk writes task_0.mp4 under --result_dir.
        rd_idx = cmd.index("--result_dir") + 1
        produced = os.path.join(cmd[rd_idx], "task_0.mp4")
        os.makedirs(cmd[rd_idx], exist_ok=True)
        with open(produced, "wb") as f:
            f.write(b"stub-musetalk-output")
        class _Proc:
            returncode = 0
            stdout = ""
            stderr = ""
        return _Proc()

    monkeypatch.setattr(_sp, "run", _fake_run)

    result = run_lipsync(proj)
    assert result["backend"] == "musetalk"
    assert result["rendered"] is True, result
    assert result["version"] == "v15"
    assert result["line_count"] == 1
    # Output lands with the _lipsync suffix next to the final video.
    assert result["output_path"].endswith("_lipsync.mp4")
    assert result["output_path"] != result["input_path"]
    # And the file is actually on disk where the artifact claims.
    out_abs = os.path.join(proj.dir, result["output_path"].replace("/", os.sep))
    assert os.path.exists(out_abs)
    # Verify we called into the MuseTalk inference module.
    assert calls["cmd"][1:4] == ["-m", "scripts.inference", "--inference_config"]


def test_musetalk_run_subprocess_failure_passthrough(tmp_path: Path, monkeypatch):
    """When MuseTalk exits non-zero, the backend captures the stderr
    tail and returns a passthrough artifact rather than halting."""
    import subprocess as _sp
    from filmmaker.lipsync import run_lipsync

    proj = _mkproj(tmp_path, lipsync_backend="musetalk")
    _write_musetalk_config(tmp_path, tmp_path)
    _write_fake_final(proj)
    speech_abs = os.path.join(proj.dir, "artifacts", "mixer_speech.wav")
    os.makedirs(os.path.dirname(speech_abs), exist_ok=True)
    with open(speech_abs, "wb") as f:
        f.write(b"stub")
    proj.write_artifact("voice_actor", {"lines": [
        {"scene_id": "s01", "shot_id": "sh1", "speaker": "M",
         "text": "Hi.", "wav": "voice/x.wav", "synthesized": True},
    ]})

    def _fake_run(cmd, **kw):
        class _Proc:
            returncode = 2
            stdout = ""
            stderr = "ImportError: libtorch_cuda.so.11.8 missing"
        return _Proc()

    monkeypatch.setattr(_sp, "run", _fake_run)

    result = run_lipsync(proj)
    assert result["backend"] == "musetalk"
    assert result["rendered"] is False
    assert "exited 2" in result["skipped"]
    assert "libtorch_cuda" in result["skipped"]
    # Passthrough points at the untouched final.
    assert result["output_path"] == result["input_path"]


def test_lipsync_backend_exception_is_contained(tmp_path: Path, monkeypatch):
    """A buggy backend that raises mid-run shouldn't halt the pipeline
    either - the dispatcher wraps the call and returns a skip-artifact
    naming the exception message."""
    from filmmaker import lipsync as lipsync_mod
    proj = _mkproj(tmp_path, lipsync_backend="musetalk")
    _write_fake_final(proj)

    # Make the MuseTalk stub claim it IS installed then raise on run().
    class Boom:
        name = "musetalk"
        def check_installed(self, project):
            from filmmaker.lipsync import BackendStatus
            return BackendStatus(name="musetalk", available=True)
        def run(self, project, input_video, voice_lines, output_video):
            raise RuntimeError("mmcv segfaulted")

    monkeypatch.setitem(lipsync_mod._BACKENDS, "musetalk", Boom())

    result = lipsync_mod.run_lipsync(proj)
    assert result["backend"] == "musetalk"
    assert result["rendered"] is False
    assert "mmcv segfaulted" in result["skipped"]
    # Passthrough - continuity can still find the untouched final.mp4.
    assert result["output_path"].endswith("final.mp4")
