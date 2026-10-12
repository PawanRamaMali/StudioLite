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


def test_lipsync_musetalk_backend_not_installed_yet(tmp_path: Path):
    """MuseTalk backend is still stubbed - check_installed should report
    available=False with a specific install_hint, and run() should return
    a passthrough artifact citing that hint."""
    from filmmaker.lipsync import run_lipsync, get_backend
    proj = _mkproj(tmp_path, lipsync_backend="musetalk")
    _write_fake_final(proj)

    status = get_backend("musetalk").check_installed(proj)
    assert status.name == "musetalk"
    assert status.available is False
    assert "venv" in status.install_hint.lower()

    result = run_lipsync(proj)
    assert result["backend"] == "musetalk"
    assert result["rendered"] is False
    assert "not yet wired" in result["skipped"]
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
