"""Tests for the continuity / QC stage.

The stage is deterministic — it walks every upstream artifact and reports
issues against them. The tests hand-build minimal artifacts, then assert
`run_continuity` emits (or doesn't emit) the expected issue."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List


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


def _write(proj, key: str, data: Dict[str, Any]) -> None:
    proj.write_artifact(key, data)


def test_continuity_clean_run_reports_ok(tmp_path: Path):
    """A tidy project with one scene / one shot / one cast speaking the one
    dialogue line should report ok=True and zero error issues."""
    from filmmaker import agents
    proj = _mkproj(tmp_path)

    _write(proj, "breakdown", {"scenes": [
        {"id": "s01", "heading": "INT. LIGHTHOUSE - NIGHT",
         "estimated_seconds": 6, "characters": ["MORGAN"]},
    ]})
    _write(proj, "storyboard", {"scenes": {"s01": [
        {"id": "sh1", "description": "wide room"},
    ]}})
    _write(proj, "cinematographer", {"scenes": {"s01": {
        "sh1": {"framing": "WS", "lens_mm": 24, "camera_move": "dolly",
                "duration_sec": 4},
    }}})
    _write(proj, "voice_cast", {"cast": {
        "MORGAN": {"voice": "en_US-ryan-high", "gender": "male"},
    }})
    _write(proj, "character_portraits", {"portraits": {
        "MORGAN": {"path": "portraits/MORGAN.png", "rendered": True},
    }})
    _write(proj, "voice_actor", {"lines": [
        {"scene_id": "s01", "shot_id": "sh1", "speaker": "MORGAN",
         "text": "Not again.", "wav": "voice/s01_sh1.wav",
         "synthesized": True, "error": None, "duration_sec": 1.1},
    ]})
    # Create the shot render on disk so the file-existence check passes.
    shot_rel = "artifacts/shots/s01_sh1.png"
    shot_abs = os.path.join(proj.dir, shot_rel.replace("/", os.sep))
    os.makedirs(os.path.dirname(shot_abs), exist_ok=True)
    with open(shot_abs, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")  # just needs to exist
    _write(proj, "shots", {"shots": [
        {"scene_id": "s01", "shot_id": "sh1", "path": shot_rel,
         "duration_sec": 4, "rendered": True, "error": None},
    ]})
    _write(proj, "motion_shots", {"motion_shots": [
        {"scene_id": "s01", "shot_id": "sh1",
         "path": "artifacts/motion/s01_sh1.mp4",
         "rendered": True, "backend": "svd"},
    ]})
    editor_rel = "silent_cut.mp4"
    editor_abs = os.path.join(proj.dir, editor_rel)
    with open(editor_abs, "wb") as f: f.write(b"stub")
    _write(proj, "editor", {"output_path": editor_rel,
                             "duration_sec": 55, "shot_count": 1})
    _write(proj, "ambient", {"path": "ambient.wav", "duration_sec": 55,
                              "rendered": True, "error": None})
    _write(proj, "composer", {"path": "score.wav", "duration_sec": 55,
                               "rendered": True, "error": None})
    _write(proj, "sound_design", {
        "scenes": {}, "cue_count": 0, "rendered": 0, "skipped": 0,
    })
    final_rel = "final.mp4"
    final_abs = os.path.join(proj.dir, final_rel)
    with open(final_abs, "wb") as f: f.write(b"stub")
    _write(proj, "titles", {"output_path": final_rel,
                             "title": "Beacon", "cast_line_count": 1})

    report = agents.run_continuity(proj)
    assert report["ok"] is True, report
    assert report["error_count"] == 0, report["issues"]
    assert report["stats"]["final_output_path"] == final_rel
    assert report["stats"]["final_source_stage"] == "titles"


def test_continuity_flags_missing_shot_file(tmp_path: Path):
    """A shots artifact that advertises a path to a file that doesn't exist
    on disk should produce an error-level issue."""
    from filmmaker import agents
    proj = _mkproj(tmp_path)
    _write(proj, "breakdown", {"scenes": [
        {"id": "s01", "heading": "INT. LIGHTHOUSE - NIGHT",
         "estimated_seconds": 4},
    ]})
    _write(proj, "storyboard", {"scenes": {"s01": [{"id": "sh1"}]}})
    _write(proj, "shots", {"shots": [
        {"scene_id": "s01", "shot_id": "sh1",
         "path": "artifacts/shots/not_on_disk.png",  # file doesn't exist
         "rendered": True, "error": None},
    ]})

    report = agents.run_continuity(proj)
    missing = [i for i in report["issues"]
               if i["stage"] == "shots" and i["severity"] == "error"]
    assert missing, report
    assert "not_on_disk" in missing[0]["message"] or "missing on disk" in missing[0]["message"]
    assert report["ok"] is False


def test_continuity_flags_uncast_speaker(tmp_path: Path):
    """A dialogue line whose speaker isn't in voice_cast should emit a
    warning-level issue (not an error — the pipeline uses a fallback voice)."""
    from filmmaker import agents
    proj = _mkproj(tmp_path)
    _write(proj, "breakdown", {"scenes": [{"id": "s01",
                                             "heading": "INT. LIGHTHOUSE - NIGHT",
                                             "estimated_seconds": 4}]})
    _write(proj, "storyboard", {"scenes": {"s01": [{"id": "sh1"}]}})
    _write(proj, "voice_cast", {"cast": {}})  # no cast at all
    _write(proj, "voice_actor", {"lines": [
        {"scene_id": "s01", "shot_id": "sh1", "speaker": "GHOST",
         "text": "Hello.", "wav": "voice/x.wav",
         "synthesized": True, "error": None},
    ]})
    report = agents.run_continuity(proj)
    uncast = [i for i in report["issues"] if i["stage"] == "voice_cast"]
    assert uncast, report["issues"]
    assert uncast[0]["severity"] == "warning"
    assert "GHOST" in uncast[0]["message"]


def test_continuity_flags_silent_fallbacks(tmp_path: Path):
    """When ambient / composer / sound_design marked rendered=False (model
    unavailable), continuity should emit warnings naming each."""
    from filmmaker import agents
    proj = _mkproj(tmp_path)
    _write(proj, "breakdown", {"scenes": [{"id": "s01",
                                             "heading": "INT.",
                                             "estimated_seconds": 4}]})
    _write(proj, "storyboard", {"scenes": {"s01": []}})
    _write(proj, "ambient", {"path": "ambient.wav", "rendered": False,
                              "error": "AudioLDM import failed"})
    _write(proj, "composer", {"path": "score.wav", "rendered": False,
                               "error": "MusicGen OOM"})
    _write(proj, "sound_design", {"scenes": {}, "cue_count": 3,
                                   "rendered": 1, "skipped": 2})

    report = agents.run_continuity(proj)
    stages_flagged = {i["stage"] for i in report["issues"]}
    assert "ambient" in stages_flagged
    assert "composer" in stages_flagged
    assert "sound_design" in stages_flagged


def test_continuity_flags_duration_drift(tmp_path: Path):
    """target_minutes=1.0 but editor produced 180s should flag editor timing."""
    from filmmaker import agents
    proj = _mkproj(tmp_path)
    _write(proj, "editor", {"output_path": "silent_cut.mp4",
                             "duration_sec": 180, "shot_count": 10})
    report = agents.run_continuity(proj)
    drift = [i for i in report["issues"] if i["stage"] == "editor"]
    assert drift, report
    assert "over" in drift[0]["message"].lower()
