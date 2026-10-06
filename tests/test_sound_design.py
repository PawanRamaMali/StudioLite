"""Tests for the sound_design stage.

AudioLDM 2 and ffmpeg are patched out so no models load and no mixer
subprocess runs. The point is the stage logic: cue normalisation,
dialogue-collision shifting, and that `run_mixer` picks the artifact up."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch

import pytest


def _mkproject(tmp: Path):
    """Build a Project with the artifacts the sound_design stage needs."""
    from filmmaker import projects
    proj = projects.create(
        str(tmp),
        brief="A keeper opens a letter.",
        title="Beacon",
        config_dict={
            "llm_backend": "ollama", "llm_model": "mock",
            "style": "stylized", "target_minutes": 1.0,
        },
    )
    # Breakdown — one scene
    proj.write_artifact("breakdown", {"scenes": [
        {"id": "s01", "heading": "INT. LIGHTHOUSE - NIGHT",
         "summary": "Morgan opens the letter.",
         "location": "lighthouse-desk", "interior": True, "time_of_day": "night",
         "characters": ["MORGAN"], "mood": "brooding stillness",
         "estimated_seconds": 8},
    ]})
    # Storyboard — three shots
    proj.write_artifact("storyboard", {"scenes": {"s01": [
        {"id": "sh1", "description": "wide of the lighthouse room",
         "subject": "MORGAN", "action": "sits at the desk", "dialogue": ""},
        {"id": "sh2", "description": "close on wax seal cracking",
         "subject": "the letter", "action": "seal snaps", "dialogue": "Not again."},
        {"id": "sh3", "description": "Morgan's eyes reflect the flame",
         "subject": "MORGAN", "action": "his eyes widen", "dialogue": ""},
    ]}})
    # Shots artifact — gives each shot a duration (used by scene_dur math).
    proj.write_artifact("shots", {"shots": [
        {"scene_id": "s01", "shot_id": "sh1", "duration_sec": 4,
         "prompt": "", "path": "stub.png", "dialogue": "",
         "rendered": True, "error": None},
        {"scene_id": "s01", "shot_id": "sh2", "duration_sec": 3,
         "prompt": "", "path": "stub.png", "dialogue": "Not again.",
         "rendered": True, "error": None},
        {"scene_id": "s01", "shot_id": "sh3", "duration_sec": 3,
         "prompt": "", "path": "stub.png", "dialogue": "",
         "rendered": True, "error": None},
    ]})
    proj.write_artifact("voice_actor", {"lines": [
        {"scene_id": "s01", "shot_id": "sh2", "speaker": "MORGAN",
         "voice": "en_US-ryan-high", "backend": "piper",
         "text": "Not again.", "wav": "voice/s01_sh2.wav",
         "duration_sec": 1.2, "synthesized": True, "error": None},
    ]})
    # Editor artifact — total duration
    proj.write_artifact("editor", {
        "output_path": "silent.mp4",
        "duration_sec": 10, "shot_count": 3,
    })
    return proj


def _cue_list_mock(*_args, **_kwargs):
    """LLM mock returning three cues: one safe hard cue (sfx1), one that
    collides with dialogue (sfx2, should shift), and one soft texture bed."""
    return json.dumps({"cues": [
        # Hard cue in a quiet moment (shot sh1, before any dialogue)
        {"sfx_id": "sfx1", "shot_id": "sh1", "prompt": "paper rustle",
         "start_sec": 1.0, "duration_sec": 0.8, "gain_db": -6},
        # Hard cue that OVERLAPS the dialogue window (4.0..5.4s). Expect shift.
        {"sfx_id": "sfx2", "shot_id": "sh2", "prompt": "wooden chair creak",
         "start_sec": 4.3, "duration_sec": 0.6, "gain_db": -5},
        # Soft texture bed, long + quiet, allowed to overlap speech.
        {"sfx_id": "sfx3", "shot_id": "sh3", "prompt": "distant wind under boards",
         "start_sec": 2.0, "duration_sec": 8.0, "gain_db": -18},
    ]})


def test_run_sound_design_normalizes_and_shifts_cues_out_of_dialogue(tmp_path: Path):
    from filmmaker import agents
    proj = _mkproject(tmp_path)

    silent_count = {"n": 0}
    def _fake_audioldm(prompt, out_path, *, duration_sec):
        # Pretend the AudioLDM-free fallback — we write silent wavs via the
        # same helper the real failure branch does, but counted so we know
        # the stage really tried to render N cues.
        silent_count["n"] += 1
        agents._write_silent_wav(out_path, max(0.3, float(duration_sec)))

    with patch.object(agents, "_chat", side_effect=_cue_list_mock), \
         patch.object(agents, "_render_audioldm", side_effect=_fake_audioldm):
        result = agents.run_sound_design(proj)

    scenes = result["scenes"]
    assert "s01" in scenes, result
    cues = scenes["s01"]
    assert len(cues) == 3, cues

    by_id = {c["sfx_id"]: c for c in cues}

    # sfx1 — unchanged, in the dialogue-free range
    assert by_id["sfx1"]["start_sec"] == pytest.approx(1.0)
    assert by_id["sfx1"]["rendered"] is True

    # sfx2 — overlaps dialogue (4.0..5.4), should move to after it.
    # Dialogue ends at 4.0 + 1.2 + 0.2 pad = 5.4, so a gap at 5.4 + 0.3 = 5.7
    # is the first that fits a 0.6s cue inside the 10s scene.
    assert by_id["sfx2"]["start_sec"] >= 5.7 - 0.01, by_id["sfx2"]
    assert by_id["sfx2"]["rendered"] is True

    # sfx3 — soft cue (long, quiet) is NOT shifted even though it overlaps.
    assert by_id["sfx3"]["start_sec"] == pytest.approx(2.0)
    # Clamped to scene duration (10s).
    assert by_id["sfx3"]["duration_sec"] <= 10.0 - 2.0 + 0.01

    # All three got rendered via the (patched) AudioLDM path.
    assert silent_count["n"] == 3
    assert result["cue_count"] == 3
    assert result["rendered"] == 3


def test_sound_design_falls_through_to_silent_cue_when_audioldm_unavailable(tmp_path: Path):
    """If AudioLDM import blows up, the stage still produces cue rows with
    `rendered=False` and silent-wav wav files on disk so the mixer has
    something to point at."""
    from filmmaker import agents
    proj = _mkproject(tmp_path)

    def _audioldm_broken(prompt, out_path, *, duration_sec):
        raise RuntimeError("AudioLDM import failed")

    with patch.object(agents, "_chat", side_effect=_cue_list_mock), \
         patch.object(agents, "_render_audioldm", side_effect=_audioldm_broken):
        result = agents.run_sound_design(proj)

    cues = result["scenes"]["s01"]
    for c in cues:
        if c.get("error") == "overlaps dialogue; no free gap":
            continue
        assert c["rendered"] is False
        assert c["wav"] is not None
        # Silent wav really lands on disk at the advertised path.
        wav_abs = os.path.join(proj.artifacts_dir, c["wav"])
        assert os.path.exists(wav_abs), wav_abs
    assert result["rendered"] == 0
    assert result["skipped"] >= 1


def test_shift_out_of_dialogue_handles_full_scene_and_empty_windows():
    from filmmaker.agents import _shift_out_of_dialogue
    # No windows — start unchanged.
    assert _shift_out_of_dialogue(2.0, 1.0, [], 10.0) == 2.0
    # Collision with one window — moves to end + 0.3.
    assert _shift_out_of_dialogue(1.5, 0.5, [(1.0, 3.0)], 10.0) == 3.3
    # No slot fits — returns None.
    assert _shift_out_of_dialogue(1.0, 2.0, [(0, 10)], 10.0) is None


def test_mixer_picks_up_sound_design_cues_without_running_ffmpeg(tmp_path: Path):
    """`run_mixer` should include has_sfx + sfx_cues in its return once the
    sound_design artifact exists with rendered cues on disk."""
    from filmmaker import agents
    proj = _mkproject(tmp_path)
    # Minimum rig to drive run_mixer up to just before the ffmpeg subprocess.
    silent_abs = os.path.join(proj.dir, "silent.mp4")
    with open(silent_abs, "wb") as f: f.write(b"stub")   # file just needs to exist
    # Composer artifact (empty score — silent fallback)
    proj.write_artifact("composer", {
        "prompt": "x", "path": "composer.wav", "duration_sec": 10,
        "rendered": False, "error": None,
    })
    proj.write_artifact("ambient", {
        "prompt": "x", "path": "ambient.wav", "duration_sec": 10,
        "rendered": False, "error": None,
    })
    # Fake sound_design artifact with one rendered cue
    sfx_rel = "sfx/s01_sfx1.wav"
    sfx_abs = os.path.join(proj.artifacts_dir, sfx_rel.replace("/", os.sep))
    os.makedirs(os.path.dirname(sfx_abs), exist_ok=True)
    agents._write_silent_wav(sfx_abs, 0.8)
    proj.write_artifact("sound_design", {
        "scenes": {"s01": [{
            "sfx_id": "sfx1", "shot_id": "sh1", "prompt": "paper rustle",
            "start_sec": 1.0, "duration_sec": 0.8, "gain_db": -6.0,
            "wav": sfx_rel, "rendered": True, "error": None,
        }]},
        "cue_count": 1, "rendered": 1, "skipped": 0,
    })

    captured: Dict[str, Any] = {}
    def _fake_mixer_ffmpeg(**kwargs):
        captured.update(kwargs)
    with patch.object(agents, "_run_mixer_ffmpeg", side_effect=_fake_mixer_ffmpeg):
        result = agents.run_mixer(proj)

    # The mixer saw our one SFX cue, correctly offset to scene start 0 + 1.0s = 1000ms.
    assert captured.get("sfx") == [{
        "wav": sfx_abs, "delay_ms": 1000, "gain_db": -6.0,
    }], captured.get("sfx")
    assert result["has_sfx"] is True
    assert result["sfx_cues"] == 1
