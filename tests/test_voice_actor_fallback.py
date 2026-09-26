"""Regression cover for the voice_actor stage.

The Late Shift render hit two systemic issues we saw in the field:
  1. The screenwriter produced a screenplay with only ONE scene heading
     (or none), so ``_split_screenplay_by_scene`` collapsed every line
     into scene 1 and every subsequent scene came back silent.
  2. Even in scene 1, the first-pass placement dropped every line whose
     shot had no ``subject`` / ``action`` text, and the second-pass loop
     only assigned up to ``len(shots)`` lines total, so a 15-line scene
     with 5 shots silently discarded 10 lines.

Together those two bugs produced a film with `voice_actor.json` empty
and no audio at all. This module locks in the fallback that
redistributes dialogue across every shot when the split fails.
"""
from __future__ import annotations

import json
import os
import tempfile
from unittest.mock import patch

import pytest

from filmmaker import agents


ORPHAN_SCREENPLAY = """INT.

JESSICA
The storm will pass.

BART
It always does. Eventually.

JESSICA
You look worse than last week.

BART
I feel worse than last month.

JESSICA
Come inside. Sit down.

BART
Not tonight.

JESSICA
Why?

BART
The light was flickering again. Twice. Same rhythm as thirty years ago.

JESSICA
That was a storm.

BART
That was not a storm.

JESSICA
You're tired.

BART
I'm awake for the first time in decades. That's the trouble.

JESSICA
Then come inside and be awake somewhere warm.

BART
I don't think it wants me warm.

JESSICA
It doesn't want anything. It's a light.
"""


class _StubProject:
    """Minimal Project stand-in that satisfies run_voice_actor without
    touching real disk beyond a temp dir."""

    def __init__(self, tmp_path):
        self.dir = str(tmp_path)
        self.artifacts_dir = str(tmp_path / "artifacts")
        os.makedirs(self.artifacts_dir, exist_ok=True)
        self._artifacts = {}

        class _Meta:
            class config:
                voice_backend = "piper"
                per_stage = {}
        self.meta = _Meta()

    def read_artifact(self, key):
        return self._artifacts.get(key)

    def write_artifact(self, key, data, snapshot=True, note=""):
        self._artifacts[key] = data


def _stub_synth(text, wav_abs, **_kwargs):
    """Piper stand-in: write a 44-byte WAV header so the mixer's file-exists
    check passes and the record is added to the artifact."""
    with open(wav_abs, "wb") as f:
        # Minimal WAV header for a 0-sample PCM file. The stage only cares
        # that the file exists and its duration reads as a non-negative
        # number; wave.open handles this shape without raising.
        f.write(b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00"
                b"\x01\x00\x01\x00D\xac\x00\x00\x88X\x01\x00"
                b"\x02\x00\x10\x00data\x00\x00\x00\x00")


class _PiperStub:
    def __init__(self, voice_name=""):
        pass

    def synthesize(self, text, wav_abs):
        _stub_synth(text, wav_abs)


def test_voice_actor_produces_a_line_per_shot_when_screenplay_has_no_scene_slugs(
        tmp_path):
    """The Late Shift scenario: 7 storyboard scenes, 28 shots, screenplay
    has no reliable scene slugs. Before the fix, the artifact came back
    with ``lines: []``. After the fix, we get one wav per shot."""
    proj = _StubProject(tmp_path)

    # Seven scenes, four shots each. Storyboard shot dicts intentionally
    # have empty subject/action so the first-pass "shot mentions speaker"
    # match never fires; the second-pass proportional distribution has to
    # carry the whole load.
    scenes = [{"id": f"s{n:02d}", "heading": "INT. Diner - DAY",
                "estimated_seconds": 20} for n in range(1, 8)]
    storyboard_scenes = {
        s["id"]: [{"id": f"sh{i}", "subject": "", "action": ""}
                   for i in range(1, 5)]
        for s in scenes
    }

    proj.write_artifact("screenwriter", {"fountain": ORPHAN_SCREENPLAY})
    proj.write_artifact("story_editor", {"revised_fountain": ""})
    proj.write_artifact("breakdown", {"scenes": scenes})
    proj.write_artifact("storyboard", {"scenes": storyboard_scenes})
    proj.write_artifact("voice_cast", {"cast": {
        "JESSICA": {"voice": "en_US-amy-medium"},
        "BART": {"voice": "en_US-lessac-medium"},
    }})

    with patch.object(agents, "PiperTTS", _PiperStub, create=True), \
         patch("mpv2.classes.PiperTts.PiperTTS", _PiperStub, create=True):
        result = agents.run_voice_actor(proj)

    # Every scene with shots should carry at least one synthesized line.
    scene_ids = {ln["scene_id"] for ln in result["lines"]}
    assert scene_ids == {f"s{n:02d}" for n in range(1, 8)}, (
        f"Some scenes ended up silent. Got scene ids {sorted(scene_ids)}"
    )
    # Every line record must carry a speaker string (never blank).
    for ln in result["lines"]:
        assert ln["speaker"], f"Blank speaker on line {ln}"
        assert ln["wav"].endswith(".wav")
