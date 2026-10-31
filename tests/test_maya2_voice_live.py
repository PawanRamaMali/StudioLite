"""Live Maya2-native voice_actor stage test.

Runs the real ``filmmaker.agents.run_voice_actor`` against a Maya2
server reachable at ``MAYA2_URL``. Skipped when the env var is unset or
the endpoint does not report ``ready:true``, so this file is safe on
Windows/CI and in any environment without a running Maya2 instance.

To exercise it against a real server (for example a Linux box with
``python -m maya3.server --ckpt ~/models/maya2-native --host 0.0.0.0``):

    MAYA2_URL=http://127.0.0.1:8787 pytest tests/test_maya2_voice_live.py -v

The scene drives both Hindi (Nivedita) and English (Abhay) lines, so the
assertions prove the per-character voice/language cascade reaches the
engine with the right values and that the resulting WAVs are real audio
rather than silence.
"""
from __future__ import annotations

import json
import os
import tempfile
import urllib.request
import wave
from pathlib import Path

import pytest

from filmmaker import agents
from filmmaker.project import Project, ProjectConfig


SCREENPLAY = """\
INT. LIGHTHOUSE - LANTERN ROOM - NIGHT

JESS (30s, practical jacket, salt-swept hair) clings to the lantern.
KAI (40s, sodden coat, eyes obscured by hair) stumbles in.

    JESS
    What are you doing here?

    KAI
    I need a place to rest.

    JESS
    This is not a shelter.

    KAI
    The storm is coming back.
"""


def _server_ready(url: str) -> tuple[bool, str]:
    """Return (ready, reason). Reason is informational for the skip message."""
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/health", timeout=5) as r:
            body = r.read().decode("utf-8", "replace")
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    try:
        payload = json.loads(body)
    except Exception:
        return False, f"non-json /health body: {body[:120]}"
    if not payload.get("ready"):
        return False, f"server not ready: {body[:200]}"
    return True, "ready"


_URL = os.environ.get("MAYA2_URL", "").strip()
_READY, _REASON = _server_ready(_URL) if _URL else (False, "MAYA2_URL not set")

pytestmark = pytest.mark.skipif(not _READY, reason=f"live Maya2 unavailable: {_REASON}")


def test_voice_actor_live_against_real_maya2(tmp_path):
    cfg = ProjectConfig(voice_backend="maya2")
    project = Project.create(root_dir=str(tmp_path), brief="lighthouse live",
                              title="Maya2 Live E2E", config=cfg)

    project.write_artifact("screenwriter", {"fountain": SCREENPLAY})
    project.write_artifact("story_editor", {"revised_fountain": SCREENPLAY})
    scenes = [{"id": "s01",
               "heading": "INT. LIGHTHOUSE - LANTERN ROOM - NIGHT",
               "estimated_seconds": 15,
               "characters": ["JESS", "KAI"]}]
    project.write_artifact("breakdown", {"scenes": scenes})
    project.write_artifact("storyboard", {"scenes": {"s01": [
        {"id": "sh01", "subject": "JESS", "action": "clings to lantern"},
        {"id": "sh02", "subject": "KAI",  "action": "steps forward"},
    ]}})
    project.write_artifact("voice_cast", {"cast": {
        "JESS": {"maya2_voice": "Nivedita", "maya2_lang": "hi",
                 "voice": "en_US-amy-medium"},
        "KAI":  {"maya2_voice": "Abhay",    "maya2_lang": "en",
                 "voice": "en_US-lessac-medium"},
    }})

    out = agents.run_voice_actor(project)
    lines = out.get("lines") or []
    assert lines, "voice_actor returned no lines"

    non_maya = [ln for ln in lines if ln.get("backend") != "maya2"]
    assert not non_maya, f"{len(non_maya)}/{len(lines)} lines did not use maya2: " \
                          f"{[ln['backend'] for ln in non_maya[:3]]}"

    for ln in lines:
        wav_rel = ln.get("wav") or ""
        wav_abs = os.path.join(project.artifacts_dir, wav_rel)
        assert os.path.exists(wav_abs), f"wav missing: {wav_abs}"
        assert os.path.getsize(wav_abs) >= 44, f"wav too small: {wav_abs}"
        with open(wav_abs, "rb") as f:
            head = f.read(12)
        assert head[:4] == b"RIFF" and head[8:12] == b"WAVE", \
            f"wav header malformed: {head!r} in {wav_abs}"
        # Non-silent sample check so this test can't be satisfied by a stub.
        with wave.open(wav_abs, "rb") as w:
            raw = w.readframes(min(w.getnframes(), w.getframerate()))
        assert any(raw), f"wav is silent (likely a stub): {wav_abs}"

    jess = [ln for ln in lines if ln.get("speaker", "").upper().startswith("JESS")]
    kai = [ln for ln in lines if ln.get("speaker", "").upper().startswith("KAI")]
    assert jess, "no JESS lines"
    assert kai, "no KAI lines"

    total_dur = sum(float(ln.get("duration_sec") or 0) for ln in lines)
    assert total_dur > 0.0, "total voice duration is zero"
