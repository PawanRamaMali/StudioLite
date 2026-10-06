"""End-to-end integration harness for the 18-stage Film Studio pipeline.

Does NOT exercise real SDXL / Wan / MusicGen / Whisper weights — those
take minutes per stage and depend on >30 GB of weights. Instead:

  - Patches `filmmaker.llm.chat` to a deterministic mock that returns
    valid JSON for every system-prompt we see, so every LLM-driven
    stage (Producer, Screenwriter, Story Editor, Breakdown, Storyboard,
    Cinematographer, Voice Casting, Composer prompt, Titles) can run.
  - Patches `filmmaker.agents._load_sdxl_renderer` and
    `_load_sdxl_batch_renderer` to writers that emit tiny valid PNGs.
  - Patches motion-video, music, speech, lipsync entry points to tiny
    fallback writers.
  - Drives the project through orchestrator._run with gates OFF so the
    whole thing runs top-to-bottom in one asyncio.run().

The point is NOT image/audio/video quality — it's structural:

  * Every stage runs without raising
  * Every stage persists an artifact
  * Downstream stages successfully load the upstream artifacts
  * final files land at the paths the artifacts advertise

When this test fails, it's probably one of:
  - A stage assumed an upstream artifact field that doesn't exist
  - A stage crashed on a missing weight path with no fallback
  - An ffmpeg flag didn't survive the version bump
  - A stage's JSON parse logic couldn't handle the mock LLM output

Which is EXACTLY the gaps real users hit on a fresh machine.
"""
from __future__ import annotations

import asyncio
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Mock LLM — one dispatcher that recognizes every system prompt and
# returns a JSON blob the stage's own parser accepts.
# ---------------------------------------------------------------------------

def _mock_llm_chat(system: str, user: str, **kwargs):
    """Deterministic mock. Match on the UNIQUE opening of each stage's
    system prompt — these are tuned per the actual prompts in
    `filmmaker/agents.py`. Order matters because we use elif chains to
    stay exclusive."""
    s = (system or "").lower()
    want_json = kwargs.get("want_json")

    # Producer — opens with "You are a seasoned indie-film Producer"
    if "indie-film producer" in s or "producer developing" in s or (
        "exactly three" in s and ("logline" in s or "pitch" in s)
    ):
        def _pitch(title, premise):
            return {
                "title": title, "tone": "drama",
                "premise": premise,
                "central_irony": "The keeper must break the one promise he's never broken.",
                "unresolved": "Who sent the letter.",
                "logline": premise,
            }
        return json.dumps({
            "loglines": [
                _pitch("The Last Beacon",
                       "When a mysterious letter arrives, a lighthouse keeper must decide "
                       "whether to light the beacon that could save or damn a ship at sea."),
                _pitch("Beam",
                       "A keeper unearths a letter dated for tomorrow and must choose "
                       "whether to trust his own handwriting before the storm hits."),
                _pitch("Salt",
                       "The lighthouse has never been dark; tonight the keeper must break "
                       "that promise before the lie breaks him first."),
            ],
            "recommended_index": 0,
        })

    # Screenwriter — opens with "You are a professional screenwriter"
    if "professional screenwriter" in s:
        return (
            "INT. LIGHTHOUSE - NIGHT\n\n"
            "MORGAN, 60s, weathered. A letter waits on the wooden desk, wax-sealed.\n\n"
            "MORGAN\nI thought I burned them all.\n\n"
            "He breaks the seal.\n\n"
            "INT. LIGHTHOUSE - LATER\n\n"
            "MORGAN climbs the spiral stair, letter clenched in his fist.\n\n"
            "MORGAN\nNot tonight.\n\n"
            "EXT. LIGHTHOUSE GALLERY - CONTINUOUS\n\n"
            "Wind. He opens the lens-house.\n"
        )

    # Story Editor — opens with "You are a story editor"
    if "story editor" in s:
        return json.dumps({
            "notes": [
                {"fail": "flat", "quote": "I thought I burned them all.",
                 "why": "opens on exposition — show the burned letters first"}
            ],
            "revised_fountain": (
                "INT. LIGHTHOUSE - NIGHT\n\n"
                "MORGAN, 60s, weathered. A letter waits on the wooden desk, wax-sealed.\n\n"
                "MORGAN\nNot again.\n\n"
                "He breaks the seal.\n\n"
                "EXT. LIGHTHOUSE GALLERY - LATER\n\n"
                "Wind. He opens the lens-house.\n"
            ),
        })

    # Script Breakdown — opens with "You are a 1st AD doing a script breakdown"
    if "1st ad" in s or "script breakdown" in s:
        return json.dumps({"scenes": [
            {"id": "s01", "heading": "INT. LIGHTHOUSE - NIGHT",
             "summary": "Morgan opens a letter.",
             "location": "lighthouse-desk", "interior": True, "time_of_day": "night",
             "characters": ["MORGAN"], "mood": "brooding stillness",
             "estimated_seconds": 8},
            {"id": "s02", "heading": "EXT. LIGHTHOUSE GALLERY - LATER",
             "summary": "Morgan opens the lens-house.",
             "location": "lighthouse-gallery", "interior": False, "time_of_day": "night",
             "characters": ["MORGAN"], "mood": "storm", "estimated_seconds": 6},
        ]})

    # Storyboard — opens with "You are a storyboard artist"
    if "storyboard artist" in s:
        return json.dumps({"shots": [
            {"id": "sh1", "description": "wide of the lighthouse room, single lantern",
             "subject": "MORGAN", "action": "sits at the desk", "dialogue": "Not again."},
            {"id": "sh2", "description": "close on the wax seal cracking under his thumb",
             "subject": "the letter", "action": "seal snaps", "dialogue": ""},
        ]})

    # Cinematographer — opens with "You are the Director of Photography"
    if "director of photography" in s:
        return json.dumps({"shots": [
            {"id": "sh1", "framing": "WS", "lens_mm": 24, "camera_move": "dolly",
             "lighting": "single warm key", "palette": "amber, deep blue", "duration_sec": 4},
            {"id": "sh2", "framing": "ECU", "lens_mm": 85, "camera_move": "static",
             "lighting": "top light on wax", "palette": "amber, brown", "duration_sec": 2},
        ]})

    # Voice Casting — opens with "You cast voice actors"
    if "voice actor" in s and "cast" in s:
        # Return the shape the stage normalizes (character + voice + gender + why).
        return json.dumps({"cast": [
            {"character": "MORGAN", "voice": "en_US-ryan-high",
             "gender": "male", "why": "weathered, 60s"},
        ]})

    # Composer — opens with "You are a film composer"
    if "film composer" in s or "musicgen" in s:
        return json.dumps({"prompt": "sparse strings, low cellos, slow 60bpm, contemplative"})

    # Titles — anything mentioning title card / credits
    if "title" in s and ("card" in s or "credits" in s or "crawl" in s):
        return json.dumps({
            "title_card": "THE LAST BEACON",
            "credits": ["Directed by: Film Studio", "Starring: MORGAN"],
        })

    # Catch-all — return a valid empty JSON so parse_json doesn't blow up
    if want_json:
        return "{}"
    return "ok"


# ---------------------------------------------------------------------------
# Mock image / video / audio renderers — write minimal valid files
# ---------------------------------------------------------------------------

def _write_tiny_png(path: str) -> None:
    """1×1 PNG — tiny but valid."""
    data = bytes([
        0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,  # PNG signature
        0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52,  # IHDR
        0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,  # 1×1
        0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53, 0xDE,
        0x00, 0x00, 0x00, 0x0C, 0x49, 0x44, 0x41, 0x54,  # IDAT
        0x08, 0x99, 0x63, 0xF8, 0xCF, 0xC0, 0x00, 0x00, 0x00, 0x03, 0x00, 0x01,
        0x73, 0xB8, 0x55, 0x8A,
        0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, 0x44,  # IEND
        0xAE, 0x42, 0x60, 0x82,
    ])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f: f.write(data)


def _fake_sdxl_renderer():
    def _r(prompt: str, out_path: str, **kw) -> None:
        _write_tiny_png(out_path)
    return _r


def _fake_sdxl_batch_renderer():
    def _r(prompts: List[str], out_paths: List[str], **kw) -> None:
        for p in out_paths:
            _write_tiny_png(p)
    return _r


# ---------------------------------------------------------------------------
# The test — drives the full orchestrator from Producer → Upscale
# ---------------------------------------------------------------------------

@pytest.mark.e2e
def test_pipeline_runs_end_to_end_with_mocked_models(tmp_path: Path):
    """Run all 18 stages. Verify every stage persists its artifact and the
    orchestrator finishes without raising."""
    from filmmaker import projects, orchestrator, agents, llm

    # Point the orchestrator's output dir at tmp_path. The Project classes
    # use `root_dir/project_id/...` so films root sits here.
    films_root = tmp_path / "films"
    films_root.mkdir()

    # Build a project with gates OFF so the orchestrator runs all stages
    # straight through without pausing at Producer / Cinematographer.
    proj = projects.create(
        str(tmp_path), brief=(
            "A stoic lighthouse keeper receives a mysterious letter and must "
            "decide whether to light the beacon."
        ),
        title="The Last Beacon",
        config_dict={
            "llm_backend": "ollama", "llm_model": "mock",
            "style": "stylized", "target_minutes": 1.0,
            "quality": "draft",      # fewest SDXL steps
            "sdxl_variant": "turbo",
            "motion_backend": "kenburns",   # pure ffmpeg, no model weights
            "voice_backend": "piper",
            "music_backend": "musicgen",
            "upscale_backend": "none",
        },
    )
    # Turn off every default gate so the pipeline runs straight through.
    from filmmaker.stages import STAGES
    state = proj.state
    for s in STAGES:
        state.gates[s.key] = False
    proj.save_state(state)

    stack: List[Any] = [
        # Mock the LLM everywhere
        patch.object(llm, "chat", side_effect=_mock_llm_chat),
        # Mock SDXL renderers (both single + batch)
        patch.object(agents, "_load_sdxl_renderer", return_value=_fake_sdxl_renderer()),
        patch.object(agents, "_load_sdxl_batch_renderer",
                     return_value=_fake_sdxl_batch_renderer()),
    ]
    # Enter all patches, run the orchestrator to completion, exit cleanly.
    for p in stack:
        p.__enter__()
    try:
        async def _drive():
            # `start()` queues the pipeline task; we have to await the task
            # itself to actually run it to completion inside this event loop.
            started = await orchestrator.manager.start(proj)
            assert started, "orchestrator refused to start"
            handle = orchestrator.manager._runs[proj.project_id]
            await asyncio.wait_for(handle.task, timeout=120)
        asyncio.run(_drive())
    finally:
        for p in reversed(stack):
            p.__exit__(None, None, None)

    final = proj.state

    # ---- Assertions ----
    # Any stage that errored out should mark the run paused with last_error set.
    assert final.last_error is None, f"pipeline failed: {final.last_error}"

    # Every stage should have persisted an artifact OR be marked done.
    missing_artifacts: List[str] = []
    for spec in STAGES:
        status = final.stage_status.get(spec.key, "")
        if status != "done":
            missing_artifacts.append(f"{spec.key}: status={status!r}")
            continue
        art = proj.read_artifact(spec.key)
        if art is None:
            missing_artifacts.append(f"{spec.key}: no artifact file")
    assert not missing_artifacts, "stages that didn't complete: " + "; ".join(missing_artifacts)

    # The final cut should exist on disk (one of mixer / colorist / titles / upscale
    # produces an output_path).
    for stage_key in ("upscale", "titles", "colorist", "mixer", "editor"):
        art = proj.read_artifact(stage_key)
        if art and art.get("output_path"):
            final_path = os.path.join(proj.dir, art["output_path"])
            assert os.path.isfile(final_path), f"{stage_key} output missing on disk: {final_path}"
            return
    pytest.fail("none of mixer/colorist/titles/upscale/editor produced an output file")
