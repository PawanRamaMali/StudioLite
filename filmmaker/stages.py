"""
Stage registry and state model for the film pipeline.

Adding a new stage means: (1) append its key to STAGES in dependency order,
(2) write the agent module and register it in filmmaker/registry.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal


StageKey = Literal[
    "producer",
    "screenwriter",
    "story_editor",
    "breakdown",
    "storyboard",
    "cinematographer",
    "voice_cast",
    "character_portraits",
    "voice_actor",
    "shots",
    "editor",
    "composer",
    "mixer",
    "colorist",
    "titles",
]

StageStatus = Literal[
    "pending",       # not started
    "running",       # currently executing
    "paused",        # user paused mid-run; will resume from checkpoint
    "done",          # completed successfully
    "failed",        # last run errored; can be retried
    "stale",         # completed once but upstream artifact changed
    "needs_review",  # completed but sitting at a gate awaiting user approval
]


@dataclass(frozen=True)
class StageSpec:
    key: StageKey
    label: str
    description: str
    # A stage is `gated` if the pipeline halts on it by default and waits for
    # the user to approve/edit before advancing. User can toggle at run time.
    gated_by_default: bool = False


# Order matters — this is the pipeline. Dialogue voices and music score fold
# into the final cut via the mixer stage, so the film comes out with sound.
STAGES: List[StageSpec] = [
    StageSpec("producer",           "Producer",           "Turns the brief into 3 loglines and picks one.",              gated_by_default=True),
    StageSpec("screenwriter",       "Screenwriter",       "Writes a first-draft screenplay from the chosen logline."),
    StageSpec("story_editor",       "Story Editor",       "Critiques and revises the draft."),
    StageSpec("breakdown",          "Script Breakdown",   "Parses the screenplay into scenes + metadata."),
    StageSpec("storyboard",         "Storyboard",         "Expands each scene into a shot list."),
    StageSpec("cinematographer",    "Cinematographer",    "Assigns framing / lens / camera move / duration per shot.",   gated_by_default=True),
    StageSpec("voice_cast",         "Voice Casting",      "Assigns a Piper voice to each named character."),
    StageSpec("character_portraits","Character Portraits","Renders one canonical portrait per named character; IP-Adapter reference for shot renders."),
    StageSpec("voice_actor",        "Voice Actor",        "Piper-synthesizes each dialogue line as a wav clip."),
    StageSpec("shots",              "Shot Generator",     "Renders one SDXL keyframe per shot, conditioned on the character portrait when the shot features a named character."),
    StageSpec("editor",             "Editor",             "Assembles keyframes into the silent cut."),
    StageSpec("composer",           "Composer",           "LLM writes a score prompt; MusicGen renders one track for the cut."),
    StageSpec("mixer",              "Mixer",              "Muxes video + dialogue + score with sidechain ducking under speech."),
    StageSpec("colorist",           "Colorist",           "Applies per-scene color grading LUT via ffmpeg."),
    StageSpec("titles",             "Titles & Credits",   "Prepends title card and appends end credits."),
]

STAGE_KEYS: List[StageKey] = [s.key for s in STAGES]


def stage_index(key: StageKey) -> int:
    for i, s in enumerate(STAGES):
        if s.key == key:
            return i
    raise KeyError(f"Unknown stage: {key!r}")


def downstream_of(key: StageKey) -> List[StageKey]:
    """Return every stage that runs after `key` (exclusive)."""
    i = stage_index(key)
    return [s.key for s in STAGES[i + 1:]]
