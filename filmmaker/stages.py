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
    "shots",
    "editor",
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


# Order matters — this is the pipeline. T1 ships all eight; the last two
# (shots, editor) are executable stubs so the whole thing runs end to end.
STAGES: List[StageSpec] = [
    StageSpec("producer",       "Producer",         "Turns the brief into 3 loglines and picks one.",              gated_by_default=True),
    StageSpec("screenwriter",   "Screenwriter",     "Writes a first-draft screenplay from the chosen logline."),
    StageSpec("story_editor",   "Story Editor",     "Critiques and revises the draft."),
    StageSpec("breakdown",      "Script Breakdown", "Parses the screenplay into scenes + metadata."),
    StageSpec("storyboard",     "Storyboard",       "Expands each scene into a shot list."),
    StageSpec("cinematographer","Cinematographer",  "Assigns framing / lens / camera move / duration per shot.",   gated_by_default=True),
    StageSpec("shots",          "Shot Generator",   "T1: renders one SDXL keyframe per shot. T2 upgrades to video."),
    StageSpec("editor",         "Editor",           "T1: assembles keyframes into a slideshow. T2 adds dialogue + music."),
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
