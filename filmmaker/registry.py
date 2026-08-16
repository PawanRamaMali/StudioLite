"""
Stage → runnable-function registry. Keeps the orchestrator ignorant of
agent module structure — adding a stage means one line here.
"""

from __future__ import annotations

from typing import Callable, Dict

from . import agents
from .project import Project
from .stages import StageKey


StageRunner = Callable[[Project], Dict]

RUNNERS: Dict[StageKey, StageRunner] = {
    "producer":        agents.run_producer,
    "screenwriter":    agents.run_screenwriter,
    "story_editor":    agents.run_story_editor,
    "breakdown":       agents.run_breakdown,
    "storyboard":      agents.run_storyboard,
    "cinematographer": agents.run_cinematographer,
    "voice_cast":      agents.run_voice_cast,
    "voice_actor":     agents.run_voice_actor,
    "shots":           agents.run_shots,
    "editor":          agents.run_editor,
    "composer":        agents.run_composer,
    "mixer":           agents.run_mixer,
}


def get_runner(key: StageKey) -> StageRunner:
    if key not in RUNNERS:
        raise KeyError(f"No runner registered for stage {key!r}")
    return RUNNERS[key]
