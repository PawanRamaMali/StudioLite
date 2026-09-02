"""
Stage → runnable-function registry. Keeps the orchestrator ignorant of
agent module structure — adding a stage means one line here.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

from . import agents
from .project import Project
from .stages import QualityReport, StageKey


StageRunner = Callable[[Project], Dict]
Verifier = Callable[[Project, Dict], QualityReport]

RUNNERS: Dict[StageKey, StageRunner] = {
    "producer":            agents.run_producer,
    "screenwriter":        agents.run_screenwriter,
    "story_editor":        agents.run_story_editor,
    "breakdown":           agents.run_breakdown,
    "storyboard":          agents.run_storyboard,
    "cinematographer":     agents.run_cinematographer,
    "voice_cast":          agents.run_voice_cast,
    "character_portraits": agents.run_character_portraits,
    "voice_actor":         agents.run_voice_actor,
    "shots":               agents.run_shots,
    "motion_shots":        agents.run_motion_shots,
    "editor":              agents.run_editor,
    "ambient":             agents.run_ambient,
    "composer":            agents.run_composer,
    "mixer":               agents.run_mixer,
    "colorist":            agents.run_colorist,
    "titles":              agents.run_titles,
}


# Optional per-stage quality verifiers. A stage without an entry here
# behaves exactly as before — one shot, no retry loop. Stages that opt
# in get up to two extra attempts if the first output fails verification,
# with the report's hints fed back to the runner as reviewer notes.
VERIFIERS: Dict[StageKey, Verifier] = {
    "producer":     agents.verify_producer,
    "screenwriter": agents.verify_screenwriter,
    "story_editor": agents.verify_story_editor,
    "storyboard":   agents.verify_storyboard,
}


def get_runner(key: StageKey) -> StageRunner:
    if key not in RUNNERS:
        raise KeyError(f"No runner registered for stage {key!r}")
    return RUNNERS[key]


def get_verifier(key: StageKey) -> Optional[Verifier]:
    return VERIFIERS.get(key)
