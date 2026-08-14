"""
Film Studio — end-to-end short-film pipeline of specialty AI agents.

Public surface used by api_server:
    from filmmaker import registry, projects, orchestrator
    from filmmaker.project import Project
    from filmmaker.stages import STAGES, StageKey
"""

from . import projects, orchestrator, registry  # noqa: F401
from .project import Project  # noqa: F401
from .stages import STAGES, StageKey  # noqa: F401
