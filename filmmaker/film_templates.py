"""Starter templates for the Film Studio pipeline.

A template packages up a brief skeleton and the config knobs that
usually go with a particular kind of film — a 1-minute short doesn't
render at the same quality settings as a 30-second product demo. The
UI can show the catalog on the "new project" screen so a user picks a
starting point instead of staring at a blank config form.

Templates are code, not data — the catalog rarely changes and shipping
it as a Python module keeps everything type-checked. Adding one is a
matter of appending a ``FilmTemplate`` below.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List


@dataclass(frozen=True)
class FilmTemplate:
    id: str                     # kebab-case slug — stable public identifier
    name: str                   # display name
    description: str            # one-liner shown on the tile
    category: str               # "narrative" | "explainer" | "promo" | "experimental"
    target_minutes: float
    style: str                  # matches ProjectConfig.style
    quality: str                # "draft" | "standard" | "quality"
    sdxl_variant: str = "turbo"
    motion_backend: str = "auto"
    voice_backend: str = "piper"
    music_backend: str = "musicgen"
    sample_brief: str = ""      # baseline brief the user can edit before creating
    brief_placeholder: str = "" # UI hint text
    tags: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


# --- Catalog ---------------------------------------------------------------

_TEMPLATES: List[FilmTemplate] = [
    FilmTemplate(
        id="short-story",
        name="Short Story",
        description="1-minute narrative short with three beats.",
        category="narrative",
        target_minutes=1.0,
        style="stylized",
        quality="standard",
        sample_brief=(
            "A weary lighthouse keeper meets a stranger who claims to know "
            "why the light has been flickering. They talk for one long "
            "night, and by dawn one of them is gone."
        ),
        brief_placeholder="A one-paragraph pitch — who, where, what changes.",
        tags=["narrative", "short"],
    ),
    FilmTemplate(
        id="explainer",
        name="Explainer",
        description="2-minute explainer with narration and clean visuals.",
        category="explainer",
        target_minutes=2.0,
        style="clean",
        quality="standard",
        voice_backend="indextts2",
        motion_backend="animatediff",
        sample_brief=(
            "Explain how a solid-state battery works in plain language, "
            "aimed at a curious high-school audience. Cover the anode, "
            "electrolyte, cathode, and why 'solid' matters for safety."
        ),
        brief_placeholder="What are you explaining, and who's watching?",
        tags=["explainer", "education"],
    ),
    FilmTemplate(
        id="teaser-trailer",
        name="Teaser Trailer",
        description="30-second high-contrast trailer with punchy cuts.",
        category="promo",
        target_minutes=0.5,
        style="cinematic",
        quality="quality",
        sdxl_variant="base",
        motion_backend="wan22",
        sample_brief=(
            "A teaser for a heist thriller set in a rain-soaked Tokyo "
            "subway. Three shots, one line of dialogue, a title card."
        ),
        brief_placeholder="Genre, one location, one hook.",
        tags=["promo", "trailer"],
    ),
    FilmTemplate(
        id="product-demo",
        name="Product Demo",
        description="45-second product walkthrough with voiceover.",
        category="promo",
        target_minutes=0.75,
        style="clean",
        quality="standard",
        voice_backend="indextts2",
        sample_brief=(
            "A 45-second walkthrough of a note-taking app that turns "
            "voice memos into Kanban cards. Show upload, transcript, "
            "auto-tagged card, and the final board view."
        ),
        brief_placeholder="What does the product do? Which three moments matter?",
        tags=["promo", "product"],
    ),
    FilmTemplate(
        id="draft-scene",
        name="Draft Scene",
        description="Fast 90-second scene at draft quality — rough cut only.",
        category="experimental",
        target_minutes=1.5,
        style="stylized",
        quality="draft",
        sdxl_variant="turbo",
        motion_backend="animatediff",
        sample_brief=(
            "A quick sketch of a scene — draft output for review before "
            "committing to full-quality render."
        ),
        brief_placeholder="Rough idea — this is a fast draft, refine later.",
        tags=["draft", "iteration"],
    ),
]


def list_templates() -> List[FilmTemplate]:
    """Return the catalog. Callers must not mutate the entries."""
    return list(_TEMPLATES)


def get_template(template_id: str) -> FilmTemplate:
    """Look up a template by id. Raises KeyError on miss so callers know
    exactly which id was bad rather than getting a silent fallback."""
    for t in _TEMPLATES:
        if t.id == template_id:
            return t
    raise KeyError(f"Unknown template id: {template_id!r}")


def template_config(template_id: str) -> Dict[str, Any]:
    """Return the config dict a ``Project.create`` call needs, populated
    from the named template. UI code merges any user overrides on top of
    this before posting to the create endpoint."""
    t = get_template(template_id)
    return {
        "style": t.style,
        "quality": t.quality,
        "target_minutes": t.target_minutes,
        "sdxl_variant": t.sdxl_variant,
        "motion_backend": t.motion_backend,
        "voice_backend": t.voice_backend,
        "music_backend": t.music_backend,
    }
