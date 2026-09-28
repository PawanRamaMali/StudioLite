"""Regression cover for the shot-prompt / bio / story-editor bugs that
took multiple hour-long film renders to surface.

Each of these bugs made it into a shipped commit and only got noticed
after a full pipeline run wasted GPU time producing broken output. The
tests below lock in the fix so it stays fixed:

  1. _extract_character_bios must dedupe by uppercased key so `JESS` and
     `Jess` don't both count as separate characters and let a false
     Title-Case intro from action prose ("then to Jess, his eyes
     narrowing.") win over the real ALL-CAPS parenthetical intro.
     Fixed in 3ede650.

  2. _extract_character_bios must be the *primary* bio source (via
     run_shots' `bio_source = screenwriter.get("fountain") ...`) because
     the story editor's revised fountain sometimes strips the parenthetical
     intros entirely, leaving the extractor no ALL-CAPS bio to find and
     letting false Title-Case bios take over.  Fixed in dbb013b.

  3. _shot_prompt must emit a real style prefix so SDXL renders
     photorealistic output when `style="cinematic"` (or "photoreal"), not
     the stylized-illustration default.  Fixed in dad0a05.

  4. run_story_editor must reject a revised_fountain that lacks any scene
     slug or character cue - otherwise a smaller model that echoes the
     schema description ("the full revised screenplay in Fountain format")
     poisons the Breakdown stage into confabulating a completely different
     story.  Fixed in 00d432c.

  5. run_story_editor must pass the screenwriter draft through when it
     produces no revision at all, so the pipeline doesn't stall with an
     empty artifact downstream.  Fixed in 3ac60b9.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from filmmaker import agents


# ---------------------------------------------------------------------------
# 1. Bio extraction is case-insensitive
# ---------------------------------------------------------------------------

# A synthetic screenplay that contains BOTH a legitimate ALL-CAPS intro
# ("JESS (30s, practical jacket, salt-swept hair)") AND a later
# Title-Case action beat that looks superficially like a comma intro
# ("Jess, her eyes narrowing"). Before 3ede650 the extractor would let
# the Title-Case version overwrite (or double up on) the real one.
SCREENPLAY_WITH_TITLE_CASE_REPEAT = (
    "INT. LIGHTHOUSE - LANTERN ROOM - NIGHT\n"
    "\n"
    "The storm rages outside, threatening to consume the lighthouse. "
    "JESS (30s, practical jacket, salt-swept hair) clings to the lantern "
    "as it sways.\n"
    "\n"
    "Then to Jess, her eyes narrowing. She stares at the sea.\n"
    "\n"
    "    JESS\n"
    "    What are you doing here?\n"
)


def test_bio_extraction_dedupes_case_insensitively():
    """The real ALL-CAPS intro should win; the later Title-Case action
    beat must not create a second `Jess` entry or overwrite the good one."""
    bios = agents._extract_character_bios(SCREENPLAY_WITH_TITLE_CASE_REPEAT)
    # Exactly one key for JESS, spelled however the first match spelled it.
    jess_keys = [k for k in bios if k.upper() == "JESS"]
    assert len(jess_keys) == 1, f"Expected one JESS entry, got {jess_keys}"
    # The one that stuck must be the real intro, not the false action-line one.
    assert "practical jacket" in bios[jess_keys[0]]
    assert "eyes narrowing" not in bios[jess_keys[0]]


def test_bio_extraction_prefers_paren_intro_over_action_prose():
    """A properly-formed `NAME (age, wardrobe)` intro should always beat
    an action-prose comma pattern that happens to include the character's
    lowercase name."""
    script = (
        "INT. DINER - DAY\n"
        "\n"
        "MARGO (mid 40s, waitress apron, dyed-red bun) wipes down the counter.\n"
        "\n"
        "Later, Margo, tired but polite, refills a mug.\n"
    )
    bios = agents._extract_character_bios(script)
    margo_keys = [k for k in bios if k.upper() == "MARGO"]
    assert len(margo_keys) == 1
    assert "waitress apron" in bios[margo_keys[0]]


# ---------------------------------------------------------------------------
# 2. Shot prompt style prefixes
# ---------------------------------------------------------------------------

_SCENE = {
    "location": "diner-kitchen",
    "time_of_day": "day",
    "mood": "warm bustle",
}
_SHOT = {"description": "medium shot of Margo pouring coffee",
         "subject": "Margo", "action": "pours coffee"}
_DP = {"framing": "MS", "lens_mm": 35,
       "lighting": "warm key from window", "palette": "amber, cream"}


def test_shot_prompt_cinematic_style_uses_photorealistic_prefix():
    """style='cinematic' must produce a photorealistic prefix. Before
    dad0a05 the map only had 'photoreal' + 'stylized', so 'cinematic'
    silently fell to the stylized-illustration default."""
    out = agents._shot_prompt("cinematic", _SCENE, _SHOT, _DP)
    assert out.startswith("cinematic photorealistic still")
    assert "stylized illustrated" not in out


def test_shot_prompt_photoreal_style_uses_photorealistic_prefix():
    out = agents._shot_prompt("photoreal", _SCENE, _SHOT, _DP)
    assert "photorealistic" in out.split(",")[0]


def test_shot_prompt_clean_style_uses_editorial_prefix():
    out = agents._shot_prompt("clean", _SCENE, _SHOT, _DP)
    assert "editorial" in out or "clean" in out.split(",")[0]


def test_shot_prompt_unknown_style_falls_back_to_stylized():
    out = agents._shot_prompt("no-such-style", _SCENE, _SHOT, _DP)
    assert out.startswith("stylized illustrated frame")


def test_shot_prompt_bakes_matched_character_bio():
    """A shot whose subject names a character with a known bio should
    have that bio inserted before the scene metadata."""
    bios = {"MARGO": "mid 40s, waitress apron, dyed-red bun"}
    out = agents._shot_prompt("cinematic", _SCENE, _SHOT, _DP,
                              character_bios=bios)
    assert "MARGO: mid 40s, waitress apron, dyed-red bun" in out
    # The character line must land before the location boilerplate so
    # CLIP's 77-token window doesn't truncate the bio.
    assert out.index("MARGO:") < out.index("location:")


# ---------------------------------------------------------------------------
# 3. Story editor rejects fake / empty output
# ---------------------------------------------------------------------------

class _StubProject:
    """Minimum project stand-in for the story_editor stage."""

    def __init__(self, screenwriter_fountain: str):
        self._artifacts = {"screenwriter": {"fountain": screenwriter_fountain}}

        class _Cfg:
            llm_backend = "ollama"
            llm_model = "stub"
            per_stage: dict = {}
        class _Meta:
            config = _Cfg()
            id = "p"
            title = "t"
        self.meta = _Meta()
        self.dir = "."

    def read_artifact(self, key):
        return self._artifacts.get(key)


_REAL_SCREENPLAY = """\
INT. DINER - DAY

MARGO wipes down the counter as the morning rush winds down.

    MARGO
    Another refill, hon?

INT. KITCHEN - CONTINUOUS

Steam rises from the dishwasher.

    MARGO (O.S.)
    Coming up.
"""


def test_story_editor_rejects_schema_description_echo():
    """A model that echoes 'the full revised screenplay in Fountain
    format' as content must NOT poison the artifact; the screenwriter
    draft should pass through unchanged."""
    proj = _StubProject(_REAL_SCREENPLAY)

    fake_llm_reply = (
        '{"notes": [], '
        '"revised_fountain": "the full revised screenplay in Fountain format"}'
    )
    with patch.object(agents, "_chat", return_value=fake_llm_reply):
        out = agents.run_story_editor(proj)

    assert out["revised_fountain"] == _REAL_SCREENPLAY, (
        "Schema-echo output slipped through as if it were a real revision"
    )


def test_story_editor_rejects_prose_only_output():
    """A model that writes a paragraph of English (no scene slug, no
    cue) must not have its output accepted as a screenplay."""
    proj = _StubProject(_REAL_SCREENPLAY)
    fake_llm_reply = (
        '{"notes": [], '
        '"revised_fountain": "This screenplay features a diner where '
        'Margo, a waitress in her forties, serves customers over the '
        'course of a slow morning. The mood is warm and the pace unhurried."}'
    )
    with patch.object(agents, "_chat", return_value=fake_llm_reply):
        out = agents.run_story_editor(proj)
    assert out["revised_fountain"] == _REAL_SCREENPLAY


def test_story_editor_passes_screenwriter_draft_when_revised_is_missing():
    """Some models return `{"notes": [...]}` with no revised field at
    all. The stage must return the original draft, not an empty string
    (which would break every downstream stage silently)."""
    proj = _StubProject(_REAL_SCREENPLAY)
    fake_llm_reply = '{"notes": [{"fail": "pacing", "quote": "", "why": "too flat"}]}'
    with patch.object(agents, "_chat", return_value=fake_llm_reply):
        out = agents.run_story_editor(proj)
    assert out["revised_fountain"] == _REAL_SCREENPLAY


def test_story_editor_accepts_real_revised_screenplay():
    """Sanity check: a properly-formed revised fountain must pass
    through as the revision."""
    proj = _StubProject(_REAL_SCREENPLAY)
    revised = (
        "INT. DINER - DAY\n\n"
        "MARGO wipes the counter with brisk, precise strokes as she "
        "watches the last of the morning rush drift out.\n\n"
        "    MARGO\n    More coffee, hon?\n\n"
        "    CUSTOMER\n    Please. And an extra napkin.\n\n"
        "INT. KITCHEN - CONTINUOUS\n\n"
        "Steam hisses from the dishwasher as Margo shoulders through "
        "the swinging door with an empty carafe.\n\n"
        "    MARGO (O.S.)\n    Fresh pot up in one.\n"
    )
    import json as _json
    fake_llm_reply = _json.dumps({"notes": [], "revised_fountain": revised})
    with patch.object(agents, "_chat", return_value=fake_llm_reply):
        out = agents.run_story_editor(proj)
    # The stage strips outer whitespace when reading the field, so compare
    # on the stripped value.
    assert out["revised_fountain"] == revised.strip()


def test_story_editor_accepts_alternate_field_names_for_revision():
    """Smaller Ollama models sometimes name the field `screenplay` or
    `revised` instead of `revised_fountain`. All four aliases should be
    accepted (as long as the content itself is a real screenplay)."""
    proj = _StubProject(_REAL_SCREENPLAY)
    revised = (
        "INT. DINER - DAY\n\n"
        "MARGO refills a mug for the truck driver at the corner booth, "
        "her free hand already reaching for the pot behind her.\n\n"
        "    MARGO\n    More coffee, sweetheart?\n\n"
        "    TRUCK DRIVER\n    You are a saint. Just a little.\n\n"
        "INT. KITCHEN - LATER\n\n"
        "Steam curls off the sink as Margo washes a stack of plates.\n\n"
        "    MARGO (V.O.)\n    Twenty years and the pot never stops.\n"
    )
    for key in ("revised_fountain", "revised_screenplay", "revised",
                "screenplay"):
        import json as _json
        fake = _json.dumps({"notes": [], key: revised})
        with patch.object(agents, "_chat", return_value=fake):
            out = agents.run_story_editor(proj)
        assert out["revised_fountain"] == revised.strip(), (
            f"Field alias `{key}` did not round-trip"
        )
