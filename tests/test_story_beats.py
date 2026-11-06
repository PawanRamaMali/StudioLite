"""Tests for story_beats.py (pure beat-template resolver)."""
from __future__ import annotations

import pytest


def test_list_templates_contains_every_builtin():
    from story_beats import list_templates
    ids = {t["id"] for t in list_templates()}
    assert {"hook_build_payoff", "question_answer", "problem_solution",
            "list_countdown", "before_after", "narrative_arc"} <= ids


def test_resolve_template_known():
    from story_beats import resolve_template
    t = resolve_template("hook_build_payoff")
    assert t is not None and t.id == "hook_build_payoff"


def test_resolve_template_unknown_returns_none():
    from story_beats import resolve_template
    assert resolve_template("") is None
    assert resolve_template(None) is None
    assert resolve_template("made-up") is None


def test_resolve_template_case_insensitive():
    from story_beats import resolve_template
    assert resolve_template("HOOK_BUILD_PAYOFF") is not None
    assert resolve_template("Before_After") is not None


def test_build_scene_stubs_sums_to_target_duration():
    from story_beats import build_scene_stubs
    stubs = build_scene_stubs("hook_build_payoff", 30.0)
    total = sum(s["duration"] for s in stubs)
    assert abs(total - 30.0) < 0.1


def test_build_scene_stubs_for_every_template():
    from story_beats import TEMPLATES, build_scene_stubs
    for tid, tmpl in TEMPLATES.items():
        stubs = build_scene_stubs(tid, 30.0)
        assert len(stubs) == len(tmpl.beats)
        total = sum(s["duration"] for s in stubs)
        assert abs(total - 30.0) < 0.1


def test_build_scene_stubs_respects_min_floor():
    # 5-second total, 6-beat template - every beat still >= min floor
    from story_beats import build_scene_stubs
    stubs = build_scene_stubs("list_countdown", 5.0, min_scene_s=0.8)
    assert all(s["duration"] >= 0.8 for s in stubs)


def test_build_scene_stubs_unknown_template_returns_empty():
    from story_beats import build_scene_stubs
    assert build_scene_stubs("nonexistent", 30.0) == []


def test_build_scene_stubs_hook_is_first_beat():
    from story_beats import build_scene_stubs
    stubs = build_scene_stubs("hook_build_payoff", 30.0)
    assert stubs[0]["role"] == "hook"


def test_build_scene_stubs_hook_is_short_in_hook_build_payoff():
    # The hook is intentionally brief; it should sit well below the build beats
    # so a 60s short's hook comes out under 10 seconds.
    from story_beats import build_scene_stubs
    stubs = build_scene_stubs("hook_build_payoff", 60.0)
    hook = stubs[0]
    assert hook["role"] == "hook"
    assert hook["duration"] < 10.0
    # And strictly shorter than the build beats
    build_durations = [s["duration"] for s in stubs if s["role"] == "build"]
    assert all(hook["duration"] < b for b in build_durations)


def test_build_scene_stubs_floor_never_below_1_point_2():
    # Very short total - floor should kick in
    from story_beats import build_scene_stubs
    stubs = build_scene_stubs("list_countdown", 8.0)
    assert all(s["duration"] >= 1.2 for s in stubs)


def test_build_scene_stubs_negative_duration_safe():
    from story_beats import build_scene_stubs
    stubs = build_scene_stubs("hook_build_payoff", -5.0)
    assert all(s["duration"] > 0 for s in stubs)
    total = sum(s["duration"] for s in stubs)
    assert total >= 1.0  # resolves to minimum of 1s


def test_build_scene_stubs_has_directive_strings():
    from story_beats import build_scene_stubs
    stubs = build_scene_stubs("question_answer", 30.0)
    assert all(s["directive"] for s in stubs)
    assert all(len(s["directive"]) > 10 for s in stubs)


def test_pacing_hint_returns_summary():
    from story_beats import pacing_hint
    hint = pacing_hint("hook_build_payoff")
    assert "Hook" in hint or "hook" in hint.lower()
    assert len(hint) > 50


def test_pacing_hint_unknown_returns_empty():
    from story_beats import pacing_hint
    assert pacing_hint("") == ""
    assert pacing_hint("nope") == ""


def test_list_templates_payload_keys():
    from story_beats import list_templates
    for t in list_templates():
        for k in ("id", "label", "description", "beat_count",
                  "min_duration_s", "max_duration_s"):
            assert k in t
