"""Tests for ``_kenburns_expr`` in ``filmmaker.agents``.

The Ken Burns helper translates the Cinematographer's ``camera_move`` field
into a zoompan filter clause. The mapping is the entire shape of #58's
polish pass, so lock it with explicit tests - a future refactor that
accidentally reroutes "pan" to "dolly" would otherwise sail through the
end-to-end pipeline tests (they only check that *some* clip was rendered).
"""
from __future__ import annotations

import pytest


@pytest.mark.parametrize("move, kind_marker", [
    # pan -> horizontal drift across the frame
    ("pan",      "x='(iw-iw/zoom)*on/"),
    # dolly -> continuous zoom-in toward 1.30
    ("dolly",    "z='min(zoom+0.0015,1.30)'"),
    # crane -> vertical drift
    ("crane",    "y='(ih-ih/zoom)*(1-on/"),
    # whip -> faster horizontal drift (min(on*2/span, 1))
    ("whip",     "x='(iw-iw/zoom)*min(on*2/"),
    # handheld -> sinusoidal jitter on both axes
    ("handheld", "sin(on/"),
    # static -> slow push-in (small z-step toward 1.08)
    ("static",   "z='min(zoom+0.0004,1.08)'"),
])
def test_kenburns_expr_picks_right_motion_for_camera_move(move, kind_marker):
    from filmmaker.agents import _kenburns_expr
    expr = _kenburns_expr(move, frames=120, fps=30)
    assert kind_marker in expr, expr


def test_kenburns_expr_unknown_move_falls_back_to_static():
    from filmmaker.agents import _kenburns_expr
    expr = _kenburns_expr("tracking", frames=120, fps=30)
    # Unknown values must not raise or produce an empty graph; they fall
    # back to the static-push-in form.
    assert "z='min(zoom+0.0004,1.08)'" in expr


def test_kenburns_expr_empty_move_treated_as_static():
    from filmmaker.agents import _kenburns_expr
    expr = _kenburns_expr("", frames=60, fps=30)
    assert "z='min(zoom+0.0004,1.08)'" in expr


def test_kenburns_expr_speed_scales_with_duration():
    """A longer shot (more frames) must stretch the pan expression's
    denominator so the motion still completes exactly at the cut, not
    midway through or past the end."""
    from filmmaker.agents import _kenburns_expr
    short = _kenburns_expr("pan", frames=30, fps=30)   # 1 sec
    long  = _kenburns_expr("pan", frames=300, fps=30)  # 10 sec
    # span = frames - 1, embedded into the x expression denominator.
    assert "on/29" in short
    assert "on/299" in long


def test_kenburns_expr_duration_clause_matches_frames():
    from filmmaker.agents import _kenburns_expr
    expr = _kenburns_expr("dolly", frames=90, fps=30)
    # zoompan's d= token drives how many output frames the ramp spans;
    # it must equal the shot's total frame count.
    assert expr.rstrip().endswith("d=90")


def test_kenburns_expr_case_insensitive():
    from filmmaker.agents import _kenburns_expr
    low = _kenburns_expr("pan", frames=60, fps=30)
    up = _kenburns_expr("PAN", frames=60, fps=30)
    mix = _kenburns_expr("Pan", frames=60, fps=30)
    assert low == up == mix


def test_kenburns_expr_single_frame_clip_is_safe():
    """When a shot is exactly 1 frame long, the span denominator must not
    be zero - the function clamps it to 1."""
    from filmmaker.agents import _kenburns_expr
    expr = _kenburns_expr("pan", frames=1, fps=30)
    # span = max(frames-1, 1) = 1; the pan x expression references it.
    assert "on/1" in expr
