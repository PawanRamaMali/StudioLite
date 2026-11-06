"""Tests for motion_brush.py (arrow strokes → motion prompt)."""
from __future__ import annotations

import pytest


def _s(x0, y0, x1, y1, label=""):
    from motion_brush import MotionStroke
    return MotionStroke(x0=x0, y0=y0, x1=x1, y1=y1, label=label)


# ---------------------------------------------------------------------------
# cardinal direction classification
# ---------------------------------------------------------------------------


def test_cardinal_right():
    from motion_brush import _cardinal
    assert _cardinal(1, 0) == "right"


def test_cardinal_left():
    from motion_brush import _cardinal
    assert _cardinal(-1, 0) == "left"


def test_cardinal_up():
    from motion_brush import _cardinal
    assert _cardinal(0, -1) == "up"


def test_cardinal_down():
    # Positive dy = down in image coords
    from motion_brush import _cardinal
    assert _cardinal(0, 1) == "down"


def test_cardinal_diagonals():
    from motion_brush import _cardinal
    assert _cardinal(1, 1) == "down-right"
    assert _cardinal(-1, 1) == "down-left"
    assert _cardinal(1, -1) == "up-right"
    assert _cardinal(-1, -1) == "up-left"


def test_cardinal_still_for_zero():
    from motion_brush import _cardinal
    assert _cardinal(0, 0) == "still"


# ---------------------------------------------------------------------------
# intensity bucketing
# ---------------------------------------------------------------------------


def test_intensity_subtle():
    from motion_brush import _distance_bucket
    assert _distance_bucket(0.05) == "subtle"


def test_intensity_moderate():
    from motion_brush import _distance_bucket
    assert _distance_bucket(0.15) == "moderate"


def test_intensity_strong():
    from motion_brush import _distance_bucket
    assert _distance_bucket(0.5) == "strong"


# ---------------------------------------------------------------------------
# region labeling
# ---------------------------------------------------------------------------


def test_region_center():
    from motion_brush import _region
    assert _region(0.5, 0.5) == "center"


def test_region_corners():
    from motion_brush import _region
    assert _region(0.1, 0.1) == "top-left"
    assert _region(0.9, 0.9) == "bottom-right"
    assert _region(0.1, 0.9) == "bottom-left"
    assert _region(0.9, 0.1) == "top-right"


def test_region_edges():
    from motion_brush import _region
    # Each edge midpoint resolves to its dominant row/column
    assert _region(0.5, 0.1) == "top-center"
    assert _region(0.5, 0.9) == "bottom-center"
    assert _region(0.1, 0.5) == "left"     # middle row + left col -> "left"
    assert _region(0.9, 0.5) == "right"    # middle row + right col -> "right"


# ---------------------------------------------------------------------------
# stroke interpretation
# ---------------------------------------------------------------------------


def test_interpret_stroke_pans_right():
    from motion_brush import interpret_stroke
    r = interpret_stroke(_s(0.2, 0.5, 0.8, 0.5, "camera"))
    assert r["direction"] == "right"
    assert r["intensity"] in ("moderate", "strong")
    assert r["label"] == "camera"


def test_interpret_clamps_oob_coords():
    from motion_brush import interpret_stroke
    r = interpret_stroke(_s(-0.5, 0.5, 1.5, 0.5))
    # Clamped to (0, 0.5) -> (1, 0.5) -> strong rightward
    assert r["direction"] == "right"
    assert r["intensity"] == "strong"


# ---------------------------------------------------------------------------
# prompt suffix
# ---------------------------------------------------------------------------


def test_format_suffix_empty_returns_empty_string():
    from motion_brush import format_prompt_suffix
    assert format_prompt_suffix([]) == ""


def test_format_suffix_single_stroke():
    from motion_brush import format_prompt_suffix
    out = format_prompt_suffix([_s(0.2, 0.5, 0.8, 0.5, "car")])
    assert "Motion:" in out
    assert "car" in out
    assert "right" in out
    assert out.endswith(".")


def test_format_suffix_multiple_strokes_joined():
    from motion_brush import format_prompt_suffix
    out = format_prompt_suffix([
        _s(0.2, 0.5, 0.8, 0.5, "car"),
        _s(0.5, 0.2, 0.5, 0.8, "camera"),
    ])
    assert "car" in out and "camera" in out
    assert "and" in out  # joined with "...; and ..."


def test_format_suffix_default_label_is_subject():
    from motion_brush import format_prompt_suffix
    out = format_prompt_suffix([_s(0.2, 0.5, 0.8, 0.5)])
    assert "subject" in out


def test_format_suffix_still_stroke_describes_still():
    from motion_brush import format_prompt_suffix
    out = format_prompt_suffix([_s(0.5, 0.5, 0.5, 0.5, "subject")])
    # Zero-length stroke -> "stays still"
    assert "still" in out


# ---------------------------------------------------------------------------
# dedup / filtering
# ---------------------------------------------------------------------------


def test_dedupe_drops_zero_length():
    from motion_brush import dedupe_strokes
    strokes = [_s(0.5, 0.5, 0.5, 0.5), _s(0.2, 0.5, 0.8, 0.5)]
    out = dedupe_strokes(strokes)
    assert len(out) == 1
    assert out[0].x0 == 0.2


def test_dedupe_removes_exact_duplicates():
    from motion_brush import dedupe_strokes
    strokes = [
        _s(0.2, 0.5, 0.8, 0.5, "car"),
        _s(0.2, 0.5, 0.8, 0.5, "car"),
        _s(0.2, 0.5, 0.8, 0.5, "Car"),  # same stroke, different case label
    ]
    out = dedupe_strokes(strokes)
    assert len(out) == 1


def test_dedupe_keeps_different_labels():
    from motion_brush import dedupe_strokes
    strokes = [
        _s(0.2, 0.5, 0.8, 0.5, "car"),
        _s(0.2, 0.5, 0.8, 0.5, "person"),
    ]
    out = dedupe_strokes(strokes)
    assert len(out) == 2


def test_dedupe_respects_min_length():
    from motion_brush import dedupe_strokes
    strokes = [_s(0.5, 0.5, 0.52, 0.5)]  # length 0.02
    assert dedupe_strokes(strokes, min_length=0.05) == []
    assert len(dedupe_strokes(strokes, min_length=0.01)) == 1
