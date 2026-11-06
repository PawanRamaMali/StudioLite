"""Tests for video_formats.py (platform preset resolver + dim adapter)."""
from __future__ import annotations

import pytest


def test_resolve_format_known():
    from video_formats import resolve_format
    p = resolve_format("reels")
    assert p.id == "reels"
    assert p.aspect == "9:16"
    assert (p.width, p.height) == (1080, 1920)


def test_resolve_format_unknown_falls_back_to_reels():
    from video_formats import resolve_format
    assert resolve_format("made-up-format").id == "reels"
    assert resolve_format(None).id == "reels"
    assert resolve_format("").id == "reels"


def test_resolve_format_case_insensitive():
    from video_formats import resolve_format
    assert resolve_format("REELS").id == "reels"
    assert resolve_format("Shorts").id == "shorts"


def test_all_presets_contains_every_platform():
    from video_formats import all_presets
    ids = {p["id"] for p in all_presets()}
    assert {"reels", "shorts", "tiktok", "square", "feed_portrait", "landscape"} <= ids


def test_dims_for_engine_rounds_down_to_stride():
    from video_formats import dims_for_engine, resolve_format
    preset = resolve_format("reels")  # gen 544x960
    w, h = dims_for_engine("wan22_14b_gguf", preset)   # stride 16
    assert w == 544 and h == 960
    assert w % 16 == 0 and h % 16 == 0


def test_dims_for_engine_ltx_rounds_to_32():
    from video_formats import dims_for_engine, FormatPreset
    preset = FormatPreset(
        id="x", label="x", aspect="9:16",
        width=1080, height=1920, gen_width=550, gen_height=970,
        max_duration_s=60, target_fps=30, max_bitrate_mbps=10, audio_lufs=-14,
        safe_top_pct=0, safe_bottom_pct=0,
    )
    w, h = dims_for_engine("ltx23_distilled_gguf", preset)
    assert w % 32 == 0 and h % 32 == 0
    assert w == 544  # 550 // 32 * 32
    assert h == 960


def test_dims_for_engine_unknown_defaults_to_16():
    from video_formats import dims_for_engine, resolve_format
    preset = resolve_format("square")
    w, h = dims_for_engine("totally_unknown_engine", preset)
    assert w % 16 == 0 and h % 16 == 0


def test_dims_for_engine_never_returns_zero():
    from video_formats import dims_for_engine, FormatPreset
    preset = FormatPreset(
        id="tiny", label="tiny", aspect="9:16",
        width=10, height=10, gen_width=5, gen_height=5,
        max_duration_s=1, target_fps=30, max_bitrate_mbps=1, audio_lufs=-14,
        safe_top_pct=0, safe_bottom_pct=0,
    )
    w, h = dims_for_engine("wan", preset)
    assert w >= 16 and h >= 16


def test_safe_zone_px_vertical_has_right_rail():
    from video_formats import resolve_format, safe_zone_px
    _top, right, _bottom, _left = safe_zone_px(resolve_format("tiktok"))
    assert right > 0  # vertical formats reserve right-rail space


def test_safe_zone_px_landscape_has_no_rail():
    from video_formats import resolve_format, safe_zone_px
    top, right, bottom, left = safe_zone_px(resolve_format("landscape"))
    assert (top, right, bottom, left) == (0, 0, 0, 0)


def test_safe_zone_px_bottom_for_shorts():
    from video_formats import resolve_format, safe_zone_px
    top, _r, bottom, _l = safe_zone_px(resolve_format("shorts"))
    # Shorts reserves ~22% of height at bottom for UI
    assert bottom > top
    assert 300 < bottom < 500


def test_clamp_duration_caps_to_platform_max():
    from video_formats import clamp_duration
    # TikTok caps at 180s; Reels at 90s; Shorts at 60s
    assert clamp_duration("shorts", 120.0) == 60.0
    assert clamp_duration("reels", 999.0) == 90.0
    assert clamp_duration("tiktok", 50.0) == 50.0


def test_clamp_duration_floor_at_one_second():
    from video_formats import clamp_duration
    assert clamp_duration("shorts", 0.1) == 1.0
    assert clamp_duration("shorts", -5.0) == 1.0


def test_clamp_duration_unknown_format_falls_through_reels():
    from video_formats import clamp_duration
    assert clamp_duration("whatever", 120.0) == 90.0  # reels cap


def test_platform_targets_serializes_everything_a_ui_needs():
    from video_formats import platform_targets
    p = platform_targets("reels")
    for k in ("id", "label", "aspect", "width", "height", "gen_width",
              "gen_height", "max_duration_s", "target_fps",
              "max_bitrate_mbps", "audio_lufs", "safe_zone_px", "notes"):
        assert k in p
    # safe_zone_px is a sub-dict
    for k in ("top", "right", "bottom", "left"):
        assert k in p["safe_zone_px"]
