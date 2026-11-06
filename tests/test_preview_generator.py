"""Tests for preview_generator.py (low-res proxy config builder)."""
from __future__ import annotations

import pytest


def test_defaults_known_engine():
    from preview_generator import preview_defaults
    d = preview_defaults("ltx23_distilled_gguf")
    assert d["num_inference_steps"] == 4
    assert d["fps"] == 24
    assert "aspects" in d


def test_defaults_unknown_engine_falls_back_to_ltx():
    from preview_generator import preview_defaults
    d = preview_defaults("made-up-engine")
    assert d["num_inference_steps"] == 4  # LTX default


def test_build_portrait_ltx():
    from preview_generator import build_preview_config
    c = build_preview_config("test", engine="ltx23_distilled_gguf", aspect="9:16", target_duration_s=1.0)
    assert c.width % 32 == 0 and c.height % 32 == 0
    assert c.width <= c.height  # portrait
    assert c.num_inference_steps == 4
    assert c.num_frames >= 9  # LTX minimum (8k+1 with k=1)


def test_build_square_respects_stride():
    from preview_generator import build_preview_config
    c = build_preview_config("test", engine="wan22_14b_gguf", aspect="1:1")
    assert c.width % 16 == 0 and c.height % 16 == 0
    assert c.width == c.height


def test_build_landscape_is_wide():
    from preview_generator import build_preview_config
    c = build_preview_config("test", engine="ltx23_distilled_gguf", aspect="16:9")
    assert c.width > c.height


def test_build_unknown_aspect_falls_back_to_9x16():
    from preview_generator import build_preview_config
    c = build_preview_config("test", engine="ltx23_distilled_gguf", aspect="whatever")
    assert c.width < c.height  # 9:16 shape


def test_frame_count_matches_ltx_stride():
    # LTX must have (8k + 1) frames
    from preview_generator import build_preview_config
    c = build_preview_config("test", engine="ltx23_distilled_gguf", target_duration_s=1.0)
    assert (c.num_frames - 1) % 8 == 0


def test_frame_count_matches_wan_stride():
    # Wan/Hunyuan/VACE must have (4k + 1) frames
    from preview_generator import build_preview_config
    c = build_preview_config("test", engine="wan22_14b_gguf", target_duration_s=1.0)
    assert (c.num_frames - 1) % 4 == 0


def test_frames_capped_at_max():
    from preview_generator import build_preview_config
    # Requesting 10 seconds would mean 240 frames; preview caps at ~25
    c = build_preview_config("test", engine="ltx23_distilled_gguf", target_duration_s=10.0)
    assert c.num_frames <= 25


def test_target_duration_recomputed_from_frames():
    from preview_generator import build_preview_config
    c = build_preview_config("test", engine="ltx23_distilled_gguf", target_duration_s=0.5)
    # Actual duration must match the actually-generated frame count
    assert abs(c.target_duration_s - (c.num_frames - 1) / c.fps) < 0.01


def test_seed_passthrough():
    from preview_generator import build_preview_config
    c = build_preview_config("test", seed=42)
    assert c.seed == 42
    c2 = build_preview_config("test", seed=None)
    assert c2.seed is None


def test_preview_config_dict_shape():
    from preview_generator import preview_config_dict
    d = preview_config_dict("test", engine="wan22_14b_gguf")
    for k in ("engine", "width", "height", "num_frames", "num_inference_steps",
              "guidance_scale", "target_duration_s", "fps", "seed"):
        assert k in d


def test_estimate_seconds_scales_with_frames():
    from preview_generator import estimate_preview_seconds
    short = estimate_preview_seconds("ltx23_distilled_gguf", num_frames=9, steps=4)
    long = estimate_preview_seconds("ltx23_distilled_gguf", num_frames=25, steps=4)
    assert long > short


def test_estimate_seconds_scales_with_steps():
    from preview_generator import estimate_preview_seconds
    few = estimate_preview_seconds("wan22_14b_gguf", num_frames=25, steps=8)
    many = estimate_preview_seconds("wan22_14b_gguf", num_frames=25, steps=30)
    assert many > few


def test_estimate_seconds_includes_base_overhead():
    from preview_generator import estimate_preview_seconds
    # Even a tiny preview costs the SDXL + pipeline-load overhead
    assert estimate_preview_seconds("ltx23_distilled_gguf", num_frames=9, steps=4) >= 2.0
