"""Tests for the per-stage backend registry.

The registry is a pure metadata source. These tests pin the shape of each
spec and the facts the deep-research round confirmed, so a drift in a
VRAM number or license tag forces a conscious update."""
from __future__ import annotations

import pytest

from filmmaker import backend_registry as br


def test_core_shipping_backends_present():
    """Every backend the stage runners already dispatch to must be in the
    registry so the UI can source its facts from one place."""
    shipping = {
        ("upscale", "none"),
        ("upscale", "realesrgan_x2"),
        ("upscale", "realesrgan_x4"),
        ("lipsync", "none"),
        ("lipsync", "musetalk"),
        ("lipsync", "latentsync"),
        ("lipsync", "echomimicv3"),
        ("motion_shots", "wan22"),
        ("composer", "musicgen"),
        ("composer", "acestep"),
        ("voice_actor", "piper"),
        ("voice_actor", "xtts"),
        ("voice_actor", "maya2"),
    }
    for stage, name in shipping:
        spec = br.get(stage, name)
        assert spec is not None, f"missing spec for {stage}/{name}"
        assert spec.available == "shipping"


def test_roadmap_backends_flagged_separately():
    """Roadmap entries exist for the UI but must not be surfaced as
    'shipping' - their stage runner is not wired yet."""
    for stage, name in (
        ("upscale", "seedvr2"),
        ("motion_shots", "wan21_flf2v"),
        ("composer", "diffrhythm2"),
        ("shots", "omnigen2"),
    ):
        spec = br.get(stage, name)
        assert spec is not None, f"missing roadmap spec {stage}/{name}"
        assert spec.available == "roadmap"


def test_echomimicv3_facts_from_research():
    """Pin the EchoMimic V3 facts the research confirmed. If any of these
    drift the registry needs a conscious update, not a silent change."""
    spec = br.get("lipsync", "echomimicv3")
    assert spec.license == "Apache-2.0"
    assert spec.min_vram_gb_fp16 == 12.0
    assert spec.cloud_burst is True
    assert "Wan2.1-Fun" in spec.base_model


def test_seedvr2_has_full_gguf_ladder():
    """cmeka/SeedVR2-GGUF publishes Q3_K_M through Q8_0. Research pinned
    the five VRAM footprints - a change here means the ladder shifted."""
    spec = br.get("upscale", "seedvr2")
    formats = {q.format: q.vram_gb for q in spec.quant_variants}
    assert formats["gguf_q3_k_m"] == 1.55
    assert formats["gguf_q4_k_m"] == 2.0
    assert formats["gguf_q5_k_m"] == 2.41
    assert formats["gguf_q6_k"] == 2.85
    assert formats["gguf_q8_0"] == 3.66


def test_wan21_flf2v_gguf_ladder_endpoints():
    """city96's Wan2.1-Fun-14B-InP GGUF ladder - the research cites the
    three endpoints Q2_K / Q8_0 / BF16."""
    spec = br.get("motion_shots", "wan21_flf2v")
    formats = {q.format: q.vram_gb for q in spec.quant_variants}
    assert formats["gguf_q2_k"] == 6.02
    assert formats["gguf_q8_0"] == 17.7
    assert formats["bf16"] == 32.8


def test_diffrhythm2_license_and_cloud_eligibility():
    """DiffRhythm 2 is Apache 2.0 and GPU-only in practice; registry must
    say so to steer the dispatcher."""
    spec = br.get("composer", "diffrhythm2")
    assert spec.license == "Apache-2.0"
    assert spec.cloud_burst is True
    assert spec.min_vram_gb_fp16 is not None and spec.min_vram_gb_fp16 >= 8.0


def test_for_stage_filters_by_stage():
    """for_stage returns only that stage's backends. Order is not
    guaranteed but membership is."""
    lipsync = {s.name for s in br.for_stage("lipsync")}
    assert {"none", "musetalk", "latentsync", "echomimicv3"} <= lipsync
    assert "wan22" not in lipsync


def test_accepted_names_is_sorted_and_lowercased():
    """accepted_names feeds the config-patch validator; it must be sorted
    for a deterministic error message."""
    names = br.accepted_names("upscale")
    assert names == sorted(names)
    assert "none" in names and "realesrgan_x2" in names


def test_register_overwrites_same_key():
    """Re-registering the same (stage, name) replaces the spec. Lets a
    plugin shim swap in richer facts without duplicating the entry."""
    before = br.get("upscale", "none")
    try:
        br.register(br.BackendSpec(
            stage="upscale", name="none",
            license="MIT", available="shipping",
            notes="replaced for this test",
        ))
        after = br.get("upscale", "none")
        assert after.notes == "replaced for this test"
        assert after.license == "MIT"
    finally:
        if before is not None:
            br.register(before)


def test_cpu_fine_backends_have_zero_or_none_vram_floor():
    """The CPU-fine defaults (piper, musicgen 'none', upscale 'none') must
    not claim a VRAM requirement."""
    piper = br.get("voice_actor", "piper")
    assert piper.min_vram_gb_fp16 in (None, 0.0)
    none_upscale = br.get("upscale", "none")
    assert none_upscale.min_vram_gb_fp16 in (None, 0.0)
