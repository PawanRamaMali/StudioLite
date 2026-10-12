"""Tests for the VBench-2.0 regression harness.

The harness is a pure-Python aggregator over pluggable Scorer functions.
We stub the scorers inline so these tests do not need ffmpeg or any model
weights - just the harness math and shape."""
from __future__ import annotations

from pathlib import Path
from typing import List

import pytest

from filmmaker import vbench2


@pytest.fixture(autouse=True)
def _isolate_scorers():
    """Snapshot the module-level scorer list so each test starts clean.

    The default scorers at import time are the ffprobe/ffmpeg ones, which
    are irrelevant to these unit tests - we blow them away for the
    duration of each test and restore afterwards so a later test still
    sees the defaults."""
    original = list(vbench2._SCORERS)
    vbench2._SCORERS.clear()
    try:
        yield
    finally:
        vbench2._SCORERS.clear()
        vbench2._SCORERS.extend(original)


def _const(value):
    """Scorer that always returns `value` ignoring the input clip."""
    def fn(_clip_path: str):
        return value
    return fn


def test_dimensions_match_paper():
    """VBench-2.0 Table 1 - 5 dimensions, 18 sub-capabilities total."""
    assert set(vbench2.DIMENSIONS) == {
        "human_fidelity", "controllability", "creativity",
        "physics", "commonsense",
    }
    total = sum(len(caps) for caps in vbench2.DIMENSIONS.values())
    assert total == 18, "paper reports 18 sub-capabilities across 5 dimensions"


def test_register_scorer_rejects_unknown_dimension():
    with pytest.raises(ValueError, match="unknown dimension"):
        vbench2.register_scorer(vbench2.ScorerSpec(
            name="x", dimension="not_a_dim", capability="whatever",
            fn=_const(1.0),
        ))


def test_register_scorer_rejects_unknown_capability():
    with pytest.raises(ValueError, match="unknown capability"):
        vbench2.register_scorer(vbench2.ScorerSpec(
            name="x", dimension="physics", capability="not_a_cap",
            fn=_const(1.0),
        ))


def test_register_scorer_replaces_same_name_idempotently():
    """Re-registering the same (name, dim, cap) replaces, does not duplicate.

    This lets a caller plug in a stronger VLM scorer at startup without
    having to tear down the previous registration."""
    vbench2.register_scorer(vbench2.ScorerSpec(
        name="v1", dimension="physics", capability="mechanics",
        fn=_const(0.3),
    ))
    vbench2.register_scorer(vbench2.ScorerSpec(
        name="v1", dimension="physics", capability="mechanics",
        fn=_const(0.9),
    ))
    specs = [s for s in vbench2.registered_scorers()
             if s.name == "v1" and s.capability == "mechanics"]
    assert len(specs) == 1
    assert specs[0].fn("anything") == 0.9


def test_score_clip_averages_within_capability_and_dimension():
    """Two scorers on the same capability average; dimension averages its
    non-None capabilities; overall averages non-None dimensions."""
    vbench2.register_scorer(vbench2.ScorerSpec(
        name="a", dimension="physics", capability="mechanics", fn=_const(0.4),
    ))
    vbench2.register_scorer(vbench2.ScorerSpec(
        name="b", dimension="physics", capability="mechanics", fn=_const(0.6),
    ))
    vbench2.register_scorer(vbench2.ScorerSpec(
        name="c", dimension="physics", capability="material", fn=_const(0.8),
    ))
    card = vbench2.score_clip("dummy.mp4")
    phys = next(d for d in card.dimensions if d.dimension == "physics")
    mech = next(c for c in phys.capabilities if c.capability == "mechanics")
    mat = next(c for c in phys.capabilities if c.capability == "material")
    assert mech.score == pytest.approx(0.5)
    assert mat.score == pytest.approx(0.8)
    assert phys.score == pytest.approx(0.65)
    # No other dimension has scorers -> all None, so overall is just physics.
    assert card.overall == pytest.approx(0.65)


def test_score_clip_none_scorer_does_not_pull_dimension_down():
    """A scorer that returns None is excluded from aggregation, not treated
    as 0.0. Critical for the real harness where some scorers only produce
    verdicts on clips with recognisable humans/objects."""
    vbench2.register_scorer(vbench2.ScorerSpec(
        name="opinion", dimension="physics", capability="mechanics",
        fn=_const(0.9),
    ))
    vbench2.register_scorer(vbench2.ScorerSpec(
        name="shrug", dimension="physics", capability="mechanics",
        fn=_const(None),
    ))
    card = vbench2.score_clip("dummy.mp4")
    phys = next(d for d in card.dimensions if d.dimension == "physics")
    mech = next(c for c in phys.capabilities if c.capability == "mechanics")
    assert mech.score == pytest.approx(0.9)


def test_scorer_exception_is_treated_as_none(caplog):
    """A broken scorer must not take down the whole run."""
    def broken(_path):
        raise RuntimeError("boom")
    vbench2.register_scorer(vbench2.ScorerSpec(
        name="broken", dimension="physics", capability="mechanics",
        fn=broken,
    ))
    vbench2.register_scorer(vbench2.ScorerSpec(
        name="ok", dimension="physics", capability="mechanics",
        fn=_const(0.5),
    ))
    card = vbench2.score_clip("dummy.mp4")
    phys = next(d for d in card.dimensions if d.dimension == "physics")
    mech = next(c for c in phys.capabilities if c.capability == "mechanics")
    assert mech.score == pytest.approx(0.5)


def test_overall_is_none_when_nothing_scored():
    """An all-None scorecard reports None at every level - not 0.0 - so a
    caller can distinguish 'ran and failed' from 'nothing had an opinion'."""
    vbench2.register_scorer(vbench2.ScorerSpec(
        name="shrug", dimension="physics", capability="mechanics",
        fn=_const(None),
    ))
    card = vbench2.score_clip("dummy.mp4")
    assert card.overall is None
    for dim in card.dimensions:
        assert dim.score is None


def test_score_dir_aggregates_across_clips(tmp_path: Path):
    """Each clip returns the same value; dir score should be that value."""
    (tmp_path / "a.mp4").write_bytes(b"stub")
    (tmp_path / "b.mp4").write_bytes(b"stub")
    (tmp_path / "notes.txt").write_text("ignored by suffix filter")
    vbench2.register_scorer(vbench2.ScorerSpec(
        name="k", dimension="physics", capability="mechanics",
        fn=_const(0.7),
    ))
    card = vbench2.score_dir(str(tmp_path))
    assert card.clip_count == 2
    phys = next(d for d in card.aggregate if d.dimension == "physics")
    assert phys.score == pytest.approx(0.7)
    assert card.overall == pytest.approx(0.7)


def test_score_dir_missing_directory_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        vbench2.score_dir(str(tmp_path / "does-not-exist"))


def test_scorecard_to_dict_round_trips_structure(tmp_path: Path):
    (tmp_path / "a.mp4").write_bytes(b"stub")
    vbench2.register_scorer(vbench2.ScorerSpec(
        name="k", dimension="creativity", capability="composition_creativity",
        fn=_const(0.42),
    ))
    card = vbench2.score_dir(str(tmp_path))
    data = vbench2.scorecard_to_dict(card)
    # Shape sanity: top-level keys, per-clip breakdown, aggregate rows.
    assert set(data.keys()) == {"clip_count", "per_clip", "aggregate", "overall"}
    assert len(data["per_clip"]) == 1
    assert len(data["aggregate"]) == len(vbench2.DIMENSIONS)
    for row in data["aggregate"]:
        assert row["dimension"] in vbench2.DIMENSIONS


def test_default_scorers_are_registered_at_import():
    """After clearing the fixture snapshot we expect the real defaults -
    the fixture restores them in the teardown - but we need to verify at
    module scope that the module itself wires the three built-ins."""
    names = {s.name for s in vbench2.registered_scorers()}
    # Fixture emptied them for the test body, so run assertion against the
    # snapshot the fixture captured, which lives in the fixture closure.
    # Easier: look at the module default-registration function directly.
    before = list(vbench2._SCORERS)
    vbench2._SCORERS.clear()
    try:
        vbench2._register_default_scorers()
        names = {s.name for s in vbench2._SCORERS}
        assert names == {
            "ffprobe.playable",
            "ffmpeg.psnr-neighbour",
            "ffprobe.resolution",
        }
    finally:
        vbench2._SCORERS.clear()
        vbench2._SCORERS.extend(before)
