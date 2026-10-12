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
            "ffprobe.duration-sanity",
            "ffprobe.resolution",
            "ffprobe.aspect-ratio",
        }
    finally:
        vbench2._SCORERS.clear()
        vbench2._SCORERS.extend(before)


# ---------------------------------------------------------------------------
# Project-level entrypoint

class _FakeProject:
    """Minimal duck-type matching what score_project needs. Stays here
    rather than importing the real Project so these tests do not create
    a films/ dir on disk per case."""

    def __init__(self, root: str, shots: list):
        self.dir = root
        self._shots = shots

    def read_artifact(self, key: str):
        if key == "motion_shots":
            return {"shots": self._shots}
        return None


def test_score_project_uses_motion_shots_artifact(tmp_path, monkeypatch):
    """When the motion_shots artifact lists output_path entries that
    exist on disk, score_project uses those - not a directory scan."""
    # Register one scorer that returns a constant so we can predict the
    # aggregate shape without ffmpeg installed.
    vbench2.register_scorer(vbench2.ScorerSpec(
        name="k", dimension="physics", capability="mechanics",
        fn=lambda _p: 0.6,
    ))
    (tmp_path / "artifacts").mkdir()
    clip_a = tmp_path / "a.mp4"
    clip_b = tmp_path / "b.mp4"
    clip_a.write_bytes(b"stub")
    clip_b.write_bytes(b"stub")
    proj = _FakeProject(str(tmp_path), [
        {"output_path": "a.mp4"}, {"output_path": "b.mp4"},
    ])
    card = vbench2.score_project(proj)
    assert card.clip_count == 2
    phys = next(d for d in card.aggregate if d.dimension == "physics")
    assert phys.score == pytest.approx(0.6)
    # Persisted next to continuity.
    assert (tmp_path / "artifacts" / "vbench2.json").exists()


def test_score_project_falls_back_to_scan_when_artifact_malformed(tmp_path):
    """A missing or wrong-shaped motion_shots artifact must not kill the
    scorer - we fall back to scanning artifacts/motion_shots/."""
    vbench2.register_scorer(vbench2.ScorerSpec(
        name="k", dimension="physics", capability="mechanics",
        fn=lambda _p: 0.9,
    ))
    motion_dir = tmp_path / "artifacts" / "motion_shots"
    motion_dir.mkdir(parents=True)
    (motion_dir / "001.mp4").write_bytes(b"stub")
    (motion_dir / "002.mp4").write_bytes(b"stub")

    class BadArtProject(_FakeProject):
        def read_artifact(self, _key):
            return {"shots": [{"no_path": "oops"}]}

    proj = BadArtProject(str(tmp_path), [])
    card = vbench2.score_project(proj, store=False)
    assert card.clip_count == 2


def test_score_project_handles_empty_project(tmp_path):
    """No motion_shots artifact, no scan dir - returns clip_count=0 and
    None scores, never raises."""
    class EmptyProject(_FakeProject):
        def read_artifact(self, _key):
            return None
    proj = EmptyProject(str(tmp_path), [])
    card = vbench2.score_project(proj, store=False)
    assert card.clip_count == 0
    assert card.overall is None
    for dim in card.aggregate:
        assert dim.score is None


# ---------------------------------------------------------------------------
# Official VBench 2.0 shell-out

def test_official_vbench2_scorer_registered_when_bin_set(tmp_path, monkeypatch):
    """VBENCH2_BIN pointing at a plausible binary registers one scorer
    per sub-capability. We stub subprocess.run so the registration call
    does not actually execute the binary."""
    stub_bin = tmp_path / "vbench2"
    stub_bin.write_text("#!/bin/sh\n")
    monkeypatch.setenv("VBENCH2_BIN", str(stub_bin))
    added = vbench2.register_official_vbench2_scorers()
    # One scorer per sub-capability, so 18.
    total_caps = sum(len(c) for c in vbench2.DIMENSIONS.values())
    assert added == total_caps
    # Each registered scorer carries the official-scorer name prefix.
    names = {s.name for s in vbench2.registered_scorers()
             if s.name.startswith("vbench2.official:")}
    assert len(names) == total_caps


def test_official_vbench2_scorer_absent_when_bin_missing(monkeypatch):
    monkeypatch.delenv("VBENCH2_BIN", raising=False)
    # Also neutralise shutil.which so a system-installed vbench2 does
    # not light the test up.
    monkeypatch.setattr(vbench2, "_which", lambda _n: None)
    assert vbench2.register_official_vbench2_scorers() == 0


def test_official_vbench2_scorer_parses_json(tmp_path, monkeypatch):
    """When the shell-out exits 0 and prints JSON, the parsed dimension
    score round-trips as the scorer's output."""
    stub_bin = tmp_path / "vbench2"
    stub_bin.write_text("#!/bin/sh\n")
    monkeypatch.setenv("VBENCH2_BIN", str(stub_bin))
    clip = tmp_path / "shot.mp4"
    clip.write_bytes(b"stub")

    import subprocess as _sp
    def fake_run(cmd, **kwargs):
        # Return the physics score the real CLI would print.
        return _sp.CompletedProcess(args=cmd, returncode=0,
                                      stdout='{"scores": {"physics": 0.73}}',
                                      stderr="")
    monkeypatch.setattr(vbench2.subprocess, "run", fake_run)

    added = vbench2.register_official_vbench2_scorers()
    assert added > 0
    # Pick the physics/mechanics scorer and run it; its fn is a closure
    # that shells out and extracts the physics number.
    physics = next(s for s in vbench2.registered_scorers()
                   if s.name == "vbench2.official:physics:mechanics")
    assert physics.fn(str(clip)) == pytest.approx(0.73)


def test_official_vbench2_scorer_tolerates_nonzero_exit(tmp_path, monkeypatch):
    """When the shell-out fails, the scorer returns None rather than
    raising - caller treats None as 'no opinion'."""
    stub_bin = tmp_path / "vbench2"
    stub_bin.write_text("#!/bin/sh\n")
    monkeypatch.setenv("VBENCH2_BIN", str(stub_bin))
    clip = tmp_path / "shot.mp4"
    clip.write_bytes(b"stub")

    import subprocess as _sp
    def fake_run(cmd, **kwargs):
        return _sp.CompletedProcess(args=cmd, returncode=2,
                                      stdout="", stderr="boom")
    monkeypatch.setattr(vbench2.subprocess, "run", fake_run)
    vbench2.register_official_vbench2_scorers()
    scorer = next(s for s in vbench2.registered_scorers()
                   if s.name.startswith("vbench2.official:"))
    assert scorer.fn(str(clip)) is None


# ---------------------------------------------------------------------------
# Prompt corpus loader

def test_prompt_corpus_loads_json_lists(tmp_path, monkeypatch):
    """One JSON file per capability. Lists and {"prompts": [...]} both
    load; everything else is skipped quietly."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "mechanics.json").write_text(
        '["ball drops", "water pours"]', encoding="utf-8")
    (corpus / "composition.json").write_text(
        '{"prompts": ["wide shot of a tower"]}', encoding="utf-8")
    (corpus / "readme.txt").write_text("ignored", encoding="utf-8")
    (corpus / "broken.json").write_text("not json {", encoding="utf-8")

    out = vbench2.load_prompt_corpus(str(corpus))
    assert out["mechanics"] == ["ball drops", "water pours"]
    assert out["composition"] == ["wide shot of a tower"]
    assert "readme" not in out
    assert "broken" not in out


def test_prompt_corpus_empty_when_dir_missing(monkeypatch):
    monkeypatch.delenv("VBENCH2_PROMPT_DIR", raising=False)
    assert vbench2.load_prompt_corpus() == {}


def test_prompt_corpus_reads_env(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "material.json").write_text('["cloth tears"]', encoding="utf-8")
    monkeypatch.setenv("VBENCH2_PROMPT_DIR", str(corpus))
    out = vbench2.load_prompt_corpus()
    assert out == {"material": ["cloth tears"]}


# ---------------------------------------------------------------------------
# Aspect ratio and duration-sanity scorers

def test_aspect_ratio_scorer_bands(tmp_path, monkeypatch):
    """Pin the band structure: 16:9 -> 1.0, 9:16 -> 0.8, 1:1 (sq) -> 0.6,
    wildly wide -> 0.0."""
    def _probe(w, h):
        return {"streams": [{"codec_type": "video", "width": w, "height": h}]}

    monkeypatch.setattr(vbench2, "_ffprobe_json",
                         lambda _p: _probe(1920, 1080))
    assert vbench2._aspect_ratio_scorer("x.mp4") == pytest.approx(1.0)
    monkeypatch.setattr(vbench2, "_ffprobe_json",
                         lambda _p: _probe(1080, 1920))
    assert vbench2._aspect_ratio_scorer("x.mp4") == pytest.approx(0.8)
    monkeypatch.setattr(vbench2, "_ffprobe_json",
                         lambda _p: _probe(1024, 1024))
    assert vbench2._aspect_ratio_scorer("x.mp4") == pytest.approx(0.6)
    monkeypatch.setattr(vbench2, "_ffprobe_json",
                         lambda _p: _probe(4000, 100))
    assert vbench2._aspect_ratio_scorer("x.mp4") == pytest.approx(0.0)


def test_duration_sanity_scorer_bands(monkeypatch):
    """1-15 s = 1.0, 0.5-30 s = 0.5, outside = 0.0, missing probe = None."""
    def _probe(duration):
        return {"streams": [{"codec_type": "video",
                              "duration": str(duration)}]}
    monkeypatch.setattr(vbench2, "_ffprobe_json", lambda _p: _probe(5.0))
    assert vbench2._duration_sanity_scorer("x.mp4") == pytest.approx(1.0)
    monkeypatch.setattr(vbench2, "_ffprobe_json", lambda _p: _probe(20.0))
    assert vbench2._duration_sanity_scorer("x.mp4") == pytest.approx(0.5)
    monkeypatch.setattr(vbench2, "_ffprobe_json", lambda _p: _probe(0.1))
    assert vbench2._duration_sanity_scorer("x.mp4") == pytest.approx(0.0)
    monkeypatch.setattr(vbench2, "_ffprobe_json", lambda _p: None)
    assert vbench2._duration_sanity_scorer("x.mp4") is None
