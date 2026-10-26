"""Tests for scripts/lipsync_regression_check.py.

The runner wraps filmmaker.lipsync_eval.compare_scorecards with CLI plumbing
and a skip-friendly 'no opinion' short-circuit for CI runners that have no
lipsync models installed. These tests drive the ``main`` entry point with
fixture scorecards, so no ffmpeg or lipsync binary is required.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def _load_runner():
    """Load scripts/lipsync_regression_check.py as a module - it is not
    under a package so importlib has to resolve it by path."""
    here = Path(__file__).resolve()
    src = here.parents[1] / "scripts" / "lipsync_regression_check.py"
    spec = importlib.util.spec_from_file_location(
        "studiolite_lipsync_gate", src)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


RUNNER = _load_runner()


def _write_scorecard(path: Path, backends):
    """backends: list of (name, {metric: value}, {metric: higher_is_better})."""
    body = {
        "corpus_count": 1,
        "backends": [
            {
                "backend": name,
                "clip_count": 1,
                "rendered_count": 1 if any(v is not None for v in metrics.values()) else 0,
                "metrics": metrics,
                "per_clip": [{
                    "reference_video": "r", "candidate_video": "c",
                    "audio": None,
                    "metrics": [
                        {"metric": m, "value": v,
                         "higher_is_better": higher.get(m, True)}
                        for m, v in metrics.items()
                    ],
                }],
            }
            for name, metrics, higher in backends
        ],
        "ranking": {},
    }
    path.write_text(json.dumps(body), encoding="utf-8")


def test_runner_skips_when_baseline_is_all_none(tmp_path: Path, capsys):
    """A placeholder baseline (every metric None) short-circuits to exit 0
    with a clear 'skipping gate' message."""
    higher = {"syncnet_confidence": True}
    baseline = tmp_path / "baseline.json"
    _write_scorecard(baseline, [("musetalk",
                                  {"syncnet_confidence": None}, higher)])
    candidate = tmp_path / "candidate.json"
    _write_scorecard(candidate, [("musetalk",
                                   {"syncnet_confidence": 7.0}, higher)])

    rc = RUNNER.main([
        "--baseline", str(baseline),
        "--candidate", str(candidate),
    ])
    out = capsys.readouterr().out
    assert rc == 0
    assert "skipping gate" in out


def test_runner_skips_when_candidate_has_no_opinion(tmp_path: Path, capsys):
    """Candidate renders zero clips (no lipsync binaries in CI) -> skip."""
    higher = {"syncnet_confidence": True}
    baseline = tmp_path / "baseline.json"
    _write_scorecard(baseline, [("musetalk",
                                  {"syncnet_confidence": 7.0}, higher)])
    candidate = tmp_path / "candidate.json"
    _write_scorecard(candidate, [("musetalk",
                                   {"syncnet_confidence": None}, higher)])
    rc = RUNNER.main([
        "--baseline", str(baseline),
        "--candidate", str(candidate),
    ])
    out = capsys.readouterr().out
    assert rc == 0
    assert "skipping gate" in out


def test_runner_pass_when_candidate_within_threshold(tmp_path: Path, capsys):
    higher = {"syncnet_confidence": True}
    baseline = tmp_path / "baseline.json"
    _write_scorecard(baseline, [("musetalk",
                                  {"syncnet_confidence": 7.0}, higher)])
    candidate = tmp_path / "candidate.json"
    _write_scorecard(candidate, [("musetalk",
                                   {"syncnet_confidence": 6.9}, higher)])
    rc = RUNNER.main([
        "--baseline", str(baseline),
        "--candidate", str(candidate),
        "--threshold", "0.05",
    ])
    out = capsys.readouterr().out
    assert rc == 0
    assert "PASS" in out


def test_runner_fails_on_regression(tmp_path: Path, capsys):
    higher = {"syncnet_confidence": True}
    baseline = tmp_path / "baseline.json"
    _write_scorecard(baseline, [("musetalk",
                                  {"syncnet_confidence": 10.0}, higher)])
    candidate = tmp_path / "candidate.json"
    _write_scorecard(candidate, [("musetalk",
                                   {"syncnet_confidence": 5.0}, higher)])
    rc = RUNNER.main([
        "--baseline", str(baseline),
        "--candidate", str(candidate),
        "--threshold", "0.05",
    ])
    out = capsys.readouterr().out
    assert rc == 1
    assert "FAIL" in out
    assert "musetalk/syncnet_confidence" in out


def test_runner_env_threshold_is_used(tmp_path: Path, monkeypatch, capsys):
    """A moderate drop that would pass at the default 5% threshold must
    fail when the env knob tightens the gate to 1%."""
    higher = {"syncnet_confidence": True}
    baseline = tmp_path / "baseline.json"
    _write_scorecard(baseline, [("musetalk",
                                  {"syncnet_confidence": 10.0}, higher)])
    candidate = tmp_path / "candidate.json"
    _write_scorecard(candidate, [("musetalk",
                                   {"syncnet_confidence": 9.7}, higher)])
    monkeypatch.setenv("LIPSYNC_REGRESSION_THRESHOLD", "0.01")
    rc = RUNNER.main([
        "--baseline", str(baseline),
        "--candidate", str(candidate),
    ])
    assert rc == 1


def test_runner_missing_baseline_errors_cleanly(tmp_path: Path, capsys):
    missing = tmp_path / "nope.json"
    rc = RUNNER.main(["--baseline", str(missing)])
    err = capsys.readouterr().err
    assert rc == 2
    assert "baseline not found" in err


def test_runner_bad_env_threshold_falls_back(tmp_path: Path, monkeypatch, capsys):
    """A non-numeric env value warns and falls back to the default, so a
    typo in CI env config does not change gate behaviour silently."""
    higher = {"syncnet_confidence": True}
    baseline = tmp_path / "baseline.json"
    _write_scorecard(baseline, [("musetalk",
                                  {"syncnet_confidence": 7.0}, higher)])
    candidate = tmp_path / "candidate.json"
    _write_scorecard(candidate, [("musetalk",
                                   {"syncnet_confidence": 6.9}, higher)])
    monkeypatch.setenv("LIPSYNC_REGRESSION_THRESHOLD", "not-a-number")
    rc = RUNNER.main([
        "--baseline", str(baseline),
        "--candidate", str(candidate),
    ])
    err = capsys.readouterr().err
    assert rc == 0
    assert "not a number" in err


def test_runner_writes_candidate_when_asked(tmp_path: Path):
    higher = {"syncnet_confidence": True}
    baseline = tmp_path / "baseline.json"
    _write_scorecard(baseline, [("musetalk",
                                  {"syncnet_confidence": 7.0}, higher)])
    candidate = tmp_path / "candidate.json"
    _write_scorecard(candidate, [("musetalk",
                                   {"syncnet_confidence": 6.9}, higher)])
    written = tmp_path / "written.json"
    rc = RUNNER.main([
        "--baseline", str(baseline),
        "--candidate", str(candidate),
        "--write-candidate", str(written),
    ])
    assert rc == 0
    body = json.loads(written.read_text(encoding="utf-8"))
    assert body["backends"][0]["backend"] == "musetalk"


def test_runner_against_shipped_placeholder_baseline():
    """The in-repo placeholder has every metric None; the runner must
    short-circuit and pass without a candidate file."""
    rc = RUNNER.main([])
    # Default baseline path points at tests/data/lipsync_baseline.json,
    # which is the placeholder - gate skips.
    assert rc == 0
