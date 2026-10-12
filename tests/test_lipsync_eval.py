"""Tests for the local lipsync-eval protocol.

Everything is driven through stub drivers and stub metrics so these
tests do not need ffmpeg, SyncNet, or any lipsync model. The real-world
usage path (shell-out to a user-owned SyncNet binary) is covered by one
dedicated test that stubs subprocess.run."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import pytest

from filmmaker import lipsync_eval as le


@pytest.fixture(autouse=True)
def _isolate_metrics():
    original = list(le._METRICS)
    le._METRICS.clear()
    try:
        yield
    finally:
        le._METRICS.clear()
        le._METRICS.extend(original)


def _const_metric(name: str, value, higher: bool = True):
    def fn(_ref, _cand, _audio):
        return value
    return le.MetricSpec(name=name, higher_is_better=higher, fn=fn)


# ---------------------------------------------------------------------------
# Shape / registration

def test_default_metrics_include_the_canonical_four():
    """After the autouse fixture snapshots them away, re-run the default
    registration and assert the four research-named metrics are there."""
    le.clear_metrics()
    le._register_default_metrics()
    names = {m.name for m in le.registered_metrics()}
    assert {"syncnet_confidence", "lmd", "fid", "fvd",
            "duration_drift_sec", "frame_count_delta"} <= names


def test_register_replaces_same_name():
    le.register_metric(_const_metric("x", 1.0))
    le.register_metric(_const_metric("x", 2.0))
    specs = [m for m in le.registered_metrics() if m.name == "x"]
    assert len(specs) == 1
    assert specs[0].fn("a", "b", None) == 2.0


def test_score_pair_runs_every_metric():
    le.register_metric(_const_metric("a", 0.5, higher=True))
    le.register_metric(_const_metric("b", 7.0, higher=False))
    result = le.score_pair("ref.mp4", "cand.mp4", "audio.wav")
    assert result.reference_video == "ref.mp4"
    assert len(result.metrics) == 2
    by_name = {m.metric: m for m in result.metrics}
    assert by_name["a"].value == 0.5 and by_name["a"].higher_is_better is True
    assert by_name["b"].value == 7.0 and by_name["b"].higher_is_better is False


def test_score_pair_exception_is_none_not_crash():
    def broken(_r, _c, _a):
        raise RuntimeError("boom")
    le.register_metric(le.MetricSpec(name="x", higher_is_better=True, fn=broken))
    le.register_metric(_const_metric("y", 0.9))
    result = le.score_pair("r.mp4", "c.mp4", None)
    by_name = {m.metric: m.value for m in result.metrics}
    assert by_name["x"] is None
    assert by_name["y"] == 0.9


# ---------------------------------------------------------------------------
# Matrix

def test_run_matrix_aggregates_and_ranks(tmp_path: Path):
    """Two backends, two corpus pairs, one metric. The backend that
    writes a candidate at the expected path should rank first."""
    le.register_metric(_const_metric("sync", 10.0, higher=True))

    def good_driver(_ref, _audio, out_path):
        Path(out_path).write_bytes(b"stub")
        return {"ok": True}

    def missing_driver(_ref, _audio, _out):
        # Writes nothing; matrix must mark rendered=False.
        return None

    corpus = [("a.mp4", None), ("b.mp4", None)]
    card = le.run_matrix({"good": good_driver, "bad": missing_driver},
                         corpus, str(tmp_path))
    assert card.corpus_count == 2
    good = next(r for r in card.backends if r.backend == "good")
    bad = next(r for r in card.backends if r.backend == "bad")
    assert good.rendered_count == 2
    assert bad.rendered_count == 0
    # 'bad' rendered zero clips so it has None on every metric; the
    # ranking excludes it rather than tying it to the bottom.
    assert card.ranking["sync"] == ["good"]


def test_run_matrix_metric_absent_backend_omitted_from_ranking(tmp_path: Path):
    """When one backend's metric returns None (uninstalled SyncNet),
    it drops out of that metric's ranking but keeps its row in the
    scorecard - the UI can show the None explicitly."""
    # Opinion only on the alpha backend's output. The marker uses a
    # path separator plus backend name so a stray 'alpha' substring
    # inside the OS tmp path cannot accidentally match.
    def fn(_r, c, _a):
        return 7.0 if (os.sep + "alpha" + os.sep) in c else None
    le.register_metric(le.MetricSpec(name="syncnet_confidence",
                                       higher_is_better=True, fn=fn))

    def driver(_ref, _audio, out_path):
        Path(out_path).write_bytes(b"stub")
    corpus = [("ref.mp4", None)]
    card = le.run_matrix({"alpha": driver, "beta": driver},
                         corpus, str(tmp_path))
    assert card.ranking["syncnet_confidence"] == ["alpha"]
    beta_row = next(r for r in card.backends if r.backend == "beta")
    assert beta_row.metrics["syncnet_confidence"] is None


def test_run_matrix_driver_raise_is_isolated(tmp_path: Path):
    """A driver that raises must not take down the matrix; its backend
    is just marked rendered=False for that pair."""
    def good(_r, _a, out):
        Path(out).write_bytes(b"stub")
    def rude(_r, _a, _out):
        raise RuntimeError("rude")
    le.register_metric(_const_metric("sync", 1.0, higher=True))
    card = le.run_matrix({"g": good, "r": rude}, [("ref.mp4", None)],
                          str(tmp_path))
    assert next(r for r in card.backends if r.backend == "r").rendered_count == 0


# ---------------------------------------------------------------------------
# CLI shell-out adapter

def test_cli_adapter_returns_none_without_env(monkeypatch):
    """No binary env var set -> metric returns None without touching
    subprocess. Covers every heavy metric by proxy."""
    for env in (le._SYNCNET_BIN, le._LMD_BIN, le._FID_BIN, le._FVD_BIN):
        monkeypatch.delenv(env, raising=False)
    assert le._syncnet_metric("r", "c", "a.wav") is None
    assert le._lmd_metric("r", "c", None) is None
    assert le._fid_metric("r", "c", None) is None
    assert le._fvd_metric("r", "c", None) is None


def test_cli_adapter_parses_bare_float(tmp_path: Path, monkeypatch):
    stub = tmp_path / "syncnet"
    stub.write_text("#!/bin/sh\necho 7.42\n")
    monkeypatch.setenv(le._SYNCNET_BIN, str(stub))

    import subprocess as _sp
    def fake_run(cmd, **kwargs):
        return _sp.CompletedProcess(args=cmd, returncode=0,
                                      stdout="7.42\n", stderr="")
    monkeypatch.setattr(le.subprocess, "run", fake_run)
    assert le._syncnet_metric("r", "c", "a.wav") == pytest.approx(7.42)


def test_cli_adapter_parses_json_score(tmp_path: Path, monkeypatch):
    stub = tmp_path / "lmd"
    stub.write_text("#!/bin/sh\necho '{\"score\": 3.14}'\n")
    monkeypatch.setenv(le._LMD_BIN, str(stub))

    import subprocess as _sp
    def fake_run(cmd, **kwargs):
        return _sp.CompletedProcess(args=cmd, returncode=0,
                                      stdout='{"score": 3.14}', stderr="")
    monkeypatch.setattr(le.subprocess, "run", fake_run)
    assert le._lmd_metric("r", "c", None) == pytest.approx(3.14)


def test_cli_adapter_nonzero_exit_is_none(tmp_path: Path, monkeypatch):
    stub = tmp_path / "fid"
    stub.write_text("#!/bin/sh\nexit 2\n")
    monkeypatch.setenv(le._FID_BIN, str(stub))

    import subprocess as _sp
    def fake_run(cmd, **kwargs):
        return _sp.CompletedProcess(args=cmd, returncode=2,
                                      stdout="", stderr="boom")
    monkeypatch.setattr(le.subprocess, "run", fake_run)
    assert le._fid_metric("r", "c", None) is None


# ---------------------------------------------------------------------------
# Studio lite backend driver wrapper

class _FakeBackend:
    def __init__(self, name: str, available: bool, writes: bool = True):
        self.name = name
        self._available = available
        self._writes = writes

    def check_installed(self, _project):
        return le  # unused value, just truthy; see below


def test_studiolite_backend_driver_marks_unavailable_as_none(tmp_path: Path, monkeypatch):
    """When a backend reports itself unavailable, the driver returns
    None and writes nothing to disk."""
    from filmmaker import lipsync as _ls

    class Unavailable:
        name = "x"
        def check_installed(self, _project):
            return _ls.BackendStatus(name="x", available=False,
                                      reason="no config")
        def run(self, *args, **kwargs):
            raise AssertionError("should not be invoked")

    monkeypatch.setattr(_ls, "get_backend", lambda _name: Unavailable())

    class _Proj:
        dir = str(tmp_path)
    driver = le.studiolite_backend_driver(_Proj(), "x")
    out = tmp_path / "cand.mp4"
    assert driver("ref.mp4", None, str(out)) is None
    assert not out.exists()


def test_studiolite_backend_driver_runs_available_backend(tmp_path: Path, monkeypatch):
    """When the backend reports available, the driver should call its
    run() with a sensible voice_lines manifest."""
    from filmmaker import lipsync as _ls

    seen = {}

    class Ready:
        name = "y"
        def check_installed(self, _project):
            return _ls.BackendStatus(name="y", available=True)
        def run(self, project, input_video, voice_lines, output_video):
            seen["project"] = project
            seen["voice_lines"] = voice_lines
            seen["output_video"] = output_video
            Path(output_video).write_bytes(b"stub")
            return {"ok": True}

    monkeypatch.setattr(_ls, "get_backend", lambda _n: Ready())

    class _Proj:
        dir = str(tmp_path)
    driver = le.studiolite_backend_driver(_Proj(), "y")
    out = tmp_path / "cand.mp4"
    r = driver("ref.mp4", str(tmp_path / "speech.wav"), str(out))
    assert r == {"ok": True}
    assert out.exists()
    # voice_lines carries one fake entry with the eval metadata.
    assert len(seen["voice_lines"]) == 1
    assert seen["voice_lines"][0]["synthesized"] is True


# ---------------------------------------------------------------------------
# CLI

def test_cli_requires_corpus_and_out(capsys):
    rc = le._main(["run", "--backend", "musetalk"])
    assert rc == 2


def test_cli_runs_with_noop_drivers(tmp_path: Path, capsys):
    """A corpus JSONL and no project-dir uses noop drivers; nothing is
    rendered but the harness still emits a well-shaped scorecard."""
    le.register_metric(_const_metric("sync", 2.0, higher=True))
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        json.dumps({"reference": "a.mp4", "audio": None}) + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "out"
    rc = le._main(["run", "--corpus", str(corpus), "--out", str(out),
                    "--backend", "musetalk"])
    assert rc == 0
    body = json.loads(capsys.readouterr().out)
    assert body["corpus_count"] == 1
    assert body["backends"][0]["backend"] == "musetalk"
    assert body["backends"][0]["rendered_count"] == 0
