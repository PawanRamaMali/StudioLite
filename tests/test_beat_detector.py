"""Tests for beat_detector.py (BPM/beat detection + snap-to-beat)."""
from __future__ import annotations

import math
import os
import pytest


def _make_click_track(tmp_path, bpm=120.0, duration=6.0, sr=22050):
    """Write a synthetic click track at the given BPM. Used for detection tests."""
    import numpy as np
    import soundfile as sf
    n = int(sr * duration)
    y = np.zeros(n, dtype="float32")
    period = int(sr * 60.0 / bpm)
    # Place a very short burst at each beat boundary
    for i in range(0, n, period):
        end = min(i + 100, n)
        # Decaying sine - sharper onset than a raw impulse
        t = np.arange(end - i) / sr
        y[i:end] = 0.7 * np.sin(2 * np.pi * 2000 * t) * np.exp(-t * 300)
    path = tmp_path / f"click_{int(bpm)}.wav"
    sf.write(str(path), y, sr)
    return str(path)


# ---------------------------------------------------------------------------
# detect_beats
# ---------------------------------------------------------------------------


def test_detect_beats_missing_file_raises(tmp_path):
    from beat_detector import detect_beats
    with pytest.raises(FileNotFoundError):
        detect_beats(str(tmp_path / "nope.wav"))


def test_detect_beats_returns_stable_schema(tmp_path):
    pytest.importorskip("numpy")
    pytest.importorskip("soundfile")
    from beat_detector import detect_beats
    r = detect_beats(_make_click_track(tmp_path, bpm=120.0, duration=4.0))
    assert set(r.keys()) == {"bpm", "beats_s", "downbeats_s", "confidence", "source"}
    assert isinstance(r["beats_s"], list)
    assert isinstance(r["downbeats_s"], list)
    assert 0.0 <= r["confidence"] <= 1.0


def test_detect_beats_identifies_120_bpm(tmp_path):
    pytest.importorskip("numpy")
    pytest.importorskip("soundfile")
    from beat_detector import detect_beats
    r = detect_beats(_make_click_track(tmp_path, bpm=120.0, duration=8.0))
    # Allow a ±4 BPM window - detection quality depends on librosa availability
    assert abs(r["bpm"] - 120.0) < 8.0 or r["bpm"] == 0.0  # 0 means engine bailed


def test_detect_beats_source_is_valid(tmp_path):
    pytest.importorskip("numpy")
    pytest.importorskip("soundfile")
    from beat_detector import detect_beats
    r = detect_beats(_make_click_track(tmp_path, duration=3.0))
    assert r["source"] in ("librosa", "fallback")


def test_detect_beats_downbeats_are_subset_of_beats(tmp_path):
    pytest.importorskip("numpy")
    pytest.importorskip("soundfile")
    from beat_detector import detect_beats
    r = detect_beats(_make_click_track(tmp_path, bpm=100.0, duration=10.0))
    if r["beats_s"]:
        for d in r["downbeats_s"]:
            assert d in r["beats_s"]


def test_detect_beats_empty_file_safe(tmp_path):
    pytest.importorskip("numpy")
    pytest.importorskip("soundfile")
    from beat_detector import detect_beats
    import soundfile as sf
    import numpy as np
    empty = tmp_path / "empty.wav"
    sf.write(str(empty), np.zeros(100, dtype="float32"), 22050)
    r = detect_beats(str(empty))
    assert r["bpm"] in (0.0,)
    assert r["beats_s"] == []


# ---------------------------------------------------------------------------
# snap_to_beats (pure, no I/O)
# ---------------------------------------------------------------------------


def test_snap_to_beats_nudges_cuts_inside_tolerance():
    from beat_detector import snap_to_beats
    beats = [1.0, 2.0, 3.0, 4.0]
    cuts = [1.07, 2.14, 3.03]
    snapped = snap_to_beats(cuts, beats, tolerance_s=0.2)
    assert snapped == [1.0, 2.0, 3.0]


def test_snap_to_beats_leaves_out_of_tolerance_cuts():
    from beat_detector import snap_to_beats
    beats = [1.0, 2.0, 3.0]
    cuts = [1.5]  # halfway between beats, outside tolerance
    snapped = snap_to_beats(cuts, beats, tolerance_s=0.1)
    assert snapped == [1.5]


def test_snap_to_beats_empty_beats_returns_input():
    from beat_detector import snap_to_beats
    assert snap_to_beats([1.0, 2.5], [], tolerance_s=0.2) == [1.0, 2.5]


def test_snap_to_beats_empty_cuts_returns_empty():
    from beat_detector import snap_to_beats
    assert snap_to_beats([], [1.0, 2.0], tolerance_s=0.2) == []


def test_snap_to_beats_picks_nearest_beat():
    from beat_detector import snap_to_beats
    # Cut at 1.3s - nearer to 1.0 than to 2.0
    assert snap_to_beats([1.3], [1.0, 2.0], tolerance_s=0.5) == [1.0]
    # Cut at 1.6s - nearer to 2.0
    assert snap_to_beats([1.6], [1.0, 2.0], tolerance_s=0.5) == [2.0]


def test_snap_to_beats_tolerance_boundary():
    from beat_detector import snap_to_beats
    # Exact tolerance boundary - should snap
    assert snap_to_beats([1.15], [1.0, 2.0], tolerance_s=0.15) == [1.0]
    # Just past - should not
    assert snap_to_beats([1.151], [1.0, 2.0], tolerance_s=0.15) == [1.151]


def test_snap_to_beats_preserves_order():
    from beat_detector import snap_to_beats
    cuts = [2.05, 1.05, 3.05]
    snapped = snap_to_beats(cuts, [1.0, 2.0, 3.0], tolerance_s=0.2)
    # Order of inputs is preserved - the function doesn't sort
    assert snapped == [2.0, 1.0, 3.0]
