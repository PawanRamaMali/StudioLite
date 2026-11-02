"""Tests for filmmaker/rife.py.

Covers:
- Multiplier validation (powers of two 2/4/8 only; 1 is a no-op copy).
- Missing-source handling, missing-library handling.
- Recursive 2x pass logic:
    * N frames -> 2x pass yields 2N-1 frames (N originals + N-1 midpoints).
    * 4x = 2x applied twice -> 4N-3.
    * 8x = three passes -> 8N-7.
- The old `extract_frames` tuple-unpack bug is handled defensively.
- `interpolate_video` returns {ok:False, install_hint} when
  rife_ncnn_vulkan_python is missing (so API callers can surface a
  clean install hint instead of a stack trace).
"""
from __future__ import annotations

import os
import sys
import types
from typing import List

import pytest


# ---------------------------------------------------------------------------
# Fakes for the external deps rife wraps.
# ---------------------------------------------------------------------------


class _FakeRifeEngine:
    """process(a, b) returns a solid gray midpoint; we only care that it
    was called the right number of times."""
    def __init__(self, gpuid: int = 0):
        self.gpuid = gpuid
        self.calls = 0

    def process(self, a, b):
        import numpy as np
        self.calls += 1
        # Shape doesn't matter; PIL just needs to save it.
        return np.full_like(a, 128)


def _install_fake_rife(monkeypatch, engine_cls=_FakeRifeEngine):
    """Create a fake `rife_ncnn_vulkan_python` module in sys.modules."""
    mod = types.ModuleType("rife_ncnn_vulkan_python")
    mod.Rife = engine_cls
    monkeypatch.setitem(sys.modules, "rife_ncnn_vulkan_python", mod)
    return mod


def _make_src_frames(tmp_path, n: int) -> List[str]:
    """Write n solid-color PNGs that _interpolate_once can PIL.open."""
    from PIL import Image
    paths = []
    for i in range(n):
        p = str(tmp_path / f"src_{i:03d}.png")
        Image.new("RGB", (8, 8), (i * 20, 0, 0)).save(p)
        paths.append(p)
    return paths


# ---------------------------------------------------------------------------
# Pure logic: validators and pass-count math.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mult, expected_passes", [(2, 1), (4, 2), (8, 3)])
def test_passes_for_powers_of_two(mult, expected_passes):
    from filmmaker.rife import _passes_for
    assert _passes_for(mult) == expected_passes


@pytest.mark.parametrize("bad", [0, 3, 5, 6, 7, 16, -2])
def test_validate_multiplier_rejects_non_power_of_two(bad):
    from filmmaker.rife import _validate_multiplier
    err = _validate_multiplier(bad)
    assert err is not None
    assert "multiplier" in err.lower()


def test_validate_multiplier_accepts_one_as_noop():
    from filmmaker.rife import _validate_multiplier
    assert _validate_multiplier(1) is None


# ---------------------------------------------------------------------------
# _interpolate_once: frame-list expansion.
# ---------------------------------------------------------------------------


def test_interpolate_once_doubles_frame_count(tmp_path, monkeypatch):
    from filmmaker.rife import _interpolate_once
    _install_fake_rife(monkeypatch)
    frames = _make_src_frames(tmp_path, 5)
    engine = _FakeRifeEngine()
    out = _interpolate_once(frames, engine, str(tmp_path), round_tag="t0")
    # N originals + (N-1) midpoints
    assert len(out) == 2 * len(frames) - 1 == 9
    assert engine.calls == len(frames) - 1 == 4
    # Originals still in order at even indices
    for i, src in enumerate(frames):
        assert out[2 * i] == src


def test_interpolate_once_single_frame_is_noop(tmp_path, monkeypatch):
    from filmmaker.rife import _interpolate_once
    _install_fake_rife(monkeypatch)
    frames = _make_src_frames(tmp_path, 1)
    engine = _FakeRifeEngine()
    out = _interpolate_once(frames, engine, str(tmp_path), round_tag="t0")
    assert out == frames
    assert engine.calls == 0


# ---------------------------------------------------------------------------
# interpolate_video: end-to-end with every external dep faked.
# ---------------------------------------------------------------------------


class _Capture:
    """Collect args passed to the fake extract_frames / frames_to_video."""
    def __init__(self):
        self.extract_calls = []
        self.frames_to_video_calls = []


def _install_fake_video_editor(monkeypatch, cap: _Capture, src_frames,
                                extract_shape: str = "tuple"):
    """Inject a fake `video_editor` module so the real cv2-backed one
    (and its heavy transitive imports) never load. extract_shape is
    'tuple' for the current (list, fps) return and 'list' for the older
    bare-list shape, so one test can exercise the defensive unpack."""

    def fake_extract(video_path, output_dir=None):
        cap.extract_calls.append((video_path, output_dir))
        if extract_shape == "list":
            return src_frames
        return src_frames, 24.0  # the real shape: (list, fps)

    def fake_frames_to_video(frame_paths, output_path=None, fps=24,
                              audio_from=None):
        cap.frames_to_video_calls.append({
            "n_frames": len(frame_paths),
            "output_path": output_path,
            "fps": fps,
            "audio_from": audio_from,
        })
        # Write a non-trivial file so the >1024 size check passes.
        with open(output_path, "wb") as f:
            f.write(b"\x00" * 4096)
        return output_path

    mod = types.ModuleType("video_editor")
    mod.extract_frames = fake_extract
    mod.frames_to_video = fake_frames_to_video
    monkeypatch.setitem(sys.modules, "video_editor", mod)


def _mk_src_video(tmp_path):
    p = tmp_path / "src.mp4"
    p.write_bytes(b"\x00" * 32)
    return str(p)


def test_interpolate_video_2x_counts(tmp_path, monkeypatch):
    from filmmaker.rife import interpolate_video
    _install_fake_rife(monkeypatch)
    src_frames = _make_src_frames(tmp_path, 5)
    cap = _Capture()
    _install_fake_video_editor(monkeypatch, cap, src_frames)

    dst = str(tmp_path / "out.mp4")
    result = interpolate_video(_mk_src_video(tmp_path), dst, multiplier=2)
    assert result["ok"], result
    assert result["multiplier"] == 2
    assert result["backend"] == "rife_x2"
    assert result["fps_in"] == 24.0
    assert result["fps_out"] == 48.0
    assert result["output_path"] == dst
    assert len(cap.frames_to_video_calls) == 1
    # 5 originals -> one 2x pass -> 2*5 - 1 = 9 frames
    assert cap.frames_to_video_calls[0]["n_frames"] == 9
    assert cap.frames_to_video_calls[0]["fps"] == 48.0
    # Audio is muxed back from the original.
    assert cap.frames_to_video_calls[0]["audio_from"] == \
        cap.extract_calls[0][0]


def test_interpolate_video_4x_counts(tmp_path, monkeypatch):
    from filmmaker.rife import interpolate_video
    _install_fake_rife(monkeypatch)
    src_frames = _make_src_frames(tmp_path, 5)
    cap = _Capture()
    _install_fake_video_editor(monkeypatch, cap, src_frames)

    dst = str(tmp_path / "out.mp4")
    result = interpolate_video(_mk_src_video(tmp_path), dst, multiplier=4)
    assert result["ok"]
    # 5 -> 9 -> 17 frames after two passes
    assert cap.frames_to_video_calls[0]["n_frames"] == 17
    assert result["fps_out"] == 96.0


def test_interpolate_video_8x_counts(tmp_path, monkeypatch):
    from filmmaker.rife import interpolate_video
    _install_fake_rife(monkeypatch)
    src_frames = _make_src_frames(tmp_path, 3)
    cap = _Capture()
    _install_fake_video_editor(monkeypatch, cap, src_frames)

    dst = str(tmp_path / "out.mp4")
    result = interpolate_video(_mk_src_video(tmp_path), dst, multiplier=8)
    assert result["ok"]
    # 3 -> 5 -> 9 -> 17 frames after three passes
    assert cap.frames_to_video_calls[0]["n_frames"] == 17
    assert result["fps_out"] == 192.0


def test_interpolate_video_missing_source(tmp_path, monkeypatch):
    from filmmaker.rife import interpolate_video
    result = interpolate_video(
        str(tmp_path / "ghost.mp4"), str(tmp_path / "out.mp4"), multiplier=2
    )
    assert result["ok"] is False
    assert "not found" in result["error"].lower()


def test_interpolate_video_rejects_bad_multiplier(tmp_path):
    from filmmaker.rife import interpolate_video
    src = _mk_src_video(tmp_path)
    result = interpolate_video(src, str(tmp_path / "out.mp4"), multiplier=3)
    assert result["ok"] is False
    assert "multiplier" in result["error"].lower()


def test_interpolate_video_multiplier_one_copies_through(tmp_path):
    from filmmaker.rife import interpolate_video
    src = _mk_src_video(tmp_path)
    dst = str(tmp_path / "out.mp4")
    result = interpolate_video(src, dst, multiplier=1)
    assert result["ok"]
    assert result["multiplier"] == 1
    assert result["backend"] == "rife-noop"
    assert os.path.exists(dst)
    assert os.path.getsize(dst) == os.path.getsize(src)


def test_interpolate_video_missing_rife_returns_install_hint(tmp_path, monkeypatch):
    from filmmaker.rife import interpolate_video
    # Make sure any previously-installed fake is cleared.
    monkeypatch.delitem(sys.modules, "rife_ncnn_vulkan_python", raising=False)
    # And block the real import if it happens to be present on this box.
    import builtins
    real_import = builtins.__import__

    def guard(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "rife_ncnn_vulkan_python":
            raise ImportError("stub: not installed")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guard)

    src = _mk_src_video(tmp_path)
    result = interpolate_video(src, str(tmp_path / "out.mp4"), multiplier=2)
    assert result["ok"] is False
    assert "install_hint" in result
    assert "rife-ncnn-vulkan-python" in result["install_hint"]


def test_interpolate_video_needs_two_frames(tmp_path, monkeypatch):
    """A single-frame source can't be interpolated: report cleanly."""
    from filmmaker.rife import interpolate_video
    _install_fake_rife(monkeypatch)
    src_frames = _make_src_frames(tmp_path, 1)
    cap = _Capture()
    _install_fake_video_editor(monkeypatch, cap, src_frames)

    result = interpolate_video(
        _mk_src_video(tmp_path), str(tmp_path / "out.mp4"), multiplier=2
    )
    assert result["ok"] is False
    assert "at least 2 frames" in result["error"]


def test_interpolate_video_handles_old_list_only_extract_shape(tmp_path, monkeypatch):
    """If a future refactor makes extract_frames return just a list again,
    we should still work (fps falls back to a sensible default)."""
    from filmmaker.rife import interpolate_video
    _install_fake_rife(monkeypatch)
    src_frames = _make_src_frames(tmp_path, 4)
    cap = _Capture()
    _install_fake_video_editor(monkeypatch, cap, src_frames,
                                extract_shape="list")

    result = interpolate_video(
        _mk_src_video(tmp_path), str(tmp_path / "out.mp4"), multiplier=2
    )
    assert result["ok"]
    # Falls back to 24 * multiplier for fps_out when fps_in is unknown
    assert result["fps_out"] == 48.0


def test_interpolate_video_progress_callback(tmp_path, monkeypatch):
    from filmmaker.rife import interpolate_video
    _install_fake_rife(monkeypatch)
    src_frames = _make_src_frames(tmp_path, 3)
    cap = _Capture()
    _install_fake_video_editor(monkeypatch, cap, src_frames)

    events = []
    interpolate_video(
        _mk_src_video(tmp_path), str(tmp_path / "out.mp4"),
        multiplier=4,
        progress_callback=lambda cur, tot, msg: events.append((cur, tot, msg)),
    )
    # Two passes for 4x + a final "muxing" event
    assert len(events) == 3
    assert events[-1][0] == events[-1][1]  # final event hits 100%
