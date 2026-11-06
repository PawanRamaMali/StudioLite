"""Tests for thumbnails.py and the /api/v1/edit/thumbnail endpoint.

Covers:
- probe_duration returns a float on success and None on failure (so
  grid mode can still fall back).
- evenly_spaced_timestamps gives midpoints nudged inside [0, duration].
- _overlay_xy maps each named position to the right libavfilter coords.
- build_single_cmd / build_grid_cmd emit the exact argv we expect,
  with and without the overlay branch.
- extract_single / extract_grid / extract_multiple report missing
  source, missing overlay, missing ffmpeg, non-zero exit, and no-output
  all as plain failure dicts rather than raising.
- extract_multiple packs a zip when asked.
- /api/v1/edit/thumbnail validates inputs and dispatches to the runner.
"""
from __future__ import annotations

import os
import subprocess
import zipfile

import pytest


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeProc:
    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = ""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _mk_src(tmp_path):
    src = tmp_path / "in.mp4"
    src.write_bytes(b"\x00" * 4096)
    return str(src)


def _mk_png(tmp_path, name="logo.png"):
    p = tmp_path / name
    p.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 128)
    return str(p)


def _touch(path: str, size: int = 1024) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"\x00" * size)


# ---------------------------------------------------------------------------
# probe_duration / evenly_spaced_timestamps
# ---------------------------------------------------------------------------


def test_probe_duration_parses_stdout():
    from thumbnails import probe_duration
    out = probe_duration(
        "whatever.mp4",
        runner=lambda c: _FakeProc(returncode=0, stdout="42.75\n"),
    )
    assert out == pytest.approx(42.75)


def test_probe_duration_none_on_nonzero_exit():
    from thumbnails import probe_duration
    out = probe_duration(
        "x.mp4",
        runner=lambda c: _FakeProc(returncode=1, stderr="no such file"),
    )
    assert out is None


def test_probe_duration_none_when_ffprobe_missing():
    from thumbnails import probe_duration

    def fake(_):
        raise FileNotFoundError("ffprobe")

    assert probe_duration("x.mp4", runner=fake) is None


def test_probe_duration_none_on_junk_stdout():
    from thumbnails import probe_duration
    out = probe_duration(
        "x.mp4",
        runner=lambda c: _FakeProc(returncode=0, stdout="N/A"),
    )
    assert out is None


@pytest.mark.parametrize("duration, count, expected", [
    (60.0, 1, [30.0]),
    (60.0, 2, [15.0, 45.0]),
    (10.0, 4, [1.25, 3.75, 6.25, 8.75]),
])
def test_evenly_spaced_timestamps(duration, count, expected):
    from thumbnails import evenly_spaced_timestamps
    out = evenly_spaced_timestamps(duration, count)
    assert out == pytest.approx(expected)


def test_evenly_spaced_timestamps_zero_count():
    from thumbnails import evenly_spaced_timestamps
    assert evenly_spaced_timestamps(60.0, 0) == []


def test_evenly_spaced_timestamps_zero_duration():
    from thumbnails import evenly_spaced_timestamps
    # 0s video, 3 frames → all land at 0.
    assert evenly_spaced_timestamps(0, 3) == [0.0, 0.0, 0.0]


# ---------------------------------------------------------------------------
# _overlay_xy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pos, expected_contains", [
    ("top-left",     "20:20"),
    ("top-right",    "main_w-overlay_w-20:20"),
    ("bottom-left",  "20:main_h-overlay_h-20"),
    ("bottom-right", "main_w-overlay_w-20:main_h-overlay_h-20"),
    ("center",       "(main_w-overlay_w)/2:(main_h-overlay_h)/2"),
    ("garbage",      "main_w-overlay_w-20:main_h-overlay_h-20"),  # falls back to bottom-right
])
def test_overlay_xy(pos, expected_contains):
    from thumbnails import _overlay_xy
    assert _overlay_xy(pos) == expected_contains


# ---------------------------------------------------------------------------
# build_single_cmd
# ---------------------------------------------------------------------------


def test_build_single_cmd_no_overlay():
    from thumbnails import build_single_cmd
    cmd = build_single_cmd("in.mp4", "out.png", 3.5, width=640)
    assert cmd[0] == "ffmpeg"
    assert "-ss" in cmd and cmd[cmd.index("-ss") + 1] == "3.5"
    assert "-i" in cmd and cmd[cmd.index("-i") + 1] == "in.mp4"
    assert "-frames:v" in cmd and cmd[cmd.index("-frames:v") + 1] == "1"
    assert "-vf" in cmd and cmd[cmd.index("-vf") + 1] == "scale=640:-1"
    assert cmd[-1] == "out.png"


def test_build_single_cmd_clamps_negative_timestamp():
    from thumbnails import build_single_cmd
    cmd = build_single_cmd("in.mp4", "out.png", -5.0)
    assert cmd[cmd.index("-ss") + 1] == "0.0"


def test_build_single_cmd_with_overlay_uses_filter_complex():
    from thumbnails import build_single_cmd
    cmd = build_single_cmd(
        "in.mp4", "out.png", 1.0,
        width=720,
        overlay_path="logo.png",
        overlay_position="top-right",
        overlay_opacity=0.5,
    )
    assert cmd.count("-i") == 2  # source + overlay
    assert "-filter_complex" in cmd
    fc = cmd[cmd.index("-filter_complex") + 1]
    assert "scale=720:-1" in fc
    assert "colorchannelmixer=aa=0.5" in fc
    assert "overlay=main_w-overlay_w-20:20" in fc
    assert "[vout]" in fc
    assert cmd[cmd.index("-map") + 1] == "[vout]"


# ---------------------------------------------------------------------------
# build_grid_cmd
# ---------------------------------------------------------------------------


def test_build_grid_cmd_with_duration_uses_fps_filter():
    from thumbnails import build_grid_cmd
    cmd = build_grid_cmd(
        "in.mp4", "sheet.png",
        cols=3, rows=3, width=1200, duration=90.0,
    )
    vf = cmd[cmd.index("-vf") + 1]
    # 9 frames across 90s -> fps = 0.1
    assert "fps=0.100000" in vf
    assert "scale=400:-1" in vf  # cell_w = 1200/3
    assert "tile=3x3" in vf
    assert cmd[cmd.index("-frames:v") + 1] == "1"


def test_build_grid_cmd_without_duration_falls_back_to_select_mod():
    from thumbnails import build_grid_cmd
    cmd = build_grid_cmd(
        "in.mp4", "sheet.png",
        cols=2, rows=2, width=800, duration=None,
    )
    vf = cmd[cmd.index("-vf") + 1]
    # select='not(mod(n,300))' with libavfilter's escaped comma
    assert "select='not(mod(n\\,300))'" in vf
    assert "tile=2x2" in vf


def test_build_grid_cmd_with_overlay_routes_through_filter_complex():
    from thumbnails import build_grid_cmd
    cmd = build_grid_cmd(
        "in.mp4", "sheet.png",
        cols=2, rows=2, width=800, duration=60.0,
        overlay_path="logo.png",
        overlay_position="center",
        overlay_opacity=1.0,
    )
    assert "-filter_complex" in cmd
    fc = cmd[cmd.index("-filter_complex") + 1]
    assert "tile=2x2" in fc
    assert "overlay=(main_w-overlay_w)/2:(main_h-overlay_h)/2" in fc


# ---------------------------------------------------------------------------
# extract_single
# ---------------------------------------------------------------------------


def test_extract_single_missing_source(tmp_path):
    from thumbnails import extract_single
    r = extract_single(
        str(tmp_path / "ghost.mp4"),
        str(tmp_path / "out.png"),
        1.0,
    )
    assert r["ok"] is False
    assert "not found" in r["error"]


def test_extract_single_missing_overlay(tmp_path):
    from thumbnails import extract_single
    src = _mk_src(tmp_path)
    r = extract_single(
        src, str(tmp_path / "out.png"), 1.0,
        overlay_path=str(tmp_path / "ghost.png"),
    )
    assert r["ok"] is False
    assert "overlay not found" in r["error"]


def test_extract_single_success(tmp_path):
    from thumbnails import extract_single
    src = _mk_src(tmp_path)
    dst = str(tmp_path / "out.png")

    def fake(cmd):
        _touch(cmd[-1])
        return _FakeProc(returncode=0)

    r = extract_single(src, dst, 1.0, runner=fake)
    assert r["ok"] is True
    assert r["mode"] == "single"
    assert r["output_paths"] == [dst]


def test_extract_single_ffmpeg_missing_returns_install_hint(tmp_path):
    from thumbnails import extract_single
    src = _mk_src(tmp_path)

    def fake(_):
        raise FileNotFoundError("ffmpeg")

    r = extract_single(src, str(tmp_path / "out.png"), 1.0, runner=fake)
    assert r["ok"] is False
    assert "install_hint" in r
    assert r["error"] == "ffmpeg not found"


def test_extract_single_nonzero_exit_surfaces_tail(tmp_path):
    from thumbnails import extract_single
    src = _mk_src(tmp_path)

    def fake(_):
        return _FakeProc(returncode=1, stderr="first\n[fatal] codec x265 not available")

    r = extract_single(src, str(tmp_path / "out.png"), 1.0, runner=fake)
    assert r["ok"] is False
    assert "codec x265" in r["error"]


def test_extract_single_no_output_file(tmp_path):
    from thumbnails import extract_single
    src = _mk_src(tmp_path)

    def fake(_):
        return _FakeProc(returncode=0)  # exits clean but writes nothing

    r = extract_single(src, str(tmp_path / "out.png"), 1.0, runner=fake)
    assert r["ok"] is False
    assert "no output" in r["error"]


# ---------------------------------------------------------------------------
# extract_grid
# ---------------------------------------------------------------------------


def test_extract_grid_success(tmp_path):
    from thumbnails import extract_grid
    src = _mk_src(tmp_path)
    dst = str(tmp_path / "sheet.png")

    def fake(cmd):
        # grid uses ffprobe first — that one has no -y; the ffmpeg call
        # that actually writes includes -y so we gate on that.
        if cmd[0].endswith("ffprobe") or "ffprobe" in cmd[0]:
            return _FakeProc(returncode=0, stdout="60.0\n")
        _touch(cmd[-1])
        return _FakeProc(returncode=0)

    r = extract_grid(src, dst, cols=3, rows=3, width=1200, runner=fake)
    assert r["ok"] is True
    assert r["mode"] == "grid"
    assert r["output_paths"] == [dst]


def test_extract_grid_rejects_bad_dims(tmp_path):
    from thumbnails import extract_grid
    src = _mk_src(tmp_path)
    r = extract_grid(src, str(tmp_path / "sheet.png"), cols=0, rows=3)
    assert r["ok"] is False
    assert "positive" in r["error"]


# ---------------------------------------------------------------------------
# extract_multiple
# ---------------------------------------------------------------------------


def test_extract_multiple_requires_timestamps(tmp_path):
    from thumbnails import extract_multiple
    src = _mk_src(tmp_path)
    r = extract_multiple(src, str(tmp_path / "out"), [])
    assert r["ok"] is False
    assert "timestamp" in r["error"]


def test_extract_multiple_writes_zip(tmp_path):
    from thumbnails import extract_multiple
    src = _mk_src(tmp_path)
    out_dir = tmp_path / "thumbs"
    archive = tmp_path / "thumbs.zip"

    def fake(cmd):
        _touch(cmd[-1])
        return _FakeProc(returncode=0)

    r = extract_multiple(
        src, str(out_dir), [1.0, 5.0, 9.0],
        archive_path=str(archive),
        runner=fake,
    )
    assert r["ok"], r
    assert len(r["output_paths"]) == 3
    assert r["archive_path"] == str(archive)
    with zipfile.ZipFile(archive) as zf:
        names = zf.namelist()
    assert len(names) == 3
    assert all(n.endswith(".png") for n in names)


def test_extract_multiple_ffmpeg_failure_names_timestamp(tmp_path):
    from thumbnails import extract_multiple
    src = _mk_src(tmp_path)

    def fake(cmd):
        return _FakeProc(returncode=1, stderr="[boom] bad frame")

    r = extract_multiple(
        src, str(tmp_path / "out"), [7.5], runner=fake,
    )
    assert r["ok"] is False
    assert "t=7.5" in r["error"]


# ---------------------------------------------------------------------------
# generate_thumbnails dispatcher
# ---------------------------------------------------------------------------


def test_generate_thumbnails_unknown_mode(tmp_path):
    from thumbnails import generate_thumbnails
    src = _mk_src(tmp_path)
    r = generate_thumbnails(src, str(tmp_path), mode="bogus")
    assert r["ok"] is False
    assert "unknown mode" in r["error"]


def test_generate_thumbnails_multiple_defaults_use_probed_duration(tmp_path):
    from thumbnails import generate_thumbnails
    src = _mk_src(tmp_path)

    calls = {"n": 0}

    def fake(cmd):
        calls["n"] += 1
        if "ffprobe" in (cmd[0] or ""):
            return _FakeProc(returncode=0, stdout="20.0\n")
        _touch(cmd[-1])
        return _FakeProc(returncode=0)

    r = generate_thumbnails(
        src, str(tmp_path), mode="multiple", job_id="abc", runner=fake,
    )
    assert r["ok"], r
    # No explicit timestamps: dispatcher asked for 5 evenly spaced.
    assert len(r["output_paths"]) == 5
    assert r["archive_path"].endswith("thumbs_abc.zip")


# ---------------------------------------------------------------------------
# /api/v1/edit/thumbnail endpoint
# ---------------------------------------------------------------------------


def _fresh_api(monkeypatch, tmp_path):
    import importlib
    import sys as _sys
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("STUDIOLITE_AUTH", "off")
    for name in list(_sys.modules):
        if name == "api_server" or name.startswith("api_server."):
            del _sys.modules[name]
    api = importlib.import_module("api_server")
    from fastapi.testclient import TestClient
    return api, TestClient(api.app)


def test_thumbnail_endpoint_rejects_missing_video(tmp_path, monkeypatch):
    _api, client = _fresh_api(monkeypatch, tmp_path)
    r = client.post(
        "/api/v1/edit/thumbnail",
        json={"video_path": str(tmp_path / "ghost.mp4")},
    )
    assert r.status_code == 422
    assert "not found" in r.json()["detail"].lower()


def test_thumbnail_endpoint_dispatches_runner(tmp_path, monkeypatch):
    api, client = _fresh_api(monkeypatch, tmp_path)
    src = tmp_path / "in.mp4"
    src.write_bytes(b"\x00" * 32)

    seen = {}

    def fake_run(job_id, req):
        seen["job_id"] = job_id
        seen["video_path"] = req.video_path
        seen["mode"] = req.mode
        seen["cols"] = req.cols
        seen["rows"] = req.rows

    monkeypatch.setattr(api, "_run_thumbnail", fake_run, raising=True)

    class _Immediate:
        def __init__(self, *, target, args, daemon=False):
            self.target = target
            self.args = args
        def start(self):
            self.target(*self.args)

    monkeypatch.setattr(api.threading, "Thread", _Immediate, raising=True)

    r = client.post(
        "/api/v1/edit/thumbnail",
        json={"video_path": str(src), "mode": "grid", "cols": 4, "rows": 3},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["job_id"] == seen["job_id"]
    assert seen["mode"] == "grid"
    assert seen["cols"] == 4
    assert seen["rows"] == 3


def test_thumbnail_request_defaults():
    from api_server import ThumbnailRequest
    req = ThumbnailRequest(video_path="x.mp4")
    assert req.mode == "single"
    assert req.timestamp == 0.0
    assert req.cols == 3
    assert req.rows == 3
    assert req.width == 1280
    assert req.overlay_path is None
    assert req.overlay_position == "bottom-right"
    assert req.overlay_opacity == 1.0
