"""Tests for scene_detection.py and the /api/v1/edit/scene-detect endpoint.

Covers:
- build_scdet_cmd argv shape + threshold clamp.
- parse_scdet_output picks times + scores from varied ffmpeg log formats.
- filter_min_length drops cuts closer than threshold.
- build_thumbnail_cmd argv shape (seek + -vframes + scale).
- detect_scenes: missing source, missing ffmpeg, non-zero exit, thumbnail
  writes, no-cuts path, scene range assembly with and without duration.
- Endpoint 422 on missing source, dispatches runner.
"""
from __future__ import annotations

import os

import pytest


class _FakeProc:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _mk_video(tmp_path):
    p = tmp_path / "in.mp4"
    p.write_bytes(b"\x00" * 16 + b"ftyp" + b"\x00" * 128)
    return str(p)


def _touch(path, size=1024):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"\x00" * size)


# ---------------------------------------------------------------------------
# argv builders
# ---------------------------------------------------------------------------


def test_build_scdet_cmd_shape():
    from scene_detection import build_scdet_cmd
    cmd = build_scdet_cmd("in.mp4", threshold=15.5)
    assert cmd[0] == "ffmpeg"
    assert "-vf" in cmd
    i = cmd.index("-vf")
    assert cmd[i + 1] == "scdet=threshold=15.50"
    assert "-an" in cmd
    assert cmd[-1] == "-"
    assert "null" in cmd


def test_build_scdet_cmd_clamps_threshold():
    from scene_detection import build_scdet_cmd
    low = build_scdet_cmd("x", threshold=-5)
    high = build_scdet_cmd("x", threshold=500)
    assert "scdet=threshold=0.00" in low
    assert "scdet=threshold=100.00" in high


def test_build_thumbnail_cmd_shape():
    from scene_detection import build_thumbnail_cmd
    cmd = build_thumbnail_cmd("in.mp4", "out.jpg", 12.5, width=400)
    i = cmd.index("-ss")
    assert cmd[i + 1] == "12.500"
    assert "-vframes" in cmd
    assert cmd[cmd.index("-vframes") + 1] == "1"
    assert "scale=400:-2" in " ".join(cmd)
    assert cmd[-1] == "out.jpg"


def test_build_thumbnail_cmd_clamps_negative_timestamp():
    from scene_detection import build_thumbnail_cmd
    cmd = build_thumbnail_cmd("in.mp4", "out.jpg", -5.0)
    assert cmd[cmd.index("-ss") + 1] == "0.000"


# ---------------------------------------------------------------------------
# parse_scdet_output / filter_min_length
# ---------------------------------------------------------------------------


_SCENE_LOG = """\
Parsed_scdet_0 @ 0x1 lavfi.scd.score: 18.2
Parsed_scdet_0 @ 0x1 lavfi.scd.time: 2.1
Parsed_scdet_0 @ 0x1 lavfi.scd.score: 24.0
Parsed_scdet_0 @ 0x1 lavfi.scd.time: 5.7
Parsed_scdet_0 @ 0x1 lavfi.scd.score=30.5
Parsed_scdet_0 @ 0x1 lavfi.scd.time=10.33
"""


def test_parse_scdet_output_picks_times_and_scores():
    from scene_detection import parse_scdet_output
    cuts = parse_scdet_output(_SCENE_LOG)
    assert len(cuts) == 3
    assert cuts[0]["time"] == pytest.approx(2.1)
    assert cuts[1]["score"] == pytest.approx(24.0)
    assert cuts[2]["time"] == pytest.approx(10.33)


def test_parse_scdet_output_empty():
    from scene_detection import parse_scdet_output
    assert parse_scdet_output("") == []
    assert parse_scdet_output("no scenes here") == []


def test_parse_scdet_handles_time_without_score():
    """If scdet only emits times (older ffmpeg), score defaults to 0."""
    from scene_detection import parse_scdet_output
    log = "lavfi.scd.time: 1.0\nlavfi.scd.time: 2.0\n"
    cuts = parse_scdet_output(log)
    assert len(cuts) == 2
    assert cuts[0]["score"] == 0.0


def test_filter_min_length_drops_close_cuts():
    from scene_detection import filter_min_length
    cuts = [
        {"time": 1.0, "score": 10},
        {"time": 1.2, "score": 11},
        {"time": 2.5, "score": 12},
        {"time": 2.7, "score": 13},
        {"time": 5.0, "score": 14},
    ]
    kept = filter_min_length(cuts, min_scene_length=0.5)
    assert [k["time"] for k in kept] == [1.0, 2.5, 5.0]


def test_filter_min_length_zero_keeps_all():
    from scene_detection import filter_min_length
    cuts = [{"time": 0.1, "score": 1}, {"time": 0.2, "score": 2}]
    assert filter_min_length(cuts, 0) == cuts


# ---------------------------------------------------------------------------
# probe_duration
# ---------------------------------------------------------------------------


def test_probe_duration_missing_ffprobe(monkeypatch):
    from scene_detection import probe_duration
    monkeypatch.setattr("scene_detection.shutil.which", lambda n: None)
    assert probe_duration("x.mp4") is None


def test_probe_duration_parses(monkeypatch):
    from scene_detection import probe_duration
    monkeypatch.setattr("scene_detection.shutil.which", lambda n: "/bin/ffprobe")

    def runner(cmd):
        return _FakeProc(returncode=0, stdout="42.0")
    assert probe_duration("x.mp4", runner=runner) == pytest.approx(42.0)


# ---------------------------------------------------------------------------
# detect_scenes dispatcher
# ---------------------------------------------------------------------------


def test_detect_scenes_missing_source(tmp_path):
    from scene_detection import detect_scenes
    r = detect_scenes(str(tmp_path / "ghost.mp4"), str(tmp_path / "thumbs"))
    assert r["ok"] is False


def test_detect_scenes_missing_ffmpeg(monkeypatch, tmp_path):
    from scene_detection import detect_scenes
    monkeypatch.setattr("scene_detection.shutil.which", lambda n: None)
    r = detect_scenes(_mk_video(tmp_path), str(tmp_path / "thumbs"))
    assert r["ok"] is False
    assert r.get("install_hint")


def test_detect_scenes_non_zero_exit(monkeypatch, tmp_path):
    from scene_detection import detect_scenes
    monkeypatch.setattr("scene_detection.shutil.which", lambda n: "/bin/" + n)

    def runner(cmd):
        return _FakeProc(returncode=2, stderr="scdet: bad input")
    r = detect_scenes(_mk_video(tmp_path), str(tmp_path / "thumbs"), runner=runner)
    assert r["ok"] is False


def test_detect_scenes_no_cuts_returns_one_scene(monkeypatch, tmp_path):
    """A clip with zero cuts should still return one scene spanning 0..duration."""
    from scene_detection import detect_scenes, SceneDetectionStyle
    monkeypatch.setattr("scene_detection.shutil.which", lambda n: "/bin/" + n)

    def runner(cmd):
        if "ffprobe" in cmd[0] or cmd[0].endswith("ffprobe"):
            return _FakeProc(returncode=0, stdout="60.0")
        if "scdet" in " ".join(cmd):
            return _FakeProc(returncode=0, stderr="")
        # Thumbnail generation
        out = cmd[-1]
        _touch(out)
        return _FakeProc(returncode=0)

    r = detect_scenes(
        _mk_video(tmp_path), str(tmp_path / "thumbs"),
        style=SceneDetectionStyle(extract_thumbnails=False),
        runner=runner,
    )
    assert r["ok"] is True
    assert r["cut_count"] == 0
    assert r["scene_count"] == 1
    assert r["scenes"][0]["start"] == 0.0
    assert r["scenes"][0]["end"] == pytest.approx(60.0)


def test_detect_scenes_with_cuts_and_thumbnails(monkeypatch, tmp_path):
    from scene_detection import detect_scenes, SceneDetectionStyle
    monkeypatch.setattr("scene_detection.shutil.which", lambda n: "/bin/" + n)

    def runner(cmd):
        if cmd[0].endswith("ffprobe"):
            return _FakeProc(returncode=0, stdout="30.0")
        if "scdet" in " ".join(cmd):
            # Two cuts at 10s and 20s
            stderr = (
                "lavfi.scd.time: 10.0\nlavfi.scd.score: 15.0\n"
                "lavfi.scd.time: 20.0\nlavfi.scd.score: 25.0\n"
            )
            return _FakeProc(returncode=0, stderr=stderr)
        # Thumbnail path
        out = cmd[-1]
        _touch(out, 1024)
        return _FakeProc(returncode=0)

    r = detect_scenes(
        _mk_video(tmp_path), str(tmp_path / "thumbs"),
        style=SceneDetectionStyle(extract_thumbnails=True),
        runner=runner,
    )
    assert r["ok"] is True
    assert r["cut_count"] == 2
    assert r["scene_count"] == 3
    # Each scene has a thumbnail
    for s in r["scenes"]:
        assert s["thumbnail_path"] is not None
        assert os.path.isfile(s["thumbnail_path"])


def test_detect_scenes_respects_min_scene_length(monkeypatch, tmp_path):
    from scene_detection import detect_scenes, SceneDetectionStyle
    monkeypatch.setattr("scene_detection.shutil.which", lambda n: "/bin/" + n)

    def runner(cmd):
        if cmd[0].endswith("ffprobe"):
            return _FakeProc(returncode=0, stdout="30.0")
        if "scdet" in " ".join(cmd):
            # Two cuts close together, one more later
            stderr = (
                "lavfi.scd.time: 2.0\nlavfi.scd.score: 15.0\n"
                "lavfi.scd.time: 2.3\nlavfi.scd.score: 16.0\n"
                "lavfi.scd.time: 10.0\nlavfi.scd.score: 25.0\n"
            )
            return _FakeProc(returncode=0, stderr=stderr)
        _touch(cmd[-1])
        return _FakeProc(returncode=0)

    r = detect_scenes(
        _mk_video(tmp_path), str(tmp_path / "thumbs"),
        style=SceneDetectionStyle(min_scene_length=1.0, extract_thumbnails=False),
        runner=runner,
    )
    # Second cut (2.3s) gets dropped because it's within 1s of 2.0s
    assert r["cut_count"] == 2
    assert r["scene_count"] == 3


# ---------------------------------------------------------------------------
# Endpoint
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


class _Immediate:
    def __init__(self, *, target, args, daemon=False):
        self.target, self.args = target, args

    def start(self):
        self.target(*self.args)


def test_scene_detect_endpoint_rejects_missing_source(tmp_path, monkeypatch):
    _api, client = _fresh_api(monkeypatch, tmp_path)
    r = client.post(
        "/api/v1/edit/scene-detect",
        json={"video_path": str(tmp_path / "ghost.mp4")},
    )
    assert r.status_code == 422


def test_scene_detect_endpoint_dispatches_runner(tmp_path, monkeypatch):
    api, client = _fresh_api(monkeypatch, tmp_path)
    src = tmp_path / "in.mp4"
    src.write_bytes(b"\x00" * 64)

    seen = {}

    def fake_run(job_id, req):
        seen["path"] = req.video_path
        seen["threshold"] = req.threshold
        seen["extract_thumbnails"] = req.extract_thumbnails
    monkeypatch.setattr(api, "_run_scene_detect", fake_run, raising=True)
    monkeypatch.setattr(api.threading, "Thread", _Immediate, raising=True)

    r = client.post(
        "/api/v1/edit/scene-detect",
        json={
            "video_path": str(src),
            "threshold": 12.0,
            "extract_thumbnails": False,
        },
    )
    assert r.status_code == 200
    assert seen["threshold"] == 12.0
    assert seen["extract_thumbnails"] is False
