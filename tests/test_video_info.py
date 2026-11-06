"""Tests for video_info.py and the /api/v1/edit/info endpoint.

Covers:
- build_probe_cmd produces the expected ffprobe argv.
- _parse_fps handles num/den, 0/0, malformed strings.
- parse_probe_json picks dimensions, codec, fps for the first video stream
  and sample_rate/channels for the first audio stream.
- parse_probe_json survives empty payload / partial JSON / missing fields.
- inspect_video failure paths and one success path.
- Endpoint 422 on missing source, dispatches correctly.
"""
from __future__ import annotations

import json
import os

import pytest


class _FakeProc:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


_SAMPLE_PROBE = {
    "format": {
        "filename": "x.mp4",
        "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
        "duration": "30.0",
        "size": "1048576",
        "bit_rate": "280000",
    },
    "streams": [
        {
            "index": 0,
            "codec_type": "video",
            "codec_name": "h264",
            "codec_long_name": "H.264 / AVC",
            "width": 1920,
            "height": 1080,
            "pix_fmt": "yuv420p",
            "avg_frame_rate": "30000/1001",
            "r_frame_rate": "30/1",
            "has_b_frames": 2,
            "display_aspect_ratio": "16:9",
            "bit_rate": "200000",
        },
        {
            "index": 1,
            "codec_type": "audio",
            "codec_name": "aac",
            "sample_rate": "48000",
            "channels": 2,
            "channel_layout": "stereo",
            "bit_rate": "80000",
        },
    ],
}


# ---------------------------------------------------------------------------
# argv
# ---------------------------------------------------------------------------


def test_build_probe_cmd_shape():
    from video_info import build_probe_cmd
    cmd = build_probe_cmd("x.mp4")
    assert cmd[0] == "ffprobe"
    assert "-print_format" in cmd
    assert "json" in cmd
    assert "-show_format" in cmd
    assert "-show_streams" in cmd
    assert cmd[-1] == "x.mp4"


# ---------------------------------------------------------------------------
# fps parser
# ---------------------------------------------------------------------------


def test_parse_fps_normal():
    from video_info import _parse_fps
    assert _parse_fps("30000/1001") == pytest.approx(29.97, rel=0.01)
    assert _parse_fps("24/1") == 24.0


def test_parse_fps_zero_denominator_returns_none():
    from video_info import _parse_fps
    assert _parse_fps("0/0") is None
    assert _parse_fps("30/0") is None


def test_parse_fps_malformed_returns_none():
    from video_info import _parse_fps
    assert _parse_fps(None) is None
    assert _parse_fps("") is None
    assert _parse_fps("30") is None
    assert _parse_fps("a/b") is None


# ---------------------------------------------------------------------------
# parse_probe_json
# ---------------------------------------------------------------------------


def test_parse_probe_json_full_payload():
    from video_info import parse_probe_json
    info = parse_probe_json(json.dumps(_SAMPLE_PROBE), "x.mp4")
    d = info.to_dict()
    assert d["format_name"] == "mov,mp4,m4a,3gp,3g2,mj2"
    assert d["duration"] == pytest.approx(30.0)
    assert d["size_bytes"] == 1048576
    assert d["bitrate"] == 280000
    assert d["video"]["codec_name"] == "h264"
    assert d["video"]["width"] == 1920
    assert d["video"]["height"] == 1080
    assert d["video"]["fps"] == pytest.approx(29.97, rel=0.01)
    assert d["audio"]["codec_name"] == "aac"
    assert d["audio"]["sample_rate"] == 48000
    assert d["audio"]["channels"] == 2
    assert len(d["streams"]) == 2


def test_parse_probe_json_empty_payload():
    from video_info import parse_probe_json
    info = parse_probe_json("", "x.mp4")
    assert info.path == "x.mp4"
    assert info.duration is None
    assert info.video is None
    assert info.audio is None


def test_parse_probe_json_malformed_json():
    from video_info import parse_probe_json
    info = parse_probe_json("{{not json", "y.mp4")
    assert info.duration is None
    assert info.streams == []


def test_parse_probe_json_video_only():
    from video_info import parse_probe_json
    payload = {
        "format": {"format_name": "matroska,webm", "duration": "15"},
        "streams": [
            {"index": 0, "codec_type": "video", "codec_name": "vp9",
             "width": 640, "height": 360, "avg_frame_rate": "24/1"},
        ],
    }
    info = parse_probe_json(json.dumps(payload), "v.webm").to_dict()
    assert info["video"]["codec_name"] == "vp9"
    assert info["audio"] is None


def test_parse_probe_json_coerces_bad_duration():
    from video_info import parse_probe_json
    payload = {"format": {"duration": "N/A"}, "streams": []}
    info = parse_probe_json(json.dumps(payload), "x").to_dict()
    assert info["duration"] is None


# ---------------------------------------------------------------------------
# inspect_video dispatcher
# ---------------------------------------------------------------------------


def test_inspect_video_missing_source(tmp_path):
    from video_info import inspect_video
    r = inspect_video(str(tmp_path / "ghost.mp4"))
    assert r["ok"] is False


def test_inspect_video_missing_ffprobe(monkeypatch, tmp_path):
    from video_info import inspect_video
    src = tmp_path / "x.mp4"
    src.write_bytes(b"\x00" * 64)
    monkeypatch.setattr("video_info.shutil.which", lambda n: None)
    r = inspect_video(str(src))
    assert r["ok"] is False
    assert r.get("install_hint")


def test_inspect_video_non_zero_exit(monkeypatch, tmp_path):
    from video_info import inspect_video
    src = tmp_path / "x.mp4"
    src.write_bytes(b"\x00" * 64)
    monkeypatch.setattr("video_info.shutil.which", lambda n: "/bin/ffprobe")

    def runner(cmd):
        return _FakeProc(returncode=1, stderr="probe: unreadable")
    r = inspect_video(str(src), runner=runner)
    assert r["ok"] is False
    assert "unreadable" in r["error"]


def test_inspect_video_success(monkeypatch, tmp_path):
    from video_info import inspect_video
    src = tmp_path / "x.mp4"
    src.write_bytes(b"\x00" * 64)
    monkeypatch.setattr("video_info.shutil.which", lambda n: "/bin/ffprobe")

    def runner(cmd):
        return _FakeProc(returncode=0, stdout=json.dumps(_SAMPLE_PROBE))

    r = inspect_video(str(src), runner=runner)
    assert r["ok"] is True
    assert r["info"]["video"]["width"] == 1920


def test_inspect_video_backfills_size_from_disk(monkeypatch, tmp_path):
    from video_info import inspect_video
    src = tmp_path / "x.mp4"
    src.write_bytes(b"\x00" * 2048)
    monkeypatch.setattr("video_info.shutil.which", lambda n: "/bin/ffprobe")
    payload = {"format": {"format_name": "mp4"}, "streams": []}

    def runner(cmd):
        return _FakeProc(returncode=0, stdout=json.dumps(payload))
    r = inspect_video(str(src), runner=runner)
    assert r["ok"] is True
    assert r["info"]["size_bytes"] == 2048


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


def test_info_endpoint_rejects_missing_source(tmp_path, monkeypatch):
    _api, client = _fresh_api(monkeypatch, tmp_path)
    r = client.post(
        "/api/v1/edit/info",
        json={"video_path": str(tmp_path / "ghost.mp4")},
    )
    assert r.status_code == 422


def test_info_endpoint_returns_info_dict(tmp_path, monkeypatch):
    api, client = _fresh_api(monkeypatch, tmp_path)
    src = tmp_path / "x.mp4"
    src.write_bytes(b"\x00" * 64)

    def fake_inspect(path, **kwargs):
        return {"ok": True, "info": {"path": path, "duration": 42.0}}
    monkeypatch.setattr(api, "_inspect_video_impl", fake_inspect, raising=False)
    # Install helper reference used by endpoint:
    import video_info
    monkeypatch.setattr(video_info, "inspect_video", fake_inspect, raising=True)

    r = client.post("/api/v1/edit/info", json={"video_path": str(src)})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["info"]["duration"] == 42.0
