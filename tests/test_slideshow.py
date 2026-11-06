"""Tests for slideshow.py and the /api/v1/edit/slideshow endpoint."""
from __future__ import annotations

import os
import subprocess

import pytest


class _FakeProc:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _mk_png(tmp_path, name="a.png"):
    p = tmp_path / name
    p.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 128)
    return str(p)


def _touch(path, size=1024):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"\x00" * size)


# ---------------------------------------------------------------------------
# kenburns_expr
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("move, marker", [
    ("zoom-in",  "z='min(zoom+0.0010,1.20)'"),
    ("zoom-out", "z='if(lte(zoom,1.0),1.20"),
    ("pan-left", "x='(iw-iw/zoom)*(1-on/"),
    ("pan-right", "x='(iw-iw/zoom)*on/"),
    ("none",     "z='1.0'"),
])
def test_kenburns_expr_variants(move, marker):
    from slideshow import kenburns_expr
    assert marker in kenburns_expr(move, 60, 30)


def test_kenburns_expr_unknown_treated_as_none():
    from slideshow import kenburns_expr
    assert "z='1.0'" in kenburns_expr("tracking", 60, 30)


def test_kenburns_expr_duration_encoded():
    from slideshow import kenburns_expr
    assert kenburns_expr("zoom-in", 90, 30).rstrip().endswith("d=90")


# ---------------------------------------------------------------------------
# build_image_clip_cmd
# ---------------------------------------------------------------------------


def test_build_image_clip_cmd_static_no_zoompan():
    from slideshow import build_image_clip_cmd, SlideshowStyle
    style = SlideshowStyle(width=1280, height=720, fps=24)
    cmd = build_image_clip_cmd("in.png", "out.mp4", duration=3.0, style=style)
    vf = cmd[cmd.index("-vf") + 1]
    assert "scale=1280:720" in vf
    assert "pad=1280:720" in vf
    assert "zoompan" not in vf  # none mode stays cheap
    assert "-t" in cmd and cmd[cmd.index("-t") + 1] == "3.000"


def test_build_image_clip_cmd_kenburns_uses_zoompan():
    from slideshow import build_image_clip_cmd, SlideshowStyle
    style = SlideshowStyle(width=1280, height=720, fps=30,
                           kenburns="zoom-in")
    cmd = build_image_clip_cmd("in.png", "out.mp4", duration=2.0, style=style)
    vf = cmd[cmd.index("-vf") + 1]
    assert "zoompan" in vf
    # Larger pre-source so the crop doesn't blur under zoom.
    assert "scale=2560:1440" in vf
    assert "s=1280x720" in vf


# ---------------------------------------------------------------------------
# build_xfade_graph
# ---------------------------------------------------------------------------


def test_build_xfade_graph_two_clips():
    from slideshow import build_xfade_graph, SlideshowStyle
    g = build_xfade_graph(
        ["a.mp4", "b.mp4"],
        style=SlideshowStyle(transition="fade", transition_duration=0.5),
        durations=[3.0, 3.0],
    )
    # One xfade step with offset = 3.0 - 0.5 = 2.5
    assert "xfade=transition=fade" in g
    assert "offset=2.500" in g


def test_build_xfade_graph_three_clips_chains_offsets():
    from slideshow import build_xfade_graph, SlideshowStyle
    g = build_xfade_graph(
        ["a.mp4", "b.mp4", "c.mp4"],
        style=SlideshowStyle(transition="dissolve", transition_duration=1.0),
        durations=[4.0, 3.0, 3.0],
    )
    # Steps:
    #   accum1 = 4 - 1 = 3    -> offset=3.000
    #   accum2 = (4-1) + (3-1) = 5  -> offset=5.000
    assert "offset=3.000" in g
    assert "offset=5.000" in g
    assert g.count("xfade=") == 2


def test_build_xfade_graph_single_clip_is_empty():
    from slideshow import build_xfade_graph, SlideshowStyle
    assert build_xfade_graph(["a.mp4"], style=SlideshowStyle(),
                              durations=[3.0]) == ""


# ---------------------------------------------------------------------------
# total_duration
# ---------------------------------------------------------------------------


def test_total_duration_without_transition():
    from slideshow import total_duration, SlideshowStyle
    assert total_duration([3, 4, 5],
                           SlideshowStyle(transition="none")) == 12


def test_total_duration_with_transition_subtracts_overlap():
    from slideshow import total_duration, SlideshowStyle
    out = total_duration(
        [3, 3, 3],
        SlideshowStyle(transition="fade", transition_duration=0.5),
    )
    # 3 clips, 2 transitions, each steals 0.5s overlap -> 9 - 1.0 = 8.0.
    assert out == pytest.approx(8.0)


def test_total_duration_empty():
    from slideshow import total_duration, SlideshowStyle
    assert total_duration([], SlideshowStyle()) == 0.0


# ---------------------------------------------------------------------------
# generate_slideshow - failure modes
# ---------------------------------------------------------------------------


def test_generate_slideshow_requires_images(tmp_path):
    from slideshow import generate_slideshow
    r = generate_slideshow([], str(tmp_path / "out.mp4"))
    assert r["ok"] is False
    assert "at least one image" in r["error"]


def test_generate_slideshow_missing_image(tmp_path):
    from slideshow import generate_slideshow
    r = generate_slideshow([str(tmp_path / "ghost.png")],
                            str(tmp_path / "out.mp4"))
    assert r["ok"] is False
    assert "image not found" in r["error"]


def test_generate_slideshow_missing_audio(tmp_path):
    from slideshow import generate_slideshow
    r = generate_slideshow(
        [_mk_png(tmp_path, "a.png")],
        str(tmp_path / "out.mp4"),
        audio_path=str(tmp_path / "ghost.mp3"),
    )
    assert r["ok"] is False
    assert "audio not found" in r["error"]


def test_generate_slideshow_rejects_unknown_transition(tmp_path):
    from slideshow import generate_slideshow, SlideshowStyle
    r = generate_slideshow(
        [_mk_png(tmp_path, "a.png"), _mk_png(tmp_path, "b.png")],
        str(tmp_path / "out.mp4"),
        style=SlideshowStyle(transition="whirligig"),
    )
    assert r["ok"] is False
    assert "unknown transition" in r["error"]


def test_generate_slideshow_rejects_unknown_kenburns(tmp_path):
    from slideshow import generate_slideshow, SlideshowStyle
    r = generate_slideshow(
        [_mk_png(tmp_path, "a.png")],
        str(tmp_path / "out.mp4"),
        style=SlideshowStyle(kenburns="spiral"),
    )
    assert r["ok"] is False
    assert "unknown kenburns" in r["error"]


def test_generate_slideshow_rejects_too_long_transition(tmp_path):
    from slideshow import generate_slideshow, SlideshowStyle
    # 2s per image, transition 2s -> would overrun, must 422.
    r = generate_slideshow(
        [_mk_png(tmp_path, "a.png"), _mk_png(tmp_path, "b.png")],
        str(tmp_path / "out.mp4"),
        style=SlideshowStyle(transition="fade", transition_duration=2.0,
                              per_image_duration=2.0),
    )
    assert r["ok"] is False
    assert "transition_duration" in r["error"]


def test_generate_slideshow_ffmpeg_missing_install_hint(tmp_path):
    from slideshow import generate_slideshow

    def fake(_):
        raise FileNotFoundError("ffmpeg")

    r = generate_slideshow(
        [_mk_png(tmp_path, "a.png")],
        str(tmp_path / "out.mp4"),
        runner=fake,
    )
    assert r["ok"] is False
    assert "install_hint" in r


def test_generate_slideshow_ffmpeg_nonzero_surfaces_tail(tmp_path):
    from slideshow import generate_slideshow

    def fake(cmd):
        return _FakeProc(returncode=1, stderr="line1\n[boom] codec missing")

    r = generate_slideshow(
        [_mk_png(tmp_path, "a.png")],
        str(tmp_path / "out.mp4"),
        runner=fake,
    )
    assert r["ok"] is False
    assert "codec missing" in r["error"]


# ---------------------------------------------------------------------------
# generate_slideshow - success paths
# ---------------------------------------------------------------------------


def test_generate_slideshow_no_audio_no_transition(tmp_path):
    from slideshow import generate_slideshow, SlideshowStyle
    imgs = [_mk_png(tmp_path, f"img{i}.png") for i in range(3)]
    out = tmp_path / "out.mp4"

    def fake(cmd):
        # Clip-render and concat commands all take `-y` and the last token
        # is the destination; touch that.
        _touch(cmd[-1], 8192)
        return _FakeProc(returncode=0)

    r = generate_slideshow(
        imgs, str(out),
        style=SlideshowStyle(transition="none", per_image_duration=2.0),
        runner=fake,
    )
    assert r["ok"], r
    assert r["image_count"] == 3
    assert r["duration_sec"] == pytest.approx(6.0)
    assert os.path.isfile(out)


def test_generate_slideshow_with_transition_builds_xfade_graph(tmp_path):
    from slideshow import generate_slideshow, SlideshowStyle
    imgs = [_mk_png(tmp_path, f"img{i}.png") for i in range(3)]
    out = tmp_path / "out.mp4"
    graphs = []

    def fake(cmd):
        if "-filter_complex" in cmd:
            graphs.append(cmd[cmd.index("-filter_complex") + 1])
        _touch(cmd[-1], 8192)
        return _FakeProc(returncode=0)

    r = generate_slideshow(
        imgs, str(out),
        style=SlideshowStyle(transition="dissolve", transition_duration=0.5,
                              per_image_duration=3.0),
        runner=fake,
    )
    assert r["ok"], r
    assert any("xfade=transition=dissolve" in g for g in graphs)
    # 3 images, 2 transitions: total = 9 - 1.0 = 8.0.
    assert r["duration_sec"] == pytest.approx(8.0)


def test_generate_slideshow_with_audio_mux(tmp_path):
    from slideshow import generate_slideshow, SlideshowStyle
    imgs = [_mk_png(tmp_path, f"img{i}.png") for i in range(2)]
    audio = tmp_path / "bg.mp3"
    audio.write_bytes(b"\xff\xfb\x00" + b"\x00" * 128)
    out = tmp_path / "out.mp4"

    calls = []

    def fake(cmd):
        calls.append(cmd)
        _touch(cmd[-1], 8192)
        return _FakeProc(returncode=0)

    r = generate_slideshow(
        imgs, str(out),
        audio_path=str(audio),
        style=SlideshowStyle(transition="none", per_image_duration=2.0,
                              audio_volume=0.5),
        runner=fake,
    )
    assert r["ok"], r
    assert r.get("has_audio") is True
    # The last ffmpeg invocation should be the audio mux, with a shortest
    # flag and the audio file as a second -i.
    last = calls[-1]
    assert "-shortest" in last
    assert "-c:a" in last and last[last.index("-c:a") + 1] == "aac"
    assert str(audio) in last


def test_generate_slideshow_per_image_durations_override(tmp_path):
    from slideshow import generate_slideshow, SlideshowStyle
    imgs = [_mk_png(tmp_path, f"img{i}.png") for i in range(3)]
    out = tmp_path / "out.mp4"

    def fake(cmd):
        _touch(cmd[-1], 8192)
        return _FakeProc(returncode=0)

    r = generate_slideshow(
        imgs, str(out),
        durations=[5.0, 2.0, 8.0],
        style=SlideshowStyle(transition="none", per_image_duration=99.0),
        runner=fake,
    )
    assert r["ok"], r
    # style default of 99 is ignored because explicit durations covered all images.
    assert r["duration_sec"] == pytest.approx(15.0)


# ---------------------------------------------------------------------------
# /api/v1/edit/slideshow endpoint
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


def test_slideshow_endpoint_rejects_empty_images(tmp_path, monkeypatch):
    _api, client = _fresh_api(monkeypatch, tmp_path)
    r = client.post("/api/v1/edit/slideshow", json={"images": []})
    assert r.status_code == 422


def test_slideshow_endpoint_rejects_missing_image(tmp_path, monkeypatch):
    _api, client = _fresh_api(monkeypatch, tmp_path)
    r = client.post(
        "/api/v1/edit/slideshow",
        json={"images": [str(tmp_path / "ghost.png")]},
    )
    assert r.status_code == 422
    assert "not found" in r.json()["detail"].lower()


def test_slideshow_endpoint_dispatches_runner(tmp_path, monkeypatch):
    api, client = _fresh_api(monkeypatch, tmp_path)
    img = tmp_path / "a.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)

    seen = {}

    def fake_run(job_id, req):
        seen["job_id"] = job_id
        seen["images"] = list(req.images)
        seen["transition"] = req.transition
        seen["kenburns"] = req.kenburns

    monkeypatch.setattr(api, "_run_slideshow", fake_run, raising=True)

    class _Immediate:
        def __init__(self, *, target, args, daemon=False):
            self.target, self.args = target, args
        def start(self):
            self.target(*self.args)

    monkeypatch.setattr(api.threading, "Thread", _Immediate, raising=True)

    r = client.post(
        "/api/v1/edit/slideshow",
        json={
            "images": [str(img)],
            "per_image_duration": 2,
            "transition": "fade",
            "kenburns": "zoom-in",
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["job_id"] == seen["job_id"]
    assert seen["images"] == [str(img)]
    assert seen["transition"] == "fade"
    assert seen["kenburns"] == "zoom-in"


def test_slideshow_request_defaults():
    from api_server import SlideshowRequest
    req = SlideshowRequest(images=["a.png"])
    assert req.width == 1920
    assert req.height == 1080
    assert req.fps == 30
    assert req.per_image_duration == 3.0
    assert req.transition == "fade"
    assert req.transition_duration == 0.5
    assert req.kenburns == "none"
    assert req.audio_path is None
