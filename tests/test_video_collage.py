"""Tests for video_collage.py and the /api/v1/edit/collage endpoint.

Covers:
- Layout lookup table (names, min/max inputs).
- xstack layout expression for 2x2 and 3x3.
- Filter-graph shape: scale+pad chain per input, hstack/vstack/xstack
  switch based on layout, pip builds base+inset+overlay graph.
- Audio mapping variants (first, mix, drop).
- Argv assembly: correct -filter_complex, -map, codec flags.
- build_collage input validation (unknown layout, too few/too many clips,
  missing source) and the ffmpeg failure paths; success path.
- Endpoint 422 on missing clips and dispatches runner.
"""
from __future__ import annotations

import os

import pytest


class _FakeProc:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _mk_video(tmp_path, name):
    p = tmp_path / name
    p.write_bytes(b"\x00" * 16 + b"ftyp" + b"\x00" * 128)
    return str(p)


def _touch(path, size=4096):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"\x00" * size)


# ---------------------------------------------------------------------------
# Layout table
# ---------------------------------------------------------------------------


def test_layouts_has_expected_entries():
    from video_collage import LAYOUTS
    assert "hstack_2" in LAYOUTS
    assert "grid_2x2" in LAYOUTS
    assert "grid_3x3" in LAYOUTS
    assert "pip" in LAYOUTS
    cols, rows, mn, mx = LAYOUTS["grid_2x2"]
    assert (cols, rows) == (2, 2)
    assert mn == 2 and mx == 4


# ---------------------------------------------------------------------------
# xstack layout
# ---------------------------------------------------------------------------


def test_build_xstack_layout_2x2():
    from video_collage import build_xstack_layout
    assert build_xstack_layout(2, 2) == "0_0|w0_0|0_h0|w0_h0"


def test_build_xstack_layout_3x3_mentions_each_cell():
    from video_collage import build_xstack_layout
    layout = build_xstack_layout(3, 3)
    assert len(layout.split("|")) == 9
    # Bottom-right cell uses sum of upstream widths and heights
    assert "w0+w1_h0+h1" in layout


# ---------------------------------------------------------------------------
# Filter graph
# ---------------------------------------------------------------------------


def test_build_stack_filter_complex_hstack():
    from video_collage import build_stack_filter_complex, CollageStyle
    g = build_stack_filter_complex(2, style=CollageStyle(layout="hstack_2"))
    assert "[0:v]scale=" in g
    assert "[1:v]scale=" in g
    assert "[c0][c1]hstack=inputs=2[v]" in g


def test_build_stack_filter_complex_grid_uses_xstack():
    from video_collage import build_stack_filter_complex, CollageStyle
    g = build_stack_filter_complex(4, style=CollageStyle(layout="grid_2x2"))
    assert "xstack=inputs=4:layout=" in g
    assert "fill=black" in g


def test_build_stack_filter_complex_fewer_inputs_than_slots():
    """grid_2x2 with 3 inputs: xstack=inputs=3, not 4."""
    from video_collage import build_stack_filter_complex, CollageStyle
    g = build_stack_filter_complex(3, style=CollageStyle(layout="grid_2x2"))
    assert "xstack=inputs=3:" in g
    assert "[c0][c1][c2]" in g
    assert "[c3]" not in g


def test_build_pip_filter_complex_bottom_right():
    from video_collage import build_pip_filter_complex, CollageStyle
    g = build_pip_filter_complex(CollageStyle(layout="pip", pip_position="bottom-right"))
    assert "[base][pip]overlay=main_w-overlay_w-20:main_h-overlay_h-20:shortest=1[v]" in g
    assert "iw*0.280" in g


def test_build_pip_filter_complex_unknown_position_falls_back():
    from video_collage import build_pip_filter_complex, CollageStyle
    g = build_pip_filter_complex(CollageStyle(layout="pip", pip_position="nowhere"))
    assert "main_w-overlay_w-20:main_h-overlay_h-20" in g


def test_build_pip_filter_complex_clamps_scale():
    from video_collage import build_pip_filter_complex, CollageStyle
    g = build_pip_filter_complex(CollageStyle(layout="pip", pip_scale=5.0))
    # Scale clamped at 0.5
    assert "iw*0.500" in g


# ---------------------------------------------------------------------------
# build_collage_cmd argv
# ---------------------------------------------------------------------------


def test_build_collage_cmd_has_filter_complex_and_map(tmp_path):
    from video_collage import build_collage_cmd, CollageStyle
    a = _mk_video(tmp_path, "a.mp4")
    b = _mk_video(tmp_path, "b.mp4")
    cmd = build_collage_cmd([a, b], str(tmp_path / "out.mp4"),
                            style=CollageStyle(layout="hstack_2"))
    assert cmd[0] == "ffmpeg"
    assert "-filter_complex" in cmd
    assert "-map" in cmd
    i = cmd.index("-map")
    assert cmd[i + 1] == "[v]"
    # audio default -> first input's audio
    assert "0:a?" in cmd
    # Codec args
    assert "libx264" in cmd
    assert "yuv420p" in cmd


def test_build_collage_cmd_audio_drop_adds_an(tmp_path):
    from video_collage import build_collage_cmd, CollageStyle
    a = _mk_video(tmp_path, "a.mp4")
    b = _mk_video(tmp_path, "b.mp4")
    cmd = build_collage_cmd([a, b], str(tmp_path / "out.mp4"),
                            style=CollageStyle(layout="hstack_2", audio="drop"))
    assert "-an" in cmd
    assert "0:a?" not in cmd


def test_build_collage_cmd_audio_mix_embeds_amix(tmp_path):
    from video_collage import build_collage_cmd, CollageStyle
    a = _mk_video(tmp_path, "a.mp4")
    b = _mk_video(tmp_path, "b.mp4")
    cmd = build_collage_cmd([a, b], str(tmp_path / "out.mp4"),
                            style=CollageStyle(layout="grid_2x2", audio="mix"))
    fc = cmd[cmd.index("-filter_complex") + 1]
    assert "amix=inputs=2" in fc
    # Two -map entries: [v] and [a]
    maps = [cmd[i + 1] for i, v in enumerate(cmd) if v == "-map"]
    assert "[v]" in maps and "[a]" in maps


def test_build_collage_cmd_crf_clamps(tmp_path):
    from video_collage import build_collage_cmd, CollageStyle
    a = _mk_video(tmp_path, "a.mp4")
    b = _mk_video(tmp_path, "b.mp4")
    cmd = build_collage_cmd([a, b], str(tmp_path / "out.mp4"),
                            style=CollageStyle(layout="hstack_2", crf=999))
    i = cmd.index("-crf")
    assert cmd[i + 1] == "51"


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def test_build_collage_rejects_unknown_layout(tmp_path):
    from video_collage import build_collage, CollageStyle
    r = build_collage([_mk_video(tmp_path, "a.mp4")], str(tmp_path / "out.mp4"),
                      style=CollageStyle(layout="nope"))
    assert r["ok"] is False
    assert "unknown" in r["error"].lower()


def test_build_collage_rejects_too_few_clips(tmp_path):
    from video_collage import build_collage, CollageStyle
    r = build_collage([_mk_video(tmp_path, "a.mp4")], str(tmp_path / "out.mp4"),
                      style=CollageStyle(layout="hstack_2"))
    assert r["ok"] is False
    assert "at least 2" in r["error"]


def test_build_collage_rejects_too_many_clips(tmp_path):
    from video_collage import build_collage, CollageStyle
    srcs = [_mk_video(tmp_path, f"v{i}.mp4") for i in range(5)]
    r = build_collage(srcs, str(tmp_path / "out.mp4"),
                      style=CollageStyle(layout="grid_2x2"))
    assert r["ok"] is False
    assert "at most 4" in r["error"]


def test_build_collage_rejects_missing_source(tmp_path):
    from video_collage import build_collage, CollageStyle
    a = _mk_video(tmp_path, "a.mp4")
    r = build_collage([a, str(tmp_path / "ghost.mp4")], str(tmp_path / "out.mp4"),
                      style=CollageStyle(layout="hstack_2"))
    assert r["ok"] is False
    assert "not found" in r["error"].lower()


def test_build_collage_missing_ffmpeg(monkeypatch, tmp_path):
    from video_collage import build_collage, CollageStyle
    monkeypatch.setattr("video_collage.shutil.which", lambda n: None)
    srcs = [_mk_video(tmp_path, "a.mp4"), _mk_video(tmp_path, "b.mp4")]
    r = build_collage(srcs, str(tmp_path / "out.mp4"),
                      style=CollageStyle(layout="hstack_2"))
    assert r["ok"] is False
    assert r.get("install_hint")


def test_build_collage_non_zero_exit(monkeypatch, tmp_path):
    from video_collage import build_collage, CollageStyle
    monkeypatch.setattr("video_collage.shutil.which", lambda n: "/bin/ffmpeg")
    srcs = [_mk_video(tmp_path, "a.mp4"), _mk_video(tmp_path, "b.mp4")]

    def runner(cmd):
        return _FakeProc(returncode=1, stderr="xstack: bad layout")
    r = build_collage(srcs, str(tmp_path / "out.mp4"),
                      style=CollageStyle(layout="grid_2x2"), runner=runner)
    assert r["ok"] is False
    assert "xstack" in r["error"]


def test_build_collage_empty_output(monkeypatch, tmp_path):
    from video_collage import build_collage, CollageStyle
    monkeypatch.setattr("video_collage.shutil.which", lambda n: "/bin/ffmpeg")
    srcs = [_mk_video(tmp_path, "a.mp4"), _mk_video(tmp_path, "b.mp4")]
    out = str(tmp_path / "out.mp4")

    def runner(cmd):
        return _FakeProc(returncode=0)  # does NOT create the file
    r = build_collage(srcs, out, style=CollageStyle(layout="hstack_2"), runner=runner)
    assert r["ok"] is False
    assert "no output" in r["error"].lower()


def test_build_collage_success(monkeypatch, tmp_path):
    from video_collage import build_collage, CollageStyle
    monkeypatch.setattr("video_collage.shutil.which", lambda n: "/bin/ffmpeg")
    srcs = [_mk_video(tmp_path, "a.mp4"), _mk_video(tmp_path, "b.mp4"),
            _mk_video(tmp_path, "c.mp4"), _mk_video(tmp_path, "d.mp4")]
    out = str(tmp_path / "grid.mp4")

    def runner(cmd):
        _touch(out, 4096)
        return _FakeProc(returncode=0)

    r = build_collage(srcs, out, style=CollageStyle(layout="grid_2x2"), runner=runner)
    assert r["ok"] is True
    assert r["backend"].endswith("grid_2x2")
    assert r["cells"] == 4


def test_build_collage_pip_success(monkeypatch, tmp_path):
    from video_collage import build_collage, CollageStyle
    monkeypatch.setattr("video_collage.shutil.which", lambda n: "/bin/ffmpeg")
    srcs = [_mk_video(tmp_path, "base.mp4"), _mk_video(tmp_path, "inset.mp4")]
    out = str(tmp_path / "pip.mp4")

    def runner(cmd):
        _touch(out, 4096)
        return _FakeProc(returncode=0)

    r = build_collage(srcs, out, style=CollageStyle(layout="pip"), runner=runner)
    assert r["ok"] is True
    assert r["cells"] == 1


# ---------------------------------------------------------------------------
# /api/v1/edit/collage
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


def test_collage_endpoint_rejects_too_few(tmp_path, monkeypatch):
    _api, client = _fresh_api(monkeypatch, tmp_path)
    r = client.post(
        "/api/v1/edit/collage",
        json={"video_paths": [str(tmp_path / "solo.mp4")], "layout": "hstack_2"},
    )
    assert r.status_code == 422


def test_collage_endpoint_rejects_missing_source(tmp_path, monkeypatch):
    _api, client = _fresh_api(monkeypatch, tmp_path)
    good = tmp_path / "a.mp4"
    good.write_bytes(b"\x00" * 64)
    r = client.post(
        "/api/v1/edit/collage",
        json={
            "video_paths": [str(good), str(tmp_path / "missing.mp4")],
            "layout": "hstack_2",
        },
    )
    assert r.status_code == 422


def test_collage_endpoint_dispatches_runner(tmp_path, monkeypatch):
    api, client = _fresh_api(monkeypatch, tmp_path)
    a = tmp_path / "a.mp4"
    b = tmp_path / "b.mp4"
    a.write_bytes(b"\x00" * 64)
    b.write_bytes(b"\x00" * 64)

    seen = {}

    def fake_run(job_id, req):
        seen["layout"] = req.layout
        seen["paths"] = list(req.video_paths)
        seen["audio"] = req.audio
    monkeypatch.setattr(api, "_run_collage", fake_run, raising=True)
    monkeypatch.setattr(api.threading, "Thread", _Immediate, raising=True)

    r = client.post(
        "/api/v1/edit/collage",
        json={
            "video_paths": [str(a), str(b)],
            "layout": "grid_2x2",
            "audio": "mix",
        },
    )
    assert r.status_code == 200
    assert seen["layout"] == "grid_2x2"
    assert seen["audio"] == "mix"
