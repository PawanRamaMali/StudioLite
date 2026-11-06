"""Tests for chroma_key.py and the /api/v1/edit/chroma-key endpoint.

Covers:
- normalize_key_color turns #hex, bare hex, 0xhex and named colors into the
  canonical 0xRRGGBB form; unknown inputs fall back to green rather than
  raising (never fail a whole render on a typo).
- build_key_filter clamps similarity/blend into [0,1].
- Each build_*_cmd emits the argv shape we expect so a flag change needs
  an explicit update.
- remove_chroma_key surfaces missing source, missing/wrong-type backgrounds,
  unknown mode, missing ffmpeg, non-zero exit, no-output; three successful
  rendering paths (transparent, color, image).
- Endpoint 422s for missing source/background and dispatches the runner.
"""
from __future__ import annotations

import os
import subprocess

import pytest


class _FakeProc:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _mk_src(tmp_path):
    p = tmp_path / "in.mp4"
    p.write_bytes(b"\x00" * 4096)
    return str(p)


def _mk_png(tmp_path, name="bg.png"):
    p = tmp_path / name
    p.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 128)
    return str(p)


def _mk_bg_video(tmp_path, name="bg.mp4"):
    p = tmp_path / name
    p.write_bytes(b"\x00" * 4096)
    return str(p)


def _touch(path, size=4096):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"\x00" * size)


# ---------------------------------------------------------------------------
# normalize_key_color
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("raw, expected", [
    ("green",       "0x00FF00"),
    ("BLUE",        "0x0000FF"),
    ("#ff00ff",     "0xFF00FF"),
    ("00FF00",      "0x00FF00"),
    ("0xaabbcc",    "0xAABBCC"),
    ("not-a-color", "0x00FF00"),   # unknown -> green fallback
    ("",            "0x00FF00"),
    ("#12",         "0x00FF00"),   # too short -> fallback
])
def test_normalize_key_color(raw, expected):
    from chroma_key import normalize_key_color
    assert normalize_key_color(raw) == expected


# ---------------------------------------------------------------------------
# build_key_filter
# ---------------------------------------------------------------------------


def test_build_key_filter_clamps_similarity_and_blend():
    from chroma_key import build_key_filter, ChromaKeyStyle
    out = build_key_filter(ChromaKeyStyle(similarity=5.0, blend=-1.0))
    assert "similarity=1.000" in out
    assert "blend=0.000" in out
    assert "color=0x00FF00" in out


def test_build_key_filter_normalizes_hex():
    from chroma_key import build_key_filter, ChromaKeyStyle
    out = build_key_filter(ChromaKeyStyle(key_color="#00FFaa"))
    assert "color=0x00FFAA" in out


# ---------------------------------------------------------------------------
# build_transparent_cmd
# ---------------------------------------------------------------------------


def test_build_transparent_cmd_shape():
    from chroma_key import build_transparent_cmd, ChromaKeyStyle
    cmd = build_transparent_cmd("in.mp4", "out.webm",
                                 style=ChromaKeyStyle(audio="copy"))
    assert cmd[0] == "ffmpeg"
    assert cmd[cmd.index("-c:v") + 1] == "libvpx-vp9"
    assert cmd[cmd.index("-pix_fmt") + 1] == "yuva420p"
    assert cmd[cmd.index("-auto-alt-ref") + 1] == "0"
    vf = cmd[cmd.index("-vf") + 1]
    assert "chromakey=color=0x00FF00" in vf
    assert "format=yuva420p" in vf
    assert "-c:a" in cmd  # audio kept
    assert cmd[-1] == "out.webm"


def test_build_transparent_cmd_drop_audio_sets_minus_an():
    from chroma_key import build_transparent_cmd, ChromaKeyStyle
    cmd = build_transparent_cmd("in.mp4", "out.webm",
                                 style=ChromaKeyStyle(audio="drop"))
    assert "-an" in cmd
    assert "-c:a" not in cmd


# ---------------------------------------------------------------------------
# build_color_bg_cmd
# ---------------------------------------------------------------------------


def test_build_color_bg_cmd_includes_lavfi_color_source():
    from chroma_key import build_color_bg_cmd, ChromaKeyStyle
    cmd = build_color_bg_cmd(
        "in.mp4", "out.mp4",
        style=ChromaKeyStyle(background_color="0x202020",
                              out_width=1920, out_height=1080, fps=24),
    )
    graph = cmd[cmd.index("-filter_complex") + 1]
    assert "color=c=0x202020:s=1920x1080:r=24" in graph
    assert "chromakey=color=0x00FF00" in graph
    assert "overlay=shortest=1" in graph
    assert cmd[cmd.index("-map") + 1] == "[vout]"


# ---------------------------------------------------------------------------
# build_image_bg_cmd / build_video_bg_cmd
# ---------------------------------------------------------------------------


def test_build_image_bg_cmd_loops_the_still():
    from chroma_key import build_image_bg_cmd, ChromaKeyStyle
    cmd = build_image_bg_cmd(
        "in.mp4", "out.mp4",
        background_path="bg.png",
        style=ChromaKeyStyle(out_width=1280, out_height=720),
    )
    # The still needs -loop 1 to be held for the whole clip duration.
    assert "-loop" in cmd and cmd[cmd.index("-loop") + 1] == "1"
    graph = cmd[cmd.index("-filter_complex") + 1]
    assert "scale=1280:720" in graph
    assert "pad=1280:720" in graph
    assert "overlay=shortest=1" in graph
    assert "-shortest" in cmd


def test_build_video_bg_cmd_no_loop_flag():
    from chroma_key import build_video_bg_cmd, ChromaKeyStyle
    cmd = build_video_bg_cmd(
        "in.mp4", "out.mp4",
        background_path="bg.mp4",
        style=ChromaKeyStyle(),
    )
    # Video backgrounds play natively - no -loop prefix.
    assert "-loop" not in cmd
    assert "bg.mp4" in cmd
    assert "-shortest" in cmd


# ---------------------------------------------------------------------------
# remove_chroma_key - failure paths
# ---------------------------------------------------------------------------


def test_remove_chroma_key_missing_source(tmp_path):
    from chroma_key import remove_chroma_key
    r = remove_chroma_key(
        str(tmp_path / "ghost.mp4"),
        str(tmp_path / "out.webm"),
    )
    assert r["ok"] is False
    assert "source not found" in r["error"]


def test_remove_chroma_key_unknown_mode(tmp_path):
    from chroma_key import remove_chroma_key
    r = remove_chroma_key(
        _mk_src(tmp_path),
        str(tmp_path / "out.mp4"),
        mode="holographic",
    )
    assert r["ok"] is False
    assert "unknown mode" in r["error"]


def test_remove_chroma_key_image_mode_requires_background(tmp_path):
    from chroma_key import remove_chroma_key
    r = remove_chroma_key(
        _mk_src(tmp_path),
        str(tmp_path / "out.mp4"),
        mode="image",
        background_path=str(tmp_path / "ghost.png"),
    )
    assert r["ok"] is False
    assert "background not found" in r["error"]


def test_remove_chroma_key_image_mode_rejects_video_bg(tmp_path):
    from chroma_key import remove_chroma_key
    r = remove_chroma_key(
        _mk_src(tmp_path),
        str(tmp_path / "out.mp4"),
        mode="image",
        background_path=_mk_bg_video(tmp_path),  # .mp4
    )
    assert r["ok"] is False
    assert "image background" in r["error"]


def test_remove_chroma_key_video_mode_rejects_image_bg(tmp_path):
    from chroma_key import remove_chroma_key
    r = remove_chroma_key(
        _mk_src(tmp_path),
        str(tmp_path / "out.mp4"),
        mode="video",
        background_path=_mk_png(tmp_path),  # .png
    )
    assert r["ok"] is False
    assert "video background" in r["error"]


def test_remove_chroma_key_ffmpeg_missing(tmp_path):
    from chroma_key import remove_chroma_key

    def fake(_):
        raise FileNotFoundError("ffmpeg")

    r = remove_chroma_key(
        _mk_src(tmp_path), str(tmp_path / "out.webm"), runner=fake,
    )
    assert r["ok"] is False
    assert "install_hint" in r


def test_remove_chroma_key_nonzero_exit_surfaces_tail(tmp_path):
    from chroma_key import remove_chroma_key

    def fake(_):
        return _FakeProc(returncode=1, stderr="meh\n[fatal] no libvpx-vp9")

    r = remove_chroma_key(
        _mk_src(tmp_path), str(tmp_path / "out.webm"), runner=fake,
    )
    assert r["ok"] is False
    assert "no libvpx-vp9" in r["error"]


def test_remove_chroma_key_empty_output_detected(tmp_path):
    from chroma_key import remove_chroma_key

    def fake(cmd):
        _touch(cmd[-1], size=32)  # too small to be real output
        return _FakeProc(returncode=0)

    r = remove_chroma_key(
        _mk_src(tmp_path), str(tmp_path / "out.webm"), runner=fake,
    )
    assert r["ok"] is False
    assert "no output" in r["error"]


# ---------------------------------------------------------------------------
# remove_chroma_key - success paths
# ---------------------------------------------------------------------------


def test_remove_chroma_key_transparent_forces_webm_extension(tmp_path):
    from chroma_key import remove_chroma_key

    def fake(cmd):
        _touch(cmd[-1], 8192)
        return _FakeProc(returncode=0)

    # Caller asks for .mp4 but transparent mode requires webm; we rewrite.
    requested = str(tmp_path / "out.mp4")
    r = remove_chroma_key(
        _mk_src(tmp_path), requested, mode="transparent", runner=fake,
    )
    assert r["ok"] is True
    assert r["output_path"].endswith(".webm")
    assert r["mode"] == "transparent"


def test_remove_chroma_key_color_bg_succeeds(tmp_path):
    from chroma_key import remove_chroma_key, ChromaKeyStyle

    def fake(cmd):
        _touch(cmd[-1], 8192)
        return _FakeProc(returncode=0)

    r = remove_chroma_key(
        _mk_src(tmp_path), str(tmp_path / "out.mp4"),
        mode="color",
        style=ChromaKeyStyle(background_color="0x101010"),
        runner=fake,
    )
    assert r["ok"] is True
    assert r["mode"] == "color"


def test_remove_chroma_key_image_bg_succeeds(tmp_path):
    from chroma_key import remove_chroma_key

    def fake(cmd):
        _touch(cmd[-1], 8192)
        return _FakeProc(returncode=0)

    r = remove_chroma_key(
        _mk_src(tmp_path), str(tmp_path / "out.mp4"),
        mode="image",
        background_path=_mk_png(tmp_path),
        runner=fake,
    )
    assert r["ok"] is True
    assert r["mode"] == "image"


# ---------------------------------------------------------------------------
# /api/v1/edit/chroma-key endpoint
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


def test_chroma_key_endpoint_rejects_missing_video(tmp_path, monkeypatch):
    _api, client = _fresh_api(monkeypatch, tmp_path)
    r = client.post(
        "/api/v1/edit/chroma-key",
        json={"video_path": str(tmp_path / "ghost.mp4")},
    )
    assert r.status_code == 422
    assert "not found" in r.json()["detail"].lower()


def test_chroma_key_endpoint_rejects_missing_background(tmp_path, monkeypatch):
    _api, client = _fresh_api(monkeypatch, tmp_path)
    src = tmp_path / "in.mp4"
    src.write_bytes(b"\x00" * 64)
    r = client.post(
        "/api/v1/edit/chroma-key",
        json={
            "video_path": str(src),
            "mode": "image",
            "background_path": str(tmp_path / "ghost.png"),
        },
    )
    assert r.status_code == 422
    assert "background not found" in r.json()["detail"].lower()


def test_chroma_key_endpoint_dispatches_runner(tmp_path, monkeypatch):
    api, client = _fresh_api(monkeypatch, tmp_path)
    src = tmp_path / "in.mp4"
    src.write_bytes(b"\x00" * 64)

    seen = {}

    def fake_run(job_id, req):
        seen["job_id"] = job_id
        seen["video_path"] = req.video_path
        seen["mode"] = req.mode
        seen["similarity"] = req.similarity
        seen["blend"] = req.blend

    monkeypatch.setattr(api, "_run_chroma_key", fake_run, raising=True)

    class _Immediate:
        def __init__(self, *, target, args, daemon=False):
            self.target, self.args = target, args
        def start(self):
            self.target(*self.args)

    monkeypatch.setattr(api.threading, "Thread", _Immediate, raising=True)

    r = client.post(
        "/api/v1/edit/chroma-key",
        json={
            "video_path": str(src),
            "mode": "color",
            "similarity": 0.25,
            "blend": 0.1,
            "background_color": "#000000",
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["job_id"] == seen["job_id"]
    assert seen["mode"] == "color"
    assert seen["similarity"] == 0.25
    assert seen["blend"] == 0.1


def test_chroma_key_request_defaults():
    from api_server import ChromaKeyRequest
    req = ChromaKeyRequest(video_path="x.mp4")
    assert req.mode == "transparent"
    assert req.key_color == "green"
    assert req.similarity == 0.15
    assert req.blend == 0.08
    assert req.out_width == 1280
    assert req.out_height == 720
    assert req.fps == 30
    assert req.audio == "copy"
    assert req.background_color == "0x000000"
    assert req.background_path is None
