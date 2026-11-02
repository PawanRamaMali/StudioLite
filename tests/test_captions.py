"""Tests for captions.py and the /api/v1/edit/caption endpoint.

Covers:
- format_timestamp_srt handles whole seconds, fractional seconds, the
  rollover on 999.5ms, and negative inputs.
- segments_to_srt drops empty text, numbers from 1, and clamps bad
  end<=start to start+0.5s.
- _color_to_ass accepts named colors, #RRGGBB and bare RRGGBB, falls
  back to white rather than failing the whole burn.
- _escape_srt_path_for_ffmpeg handles Windows drive-letter colons and
  the quote character.
- build_subtitles_filter / build_ffmpeg_cmd assemble exactly the argv
  we expect (so a future ffmpeg flag change needs an explicit update).
- burn_captions_into_video surfaces missing source, missing SRT, non-
  zero ffmpeg exit, and missing-ffmpeg-binary as clean failure dicts.
- /api/v1/edit/caption validates inputs and dispatches to the runner.
"""
from __future__ import annotations

import os
import subprocess
import sys
import types

import pytest


# ---------------------------------------------------------------------------
# format_timestamp_srt
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("secs, expected", [
    (0, "00:00:00,000"),
    (1.5, "00:00:01,500"),
    (3661.25, "01:01:01,250"),
    (0.999, "00:00:00,999"),
    # 999.5 ms rounds to 1000 which must carry into the seconds field.
    (0.9995, "00:00:01,000"),
    (-5, "00:00:00,000"),
])
def test_format_timestamp_srt(secs, expected):
    from captions import format_timestamp_srt
    assert format_timestamp_srt(secs) == expected


# ---------------------------------------------------------------------------
# segments_to_srt
# ---------------------------------------------------------------------------


def test_segments_to_srt_basic():
    from captions import segments_to_srt
    segs = [
        {"start": 0.0, "end": 1.0, "text": "Hello"},
        {"start": 1.0, "end": 2.5, "text": "world"},
    ]
    out = segments_to_srt(segs)
    assert "1\n00:00:00,000 --> 00:00:01,000\nHello" in out
    assert "2\n00:00:01,000 --> 00:00:02,500\nworld" in out


def test_segments_to_srt_drops_empty_text():
    from captions import segments_to_srt
    segs = [
        {"start": 0.0, "end": 1.0, "text": "A"},
        {"start": 1.0, "end": 2.0, "text": "   "},  # dropped
        {"start": 2.0, "end": 3.0, "text": "B"},
    ]
    out = segments_to_srt(segs)
    # Only two numbered blocks.
    assert "1\n" in out
    assert "2\n" in out
    assert "3\n" not in out


def test_segments_to_srt_clamps_bad_end():
    from captions import segments_to_srt
    segs = [{"start": 5.0, "end": 4.0, "text": "flipped"}]
    out = segments_to_srt(segs)
    # end was clamped to start + 0.5s
    assert "00:00:05,000 --> 00:00:05,500" in out


# ---------------------------------------------------------------------------
# write_srt_file
# ---------------------------------------------------------------------------


def test_write_srt_file_adds_utf8_bom(tmp_path):
    from captions import write_srt_file
    dst = tmp_path / "x.srt"
    write_srt_file([{"start": 0, "end": 1, "text": "héllo"}], str(dst))
    data = dst.read_bytes()
    assert data[:3] == b"\xef\xbb\xbf"
    assert "héllo" in data.decode("utf-8-sig")


# ---------------------------------------------------------------------------
# Color + path escaping
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("inp, expected", [
    ("white",   "&H00FFFFFF"),
    ("black",   "&H00000000"),
    ("yellow",  "&H0000FFFF"),  # named
    ("#ff0000", "&H000000FF"),  # red: libass is BGR
    ("00ff00",  "&H0000FF00"),  # green
    ("not-a-color", "&H00FFFFFF"),  # fallback to white
])
def test_color_to_ass(inp, expected):
    from captions import _color_to_ass
    assert _color_to_ass(inp) == expected


def test_escape_srt_path_for_ffmpeg_windows():
    from captions import _escape_srt_path_for_ffmpeg
    p = _escape_srt_path_for_ffmpeg(r"C:\caps\talk.srt")
    assert ":" not in p.replace(r"\:", "")  # all `:` are escaped
    assert "\\" not in p.replace(r"\:", "").replace(r"\'", "")  # no backslashes left
    assert p == r"C\:/caps/talk.srt"


def test_escape_srt_path_for_ffmpeg_quote():
    from captions import _escape_srt_path_for_ffmpeg
    p = _escape_srt_path_for_ffmpeg("/tmp/O'Brien.srt")
    assert r"\'" in p


# ---------------------------------------------------------------------------
# Filter + argv assembly
# ---------------------------------------------------------------------------


def test_build_subtitles_filter_has_style_block(tmp_path):
    from captions import build_subtitles_filter, CaptionStyle
    f = build_subtitles_filter(str(tmp_path / "x.srt"),
                                CaptionStyle(font_size=30, bold=True,
                                             alignment=8, margin_v=48))
    assert f.startswith("subtitles='")
    assert "force_style='" in f
    assert "Fontsize=30" in f
    assert "Bold=-1" in f
    assert "Alignment=8" in f
    assert "MarginV=48" in f


def test_build_ffmpeg_cmd_argv_shape(tmp_path):
    from captions import build_ffmpeg_cmd
    src = str(tmp_path / "in.mp4")
    dst = str(tmp_path / "out.mp4")
    srt = str(tmp_path / "cap.srt")
    cmd = build_ffmpeg_cmd(src, dst, srt, ffmpeg_bin="ffmpeg")
    assert cmd[:2] == ["ffmpeg", "-y"]
    assert "-i" in cmd and cmd[cmd.index("-i") + 1] == src
    assert "-vf" in cmd
    vf = cmd[cmd.index("-vf") + 1]
    assert "subtitles='" in vf
    assert cmd[-1] == dst
    # Caption burn needs re-encode on the video but can copy audio.
    assert "libx264" in cmd
    assert cmd[cmd.index("-c:a") + 1] == "copy"


# ---------------------------------------------------------------------------
# burn_captions_into_video high-level
# ---------------------------------------------------------------------------


class _FakeProc:
    def __init__(self, returncode=0, stderr=""):
        self.returncode = returncode
        self.stderr = stderr


def _mk_src(tmp_path):
    p = tmp_path / "in.mp4"
    p.write_bytes(b"\x00" * 32)
    return str(p)


def _mk_srt(tmp_path, text="hi"):
    p = tmp_path / "cap.srt"
    p.write_text(f"1\n00:00:00,000 --> 00:00:01,000\n{text}\n", encoding="utf-8")
    return str(p)


def test_burn_missing_source(tmp_path):
    from captions import burn_captions_into_video
    r = burn_captions_into_video(
        str(tmp_path / "ghost.mp4"),
        str(tmp_path / "out.mp4"),
        srt_path=_mk_srt(tmp_path),
    )
    assert r["ok"] is False
    assert "not found" in r["error"].lower()


def test_burn_requires_srt_or_segments(tmp_path):
    from captions import burn_captions_into_video
    r = burn_captions_into_video(_mk_src(tmp_path), str(tmp_path / "out.mp4"))
    assert r["ok"] is False
    assert "srt_path or segments" in r["error"]


def test_burn_missing_srt_file(tmp_path):
    from captions import burn_captions_into_video
    r = burn_captions_into_video(
        _mk_src(tmp_path),
        str(tmp_path / "out.mp4"),
        srt_path=str(tmp_path / "ghost.srt"),
    )
    assert r["ok"] is False
    assert "srt not found" in r["error"].lower()


def test_burn_calls_ffmpeg_and_reports_success(tmp_path):
    from captions import burn_captions_into_video
    src = _mk_src(tmp_path)
    srt = _mk_srt(tmp_path)
    dst = str(tmp_path / "out.mp4")

    called = {}

    def fake_runner(cmd):
        called["cmd"] = cmd
        # Pretend ffmpeg wrote a real file.
        with open(dst, "wb") as f:
            f.write(b"\x00" * 4096)
        return _FakeProc(returncode=0)

    r = burn_captions_into_video(src, dst, srt_path=srt, runner=fake_runner)
    assert r["ok"], r
    assert r["output_path"] == dst
    assert r["srt_path"] == srt
    assert r["backend"] == "ffmpeg_subtitles"
    assert "cmd" in called
    assert called["cmd"][-1] == dst


def test_burn_nonzero_ffmpeg_exit_surfaces_tail(tmp_path):
    from captions import burn_captions_into_video
    src = _mk_src(tmp_path)
    srt = _mk_srt(tmp_path)

    def fake_runner(cmd):
        return _FakeProc(returncode=1, stderr="line a\nline b\n[fatal] codec not found")

    r = burn_captions_into_video(
        src, str(tmp_path / "out.mp4"), srt_path=srt, runner=fake_runner
    )
    assert r["ok"] is False
    assert "codec not found" in r["error"]


def test_burn_missing_ffmpeg_binary_reports_install_hint(tmp_path, monkeypatch):
    from captions import burn_captions_into_video
    src = _mk_src(tmp_path)
    srt = _mk_srt(tmp_path)

    def fake_runner(cmd):
        raise FileNotFoundError("ffmpeg: not found")

    r = burn_captions_into_video(
        src, str(tmp_path / "out.mp4"), srt_path=srt, runner=fake_runner
    )
    assert r["ok"] is False
    assert r["error"] == "ffmpeg not found"
    assert "install_hint" in r
    assert "PATH" in r["install_hint"]


def test_burn_writes_srt_from_segments(tmp_path):
    from captions import burn_captions_into_video
    src = _mk_src(tmp_path)
    dst = str(tmp_path / "out.mp4")

    def fake_runner(cmd):
        with open(dst, "wb") as f:
            f.write(b"\x00" * 4096)
        return _FakeProc(returncode=0)

    r = burn_captions_into_video(
        src, dst,
        segments=[{"start": 0, "end": 1, "text": "hello"}],
        runner=fake_runner,
    )
    assert r["ok"]
    assert os.path.isfile(r["srt_path"])
    # The sibling SRT should carry our text.
    assert "hello" in open(r["srt_path"], encoding="utf-8-sig").read()


# ---------------------------------------------------------------------------
# /api/v1/edit/caption endpoint
# ---------------------------------------------------------------------------


def _fresh_api(monkeypatch, tmp_path):
    """Reload api_server with auth disabled so we can post without a token."""
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


def test_caption_endpoint_rejects_missing_video(tmp_path, monkeypatch):
    _api, client = _fresh_api(monkeypatch, tmp_path)
    r = client.post(
        "/api/v1/edit/caption",
        json={"video_path": str(tmp_path / "ghost.mp4")},
    )
    assert r.status_code == 422
    assert "not found" in r.json()["detail"].lower()


def test_caption_endpoint_rejects_missing_srt(tmp_path, monkeypatch):
    _api, client = _fresh_api(monkeypatch, tmp_path)
    src = tmp_path / "in.mp4"
    src.write_bytes(b"\x00" * 32)
    r = client.post(
        "/api/v1/edit/caption",
        json={"video_path": str(src), "srt_path": str(tmp_path / "ghost.srt")},
    )
    assert r.status_code == 422
    assert "srt not found" in r.json()["detail"].lower()


def test_caption_endpoint_dispatches_runner(tmp_path, monkeypatch):
    api, client = _fresh_api(monkeypatch, tmp_path)
    src = tmp_path / "in.mp4"
    src.write_bytes(b"\x00" * 32)

    seen = {}

    def fake_run(job_id, req):
        seen["job_id"] = job_id
        seen["video_path"] = req.video_path
        seen["font_size"] = req.font_size

    monkeypatch.setattr(api, "_run_caption", fake_run, raising=True)

    # Threads are fire-and-forget; swap for inline exec so the assertions
    # don't race the background start.
    class _Immediate:
        def __init__(self, *, target, args, daemon=False):
            self.target = target
            self.args = args
        def start(self):
            self.target(*self.args)

    monkeypatch.setattr(api.threading, "Thread", _Immediate, raising=True)

    r = client.post(
        "/api/v1/edit/caption",
        json={"video_path": str(src), "font_size": 30, "position": "top"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["job_id"] == seen["job_id"]
    assert seen["video_path"] == str(src)
    assert seen["font_size"] == 30


def test_caption_request_defaults():
    from api_server import CaptionRequest
    req = CaptionRequest(video_path="x.mp4")
    assert req.font_name == "Arial"
    assert req.font_size == 24
    assert req.position == "bottom"
    assert req.srt_path is None
    assert req.language is None
    assert req.bold is False
