"""Tests for social_export.py (pure argv builders + run_export orchestration)."""
from __future__ import annotations

import os
import pytest


class _FakeProc:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _mk_mp4(tmp_path, name="in.mp4"):
    p = tmp_path / name
    # ftyp box magic bytes at offset 4 - not validated here but keeps it realistic
    p.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 1024)
    return str(p)


# ---------------------------------------------------------------------------
# fit filter
# ---------------------------------------------------------------------------


def test_fit_stretch_scales_to_exact_dims():
    from social_export import _fit_filter
    assert _fit_filter("stretch", 1080, 1920) == "scale=1080:1920:flags=lanczos"


def test_fit_crop_increases_then_crops():
    from social_export import _fit_filter
    f = _fit_filter("crop", 1080, 1920)
    assert "force_original_aspect_ratio=increase" in f
    assert "crop=1080:1920" in f


def test_fit_pad_black_decreases_then_pads():
    from social_export import _fit_filter
    f = _fit_filter("pad_black", 1080, 1920)
    assert "force_original_aspect_ratio=decrease" in f
    assert "pad=1080:1920" in f
    assert "color=black" in f


def test_fit_pad_blur_splits_and_overlays():
    from social_export import _fit_filter
    f = _fit_filter("pad_blur", 1080, 1920)
    assert "split" in f and "boxblur" in f and "overlay" in f


def test_fit_pad_blur_unknown_mode_defaults_to_blur():
    from social_export import _fit_filter
    f = _fit_filter("bogus-mode", 1080, 1920)
    assert "boxblur" in f


# ---------------------------------------------------------------------------
# subtitle filter
# ---------------------------------------------------------------------------


def test_subtitle_filter_empty_path_returns_empty_string():
    from social_export import _subtitle_filter, SocialExportStyle
    from video_formats import resolve_format
    assert _subtitle_filter("", resolve_format("reels"), SocialExportStyle()) == ""


def test_subtitle_filter_uses_safe_zone_margin(tmp_path):
    from social_export import _subtitle_filter, SocialExportStyle
    from video_formats import resolve_format
    srt = tmp_path / "c.srt"
    srt.write_text("1\n00:00:01,000 --> 00:00:03,000\nhello\n")
    out = _subtitle_filter(str(srt), resolve_format("reels"), SocialExportStyle())
    assert "subtitles=" in out
    assert "Alignment=2" in out        # bottom-center
    assert "MarginV=" in out


def test_subtitle_filter_font_size_scales_with_height():
    from social_export import _subtitle_filter, SocialExportStyle
    from video_formats import resolve_format
    small = _subtitle_filter("/t.srt", resolve_format("reels"),
                             SocialExportStyle(caption_font_size_pct=0.03))
    large = _subtitle_filter("/t.srt", resolve_format("reels"),
                             SocialExportStyle(caption_font_size_pct=0.08))
    def _sz(s):
        m = s.split("FontSize=")[1].split(",")[0]
        return int(m)
    assert _sz(large) > _sz(small)


def test_subtitle_filter_escapes_windows_drive_colon():
    from social_export import _subtitle_filter, SocialExportStyle
    from video_formats import resolve_format
    out = _subtitle_filter(r"C:\captions\clip.srt", resolve_format("reels"), SocialExportStyle())
    # ffmpeg subtitles filter needs the colon in "C:" escaped so it isn't
    # parsed as a key:value pair.
    assert r"C\:" in out


# ---------------------------------------------------------------------------
# audio filter
# ---------------------------------------------------------------------------


def test_audio_filter_normalize_targets_platform_lufs():
    from social_export import build_audio_filter, SocialExportStyle
    from video_formats import resolve_format
    af = build_audio_filter(resolve_format("reels"), SocialExportStyle(normalize_loudness=True))
    assert "loudnorm=I=-14.0" in af


def test_audio_filter_disabled_returns_anull():
    from social_export import build_audio_filter, SocialExportStyle
    from video_formats import resolve_format
    af = build_audio_filter(resolve_format("reels"), SocialExportStyle(normalize_loudness=False))
    assert af == "anull"


# ---------------------------------------------------------------------------
# full argv
# ---------------------------------------------------------------------------


def test_build_export_argv_reels_portrait(tmp_path):
    from social_export import build_export_argv, SocialExportStyle
    inp = _mk_mp4(tmp_path)
    out = str(tmp_path / "out.mp4")
    argv = build_export_argv(inp, out, SocialExportStyle(format_id="reels", fit_mode="crop"))
    assert argv[0] == "ffmpeg"
    assert "-i" in argv and inp in argv
    assert "-vf" in argv
    assert "libx264" in argv
    assert "aac" in argv
    assert argv[-1] == out


def test_build_export_argv_pad_blur_uses_filter_complex(tmp_path):
    from social_export import build_export_argv, SocialExportStyle
    inp = _mk_mp4(tmp_path)
    out = str(tmp_path / "out.mp4")
    argv = build_export_argv(inp, out, SocialExportStyle(format_id="reels", fit_mode="pad_blur"))
    assert "-filter_complex" in argv
    assert "-vf" not in argv


def test_build_export_argv_respects_duration_cap(tmp_path):
    from social_export import build_export_argv, SocialExportStyle
    inp = _mk_mp4(tmp_path)
    out = str(tmp_path / "out.mp4")
    argv = build_export_argv(inp, out, SocialExportStyle(format_id="shorts", max_duration_s=42.5))
    ti = argv.index("-t")
    assert float(argv[ti + 1]) == pytest.approx(42.5)


def test_build_export_argv_default_duration_is_platform_max(tmp_path):
    from social_export import build_export_argv, SocialExportStyle
    inp = _mk_mp4(tmp_path)
    out = str(tmp_path / "out.mp4")
    argv = build_export_argv(inp, out, SocialExportStyle(format_id="shorts"))
    ti = argv.index("-t")
    assert float(argv[ti + 1]) == 60.0  # shorts caps at 60s


def test_build_export_argv_fps_matches_preset(tmp_path):
    from social_export import build_export_argv, SocialExportStyle
    inp = _mk_mp4(tmp_path)
    out = str(tmp_path / "out.mp4")
    argv = build_export_argv(inp, out, SocialExportStyle(format_id="reels"))
    ri = argv.index("-r")
    assert argv[ri + 1] == "30"


def test_build_export_argv_bitrate_derives_from_platform_cap(tmp_path):
    from social_export import build_export_argv, SocialExportStyle
    inp = _mk_mp4(tmp_path)
    out = str(tmp_path / "out.mp4")
    # TikTok caps at 10 Mbps -> target around 7500k
    argv = build_export_argv(inp, out, SocialExportStyle(format_id="tiktok"))
    bi = argv.index("-b:v")
    assert argv[bi + 1] == "7500k"


def test_build_export_argv_with_subtitle_adds_subtitles_filter(tmp_path):
    from social_export import build_export_argv, SocialExportStyle
    inp = _mk_mp4(tmp_path)
    srt = tmp_path / "c.srt"
    srt.write_text("1\n00:00:01,000 --> 00:00:03,000\nhello\n")
    out = str(tmp_path / "out.mp4")
    argv = build_export_argv(inp, out, SocialExportStyle(
        format_id="reels", fit_mode="crop", subtitle_path=str(srt)))
    vfi = argv.index("-vf")
    assert "subtitles=" in argv[vfi + 1]


# ---------------------------------------------------------------------------
# probe_duration
# ---------------------------------------------------------------------------


def test_probe_duration_parses_json(tmp_path):
    from social_export import probe_duration
    inp = _mk_mp4(tmp_path)
    r = probe_duration(inp, runner=lambda a: _FakeProc(
        0, '{"format": {"duration": "12.345"}}', ""
    ))
    assert r == pytest.approx(12.345)


def test_probe_duration_missing_ffprobe_returns_none(tmp_path):
    from social_export import probe_duration
    def _raise(_a):
        raise FileNotFoundError("ffprobe")
    assert probe_duration(_mk_mp4(tmp_path), runner=_raise) is None


def test_probe_duration_non_zero_exit_returns_none(tmp_path):
    from social_export import probe_duration
    assert probe_duration(_mk_mp4(tmp_path),
                          runner=lambda _a: _FakeProc(1, "", "boom")) is None


def test_probe_duration_junk_stdout_returns_none(tmp_path):
    from social_export import probe_duration
    assert probe_duration(_mk_mp4(tmp_path),
                          runner=lambda _a: _FakeProc(0, "not json", "")) is None


# ---------------------------------------------------------------------------
# run_export orchestration
# ---------------------------------------------------------------------------


def test_run_export_missing_input_raises(tmp_path):
    from social_export import run_export, SocialExportStyle
    with pytest.raises(FileNotFoundError):
        run_export(str(tmp_path / "nope.mp4"), str(tmp_path / "out.mp4"),
                   SocialExportStyle())


def test_run_export_success_path_records_argv(tmp_path):
    from social_export import run_export, SocialExportStyle
    inp = _mk_mp4(tmp_path)
    out = str(tmp_path / "out.mp4")
    seen = {}
    def _fake(argv):
        seen["argv"] = argv
        return _FakeProc(0, "", "")
    r = run_export(inp, out, SocialExportStyle(format_id="reels", max_duration_s=30.0),
                   runner=_fake)
    assert r["ok"] is True
    assert r["output_path"] == out
    assert r["format"]["id"] == "reels"
    assert seen["argv"][0] == "ffmpeg"


def test_run_export_failure_surfaces_stderr(tmp_path):
    from social_export import run_export, SocialExportStyle
    inp = _mk_mp4(tmp_path)
    out = str(tmp_path / "out.mp4")
    r = run_export(inp, out, SocialExportStyle(format_id="reels", max_duration_s=5.0),
                   runner=lambda _a: _FakeProc(1, "", "ffmpeg crashed"))
    assert r["ok"] is False
    assert r["output_path"] == ""
    assert "ffmpeg crashed" in r["stderr"]
