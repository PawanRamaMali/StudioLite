"""Tests for karaoke_captions.py (animated word-highlight ASS output)."""
from __future__ import annotations

import os
import pytest


# ---------------------------------------------------------------------------
# Time formatting
# ---------------------------------------------------------------------------


def test_fmt_ass_time_basic():
    from karaoke_captions import _fmt_ass_time
    assert _fmt_ass_time(0.0) == "0:00:00.00"
    assert _fmt_ass_time(1.23) == "0:00:01.23"
    assert _fmt_ass_time(61.5) == "0:01:01.50"
    assert _fmt_ass_time(3600) == "1:00:00.00"


def test_fmt_ass_time_rounds_centiseconds():
    from karaoke_captions import _fmt_ass_time
    assert _fmt_ass_time(1.996) == "0:00:02.00"  # rounds up cleanly


def test_fmt_ass_time_clamps_negative():
    from karaoke_captions import _fmt_ass_time
    assert _fmt_ass_time(-5.0) == "0:00:00.00"


# ---------------------------------------------------------------------------
# Color conversion
# ---------------------------------------------------------------------------


def test_rrggbb_to_ass_white():
    from karaoke_captions import _rrggbb_to_ass
    assert _rrggbb_to_ass("FFFFFF") == "&H00FFFFFF"


def test_rrggbb_to_ass_order_swap():
    from karaoke_captions import _rrggbb_to_ass
    # ASS = &H00BBGGRR. Red input -> blue bytes in position
    assert _rrggbb_to_ass("FF0000") == "&H000000FF"
    assert _rrggbb_to_ass("00FF00") == "&H0000FF00"
    assert _rrggbb_to_ass("0000FF") == "&H00FF0000"


def test_rrggbb_to_ass_hash_prefix_ok():
    from karaoke_captions import _rrggbb_to_ass
    assert _rrggbb_to_ass("#FFD54A") == "&H004AD5FF"


def test_rrggbb_to_ass_invalid_falls_back_to_white():
    from karaoke_captions import _rrggbb_to_ass
    assert _rrggbb_to_ass("zzz") == "&H00FFFFFF"


# ---------------------------------------------------------------------------
# Word normalization
# ---------------------------------------------------------------------------


def test_normalize_words_from_bare_strings():
    from karaoke_captions import _normalize_words
    out = _normalize_words(["hi", "world"])
    assert len(out) == 2
    assert out[0]["word"] == "hi"
    assert out[1]["start"] > out[0]["start"]


def test_normalize_words_with_timings():
    from karaoke_captions import _normalize_words
    out = _normalize_words([
        {"word": "hello", "start": 1.0, "end": 1.4},
        {"word": "there", "start": 1.4, "end": 1.8},
    ])
    assert out[0]["start"] == 1.0 and out[0]["end"] == 1.4
    assert out[1]["start"] == 1.4


def test_normalize_words_drops_empty():
    from karaoke_captions import _normalize_words
    out = _normalize_words([{"word": "", "start": 0, "end": 1}, {"word": "ok", "start": 1, "end": 2}])
    assert len(out) == 1
    assert out[0]["word"] == "ok"


def test_normalize_words_clamps_inverted_timing():
    from karaoke_captions import _normalize_words
    out = _normalize_words([{"word": "x", "start": 5.0, "end": 3.0}])
    assert out[0]["end"] > out[0]["start"]


# ---------------------------------------------------------------------------
# Line grouping
# ---------------------------------------------------------------------------


def test_group_respects_max_per_line():
    from karaoke_captions import _group_into_lines
    words = [{"word": f"w{i}", "start": i * 0.1, "end": i * 0.1 + 0.1} for i in range(12)]
    lines = _group_into_lines(words, max_per_line=4)
    assert all(len(l) <= 4 for l in lines)
    assert len(lines) == 3


def test_group_breaks_on_silence_gap():
    from karaoke_captions import _group_into_lines
    words = [
        {"word": "a", "start": 0.0, "end": 0.3},
        {"word": "b", "start": 0.3, "end": 0.6},
        # 1-second gap - should break the line here
        {"word": "c", "start": 1.6, "end": 1.9},
    ]
    lines = _group_into_lines(words, max_per_line=10)
    assert len(lines) == 2
    assert [w["word"] for w in lines[0]] == ["a", "b"]
    assert [w["word"] for w in lines[1]] == ["c"]


# ---------------------------------------------------------------------------
# Text escaping
# ---------------------------------------------------------------------------


def test_escape_ass_text_handles_braces():
    from karaoke_captions import _escape_ass_text
    assert _escape_ass_text("{hi}") == r"\{hi\}"


def test_escape_ass_text_handles_backslash():
    from karaoke_captions import _escape_ass_text
    assert _escape_ass_text("\\path") == "\\\\path"


# ---------------------------------------------------------------------------
# Full ASS build
# ---------------------------------------------------------------------------


def test_build_ass_file_has_sections():
    from karaoke_captions import build_ass_file
    out = build_ass_file([{"word": "hello", "start": 0.0, "end": 0.5},
                          {"word": "world", "start": 0.5, "end": 1.0}])
    assert "[Script Info]" in out
    assert "[V4+ Styles]" in out
    assert "[Events]" in out
    assert "Dialogue:" in out


def test_build_ass_file_uses_karaoke_tag():
    from karaoke_captions import build_ass_file
    out = build_ass_file([{"word": "hi", "start": 0, "end": 0.4}])
    # \kf is the karaoke fill tag (~40cs for 0.4s)
    assert r"\kf" in out


def test_build_ass_file_includes_timing():
    from karaoke_captions import build_ass_file
    out = build_ass_file([{"word": "hi", "start": 1.5, "end": 2.0}])
    assert "0:00:01.50" in out
    assert "0:00:02.00" in out


def test_build_ass_file_empty_words_produces_empty_events():
    from karaoke_captions import build_ass_file
    out = build_ass_file([])
    assert "Dialogue:" not in out
    assert "[Events]" in out  # section exists, just no events


def test_build_ass_file_writes_file(tmp_path):
    from karaoke_captions import write_ass_file
    out = tmp_path / "out.ass"
    path = write_ass_file([{"word": "hi", "start": 0, "end": 0.5}], str(out))
    assert os.path.isfile(path)
    content = out.read_text(encoding="utf-8")
    assert "[Events]" in content


# ---------------------------------------------------------------------------
# words_from_segments
# ---------------------------------------------------------------------------


def test_words_from_segments_prefers_word_level():
    from karaoke_captions import words_from_segments
    segs = [{
        "text": "hi there",
        "start": 0.0, "end": 1.0,
        "words": [
            {"word": "hi", "start": 0.0, "end": 0.4},
            {"word": "there", "start": 0.5, "end": 1.0},
        ],
    }]
    out = words_from_segments(segs)
    assert len(out) == 2
    assert out[0]["start"] == 0.0 and out[1]["start"] == 0.5


def test_words_from_segments_falls_back_to_even_distribution():
    from karaoke_captions import words_from_segments
    segs = [{"text": "one two three four", "start": 0.0, "end": 2.0}]
    out = words_from_segments(segs)
    assert len(out) == 4
    assert out[0]["start"] == 0.0
    # Each word gets 0.5s
    assert abs(out[1]["start"] - 0.5) < 0.01
    assert abs(out[-1]["end"] - 2.0) < 0.01


def test_words_from_segments_skips_empty_text():
    from karaoke_captions import words_from_segments
    out = words_from_segments([{"text": "   ", "start": 0, "end": 1}])
    assert out == []
