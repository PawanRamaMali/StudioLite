"""Caption / subtitle utilities.

Pure-Python helpers that convert transcription segments to SRT, build
an ffmpeg `subtitles=…` filter with a configurable force_style, and
burn captions into a video via ffmpeg. Keeping the ffmpeg argv
assembly here instead of inside api_server lets us unit-test the
exact command without actually invoking ffmpeg.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


# ---------------------------------------------------------------------------
# Timestamp formatting
# ---------------------------------------------------------------------------


def format_timestamp_srt(seconds: float) -> str:
    """`00:00:00,000` form that SRT and ffmpeg both parse. Clamps
    negatives to zero so a bad segment doesn't produce `-00:00:01,000`."""
    s = max(0.0, float(seconds))
    hours = int(s // 3600)
    minutes = int((s % 3600) // 60)
    secs = int(s % 60)
    ms = int(round((s - int(s)) * 1000))
    if ms == 1000:  # round-up carry
        secs += 1
        ms = 0
        if secs == 60:
            secs = 0
            minutes += 1
            if minutes == 60:
                minutes = 0
                hours += 1
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


# ---------------------------------------------------------------------------
# SRT assembly
# ---------------------------------------------------------------------------


def segments_to_srt(segments: List[Dict[str, Any]]) -> str:
    """Render whisper-style segments ({start, end, text}) as SRT text.

    Empty-text segments are dropped (they add churn and some players
    render them as a blank card). end<start is clamped to start+0.5s
    so a mis-aligned segment still shows for a readable moment.
    """
    out: List[str] = []
    index = 0
    for seg in segments:
        text = str(seg.get("text", "")).strip()
        if not text:
            continue
        start = float(seg.get("start", 0.0) or 0.0)
        end = float(seg.get("end", start + 1.0) or start + 1.0)
        if end <= start:
            end = start + 0.5
        index += 1
        out.append(str(index))
        out.append(f"{format_timestamp_srt(start)} --> {format_timestamp_srt(end)}")
        out.append(text)
        out.append("")
    return "\n".join(out)


def write_srt_file(segments: List[Dict[str, Any]], out_path: str) -> str:
    """Write segments as SRT, UTF-8 with BOM so players on Windows
    render non-ASCII correctly without having to guess the encoding."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    content = segments_to_srt(segments)
    with open(out_path, "wb") as f:
        # UTF-8 BOM — several Windows subtitle renderers assume cp1252
        # without it.
        f.write(b"\xef\xbb\xbf")
        f.write(content.encode("utf-8"))
    return out_path


# ---------------------------------------------------------------------------
# ffmpeg filter construction
# ---------------------------------------------------------------------------


@dataclass
class CaptionStyle:
    """ASS/SSA style knobs threaded into ffmpeg's subtitles filter.
    All fields are optional - defaults match a readable sans-serif
    with a thin outline that stays legible on any background."""
    font_name: str = "Arial"
    font_size: int = 24
    primary_color: str = "white"          # outline is used too
    outline_color: str = "black"
    outline_width: int = 2
    shadow: int = 0
    bold: bool = False
    italic: bool = False
    # Alignment uses libass numpad codes: 2 = bottom-centre (default),
    # 8 = top-centre, 5 = middle-centre. Anything else is passed through.
    alignment: int = 2
    margin_v: int = 32  # vertical margin from the alignment edge


_NAMED_COLORS = {
    "white":   "FFFFFF",
    "black":   "000000",
    "red":     "0000FF",   # libass is BGR, hence the swap below
    "green":   "00FF00",
    "blue":    "FF0000",
    "yellow":  "00FFFF",
    "cyan":    "FFFF00",
    "magenta": "FF00FF",
}


def _color_to_ass(color: str) -> str:
    """libass color literal: `&HAABBGGRR`. Accepts a #RRGGBB hex, a
    bare 6-hex-digit RGB, or one of the named colors above."""
    c = color.strip()
    if c.startswith("#"):
        c = c[1:]
    if c.lower() in _NAMED_COLORS:
        bgr = _NAMED_COLORS[c.lower()]
    elif len(c) == 6 and all(ch in "0123456789abcdefABCDEF" for ch in c):
        rr, gg, bb = c[0:2], c[2:4], c[4:6]
        bgr = (bb + gg + rr).upper()
    else:
        bgr = "FFFFFF"  # fall back to white rather than fail the whole burn
    return f"&H00{bgr}"


def _format_force_style(style: CaptionStyle) -> str:
    """Comma-separated libass style assignments that go into
    subtitles=…:force_style='…'."""
    parts = [
        f"Fontname={style.font_name}",
        f"Fontsize={style.font_size}",
        f"PrimaryColour={_color_to_ass(style.primary_color)}",
        f"OutlineColour={_color_to_ass(style.outline_color)}",
        f"Outline={style.outline_width}",
        f"Shadow={style.shadow}",
        f"Bold={-1 if style.bold else 0}",
        f"Italic={-1 if style.italic else 0}",
        f"Alignment={style.alignment}",
        f"MarginV={style.margin_v}",
    ]
    return ",".join(parts)


def _escape_srt_path_for_ffmpeg(path: str) -> str:
    """ffmpeg's subtitles filter parses the path through libavfilter's
    option tokenizer, which treats `:` as a key/value separator and
    `\\` as an escape. On Windows the drive letter's colon must become
    `\\:` and backslashes must become forward slashes."""
    p = path.replace("\\", "/")
    # Escape the `:` after the drive letter (and any other `:` that
    # might appear in a path): libavfilter needs `\:`.
    p = p.replace(":", r"\:")
    # Single quotes inside the filter string need to be escaped too.
    p = p.replace("'", r"\'")
    return p


def build_subtitles_filter(srt_path: str,
                            style: Optional[CaptionStyle] = None) -> str:
    """`subtitles=<escaped_path>:force_style='…'` — the ffmpeg -vf arg
    that burns the SRT at `srt_path` with the given style. Returned
    as a string instead of a shell command so callers can keep argv
    unquoted and pass it to subprocess directly."""
    style = style or CaptionStyle()
    esc_path = _escape_srt_path_for_ffmpeg(srt_path)
    force = _format_force_style(style)
    return f"subtitles='{esc_path}':force_style='{force}'"


def build_ffmpeg_cmd(src: str, dst: str, srt_path: str,
                      style: Optional[CaptionStyle] = None,
                      ffmpeg_bin: str = "ffmpeg") -> List[str]:
    """Argv for a one-pass burn-in with libx264 + aac copy."""
    vf = build_subtitles_filter(srt_path, style)
    return [
        ffmpeg_bin, "-y",
        "-i", src,
        "-vf", vf,
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-c:a", "copy",
        dst,
    ]


# ---------------------------------------------------------------------------
# High-level: burn in from an SRT file (or from in-memory segments).
# ---------------------------------------------------------------------------


def burn_captions_into_video(
    src_abs: str,
    dst_abs: str,
    *,
    srt_path: Optional[str] = None,
    segments: Optional[List[Dict[str, Any]]] = None,
    style: Optional[CaptionStyle] = None,
    ffmpeg_bin: str = "ffmpeg",
    runner: Callable[[List[str]], subprocess.CompletedProcess] = None,
) -> Dict[str, Any]:
    """Burn captions into src_abs → dst_abs. Pass either `srt_path`
    (existing SRT on disk) or `segments` (and we write a sibling SRT).

    Returns {ok, backend, output_path, srt_path} on success or
    {ok: False, error} on any failure. Never raises.
    """
    if not os.path.isfile(src_abs):
        return {"ok": False, "error": f"source not found: {src_abs}"}
    if srt_path is None and segments is None:
        return {"ok": False, "error": "either srt_path or segments is required"}

    chosen_srt = srt_path
    if chosen_srt is None:
        base, _ = os.path.splitext(dst_abs)
        chosen_srt = base + ".srt"
        try:
            write_srt_file(segments or [], chosen_srt)
        except OSError as e:
            return {"ok": False, "error": f"srt write failed: {e}"}
    elif not os.path.isfile(chosen_srt):
        return {"ok": False, "error": f"srt not found: {chosen_srt}"}

    ff = shutil.which(ffmpeg_bin) or ffmpeg_bin
    cmd = build_ffmpeg_cmd(src_abs, dst_abs, chosen_srt, style, ffmpeg_bin=ff)

    try:
        _runner = runner or (lambda c: subprocess.run(
            c, capture_output=True, text=True, check=False,
        ))
        proc = _runner(cmd)
    except FileNotFoundError:
        return {"ok": False,
                "error": "ffmpeg not found",
                "install_hint": "Install ffmpeg and ensure it is on PATH."}
    except OSError as e:
        return {"ok": False, "error": f"ffmpeg launch failed: {e}"}

    if proc.returncode != 0:
        stderr = (getattr(proc, "stderr", "") or "").strip()
        tail = stderr.splitlines()[-1] if stderr else f"exit {proc.returncode}"
        return {"ok": False, "error": f"ffmpeg failed: {tail}"}

    if not (os.path.exists(dst_abs) and os.path.getsize(dst_abs) > 1024):
        return {"ok": False, "error": "caption burn produced no output"}

    return {
        "ok": True,
        "backend": "ffmpeg_subtitles",
        "output_path": dst_abs,
        "srt_path": chosen_srt,
    }
