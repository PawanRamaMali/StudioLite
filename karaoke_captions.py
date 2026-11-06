"""Animated word-highlight (karaoke) captions.

Takes whisper word-level timings and emits an ASS subtitle file with the
active word highlighted in real time. Works with ffmpeg's ``subtitles=``
filter via libass - no extra dependencies.

Why ASS instead of styled SRT:
    SRT has no in-line timing. Styled SRT with `<font color>` tags at best
    turns a line one color at a time, not word-by-word. ASS's ``{\\k<cs>}``
    karaoke tag highlights for a specific number of centiseconds, which is
    exactly what the social-shorts caption look needs.

Word records expected shape:
    {"word": "hello", "start": 0.12, "end": 0.48}
Missing timings default to a uniform 0.25s per word, so the caller can
feed raw transcription output without having to post-process.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass
class KaraokeStyle:
    font_name: str = "Arial"
    font_size: int = 72
    primary_color: str = "FFFFFF"       # inactive word color (hex RRGGBB)
    highlight_color: str = "FFD54A"     # active word color (hex RRGGBB)
    outline_color: str = "000000"
    outline_width: int = 3
    shadow: int = 0
    bold: bool = True
    italic: bool = False
    alignment: int = 2                  # bottom-centre
    margin_v: int = 120
    video_width: int = 1080
    video_height: int = 1920
    max_words_per_line: int = 4         # keep lines short for mobile viewers


def _rrggbb_to_ass(hex_rgb: str) -> str:
    """ASS stores colors as &H00BBGGRR - alpha byte first, then BGR."""
    c = hex_rgb.lstrip("#").strip()
    if len(c) != 6:
        c = "FFFFFF"
    rr, gg, bb = c[0:2], c[2:4], c[4:6]
    return f"&H00{bb.upper()}{gg.upper()}{rr.upper()}"


def _fmt_ass_time(seconds: float) -> str:
    """ASS wants `H:MM:SS.cs` (centiseconds, not ms)."""
    s = max(0.0, float(seconds))
    hours = int(s // 3600)
    minutes = int((s % 3600) // 60)
    secs = s % 60
    cs = int(round((secs - int(secs)) * 100))
    i_secs = int(secs)
    if cs == 100:
        cs = 0
        i_secs += 1
        if i_secs == 60:
            i_secs = 0
            minutes += 1
            if minutes == 60:
                minutes = 0
                hours += 1
    return f"{hours}:{minutes:02d}:{i_secs:02d}.{cs:02d}"


def _normalize_words(words: Iterable[Any]) -> list[dict]:
    """Coerce the input into [{word, start, end}] with sane defaults."""
    out = []
    default_dur = 0.25
    cursor = 0.0
    for raw in words:
        if isinstance(raw, str):
            out.append({"word": raw, "start": cursor, "end": cursor + default_dur})
            cursor += default_dur
            continue
        text = str(raw.get("word", raw.get("text", ""))).strip()
        if not text:
            continue
        start = raw.get("start", raw.get("t0"))
        end = raw.get("end", raw.get("t1"))
        try:
            start_f = float(start) if start is not None else cursor
        except (TypeError, ValueError):
            start_f = cursor
        try:
            end_f = float(end) if end is not None else start_f + default_dur
        except (TypeError, ValueError):
            end_f = start_f + default_dur
        if end_f <= start_f:
            end_f = start_f + default_dur
        out.append({"word": text, "start": start_f, "end": end_f})
        cursor = end_f
    return out


def _group_into_lines(words: list[dict], max_per_line: int) -> list[list[dict]]:
    """Break the word stream into short caption lines.

    Breaks on big gaps (>0.6s silence suggests a sentence boundary) before
    hitting the per-line cap, so lines align with speech instead of being
    arbitrarily chunked mid-phrase.
    """
    lines: list[list[dict]] = []
    cur: list[dict] = []
    for i, w in enumerate(words):
        if not cur:
            cur.append(w)
            continue
        prev_end = cur[-1]["end"]
        gap = w["start"] - prev_end
        if gap > 0.6 or len(cur) >= max_per_line:
            lines.append(cur)
            cur = [w]
        else:
            cur.append(w)
    if cur:
        lines.append(cur)
    return lines


def _escape_ass_text(s: str) -> str:
    """ASS treats `{` as an override-block opener and `\\` as an escape."""
    return (
        s.replace("\\", "\\\\")
         .replace("{", "\\{")
         .replace("}", "\\}")
    )


def _build_line_text(line: list[dict], style: KaraokeStyle) -> tuple[str, float, float]:
    """Return (ass_text, line_start_s, line_end_s) for one caption line."""
    start = line[0]["start"]
    end = line[-1]["end"]
    primary = _rrggbb_to_ass(style.primary_color)
    highlight = _rrggbb_to_ass(style.highlight_color)
    parts: list[str] = []
    # Start the line in the inactive (primary) color
    parts.append(f"{{\\c{primary}}}")
    for w in line:
        word_dur_cs = max(1, int(round((w["end"] - w["start"]) * 100)))
        safe_word = _escape_ass_text(w["word"])
        # \kf<cs> is "karaoke fill" - word fills with the secondary color
        # for the duration, giving the active-word animation.
        parts.append(f"{{\\c{primary}\\kf{word_dur_cs}\\c{highlight}}}{safe_word} ")
    return "".join(parts).rstrip(), start, end


def build_ass_file(words: list[Any], style: KaraokeStyle = None) -> str:
    """Return a complete ASS subtitle document for the word list."""
    style = style or KaraokeStyle()
    normalized = _normalize_words(words)
    lines = _group_into_lines(normalized, style.max_words_per_line)
    primary = _rrggbb_to_ass(style.primary_color)
    secondary = _rrggbb_to_ass(style.highlight_color)
    outline = _rrggbb_to_ass(style.outline_color)

    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {style.video_width}\n"
        f"PlayResY: {style.video_height}\n"
        "WrapStyle: 2\n"
        "ScaledBorderAndShadow: yes\n"
        "YCbCr Matrix: TV.709\n"
        "\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Default,{style.font_name},{style.font_size},"
        f"{primary},{secondary},{outline},&H00000000,"
        f"{-1 if style.bold else 0},{-1 if style.italic else 0},0,0,"
        f"100,100,0,0,1,{style.outline_width},{style.shadow},"
        f"{style.alignment},40,40,{style.margin_v},0\n"
        "\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )

    events: list[str] = []
    for line in lines:
        text, s, e = _build_line_text(line, style)
        events.append(
            f"Dialogue: 0,{_fmt_ass_time(s)},{_fmt_ass_time(e)},Default,,0,0,0,,{text}"
        )
    return header + "\n".join(events) + "\n"


def write_ass_file(words: list[Any], out_path: str, style: KaraokeStyle = None) -> str:
    """Write the ASS document to ``out_path``. Creates parent dirs."""
    import os
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    content = build_ass_file(words, style)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)
    return out_path


def words_from_segments(segments: Iterable[dict]) -> list[dict]:
    """Flatten whisper segments ({words: [...]} or {text, start, end}) into a word list.

    Segments with per-word timing are preferred; those without fall back to
    even distribution of the segment's duration across its words.
    """
    out: list[dict] = []
    for seg in segments:
        words = seg.get("words") or []
        if words:
            for w in words:
                out.append({
                    "word": str(w.get("word", w.get("text", ""))).strip(),
                    "start": float(w.get("start", 0.0) or 0.0),
                    "end": float(w.get("end", 0.0) or 0.0),
                })
            continue
        text = str(seg.get("text", "")).strip()
        if not text:
            continue
        toks = text.split()
        if not toks:
            continue
        start = float(seg.get("start", 0.0) or 0.0)
        end = float(seg.get("end", start + 1.0) or start + 1.0)
        step = (end - start) / len(toks)
        for i, tok in enumerate(toks):
            out.append({
                "word": tok,
                "start": start + i * step,
                "end": start + (i + 1) * step,
            })
    return [w for w in out if w["word"]]
