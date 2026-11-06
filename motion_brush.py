"""Motion brush: arrow strokes → directional motion prompts.

Pragmatic, not perfect. True motion brush needs DragNUWA / Motion-I2V's
trajectory conditioning, which isn't integrated here yet. In the meantime
this module turns the user's arrow strokes on the preview into explicit
directional language tacked onto the I2V prompt, so the generator at least
knows which way things should move.

Workflow:
1. Frontend canvas captures arrow strokes: each stroke is a start (x, y) and
   end (x, y) in normalized 0.0-1.0 image space, plus an optional label.
2. This module translates each stroke to a cardinal/diagonal direction, a
   distance bucket (subtle / moderate / strong), and a region label
   (top-left / center / bottom-right / ...).
3. ``format_prompt_suffix`` renders the lot as a single sentence appended to
   the base prompt.

When the user later drops in a proper trajectory-conditioned backend, swap
``format_prompt_suffix`` for a different ``render_fn`` and the panel keeps
working.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class MotionStroke:
    """One arrow stroke drawn on the preview canvas.

    Coordinates are normalized to [0, 1] over image width and height.
    (0, 0) is top-left, (1, 1) is bottom-right.
    """
    x0: float
    y0: float
    x1: float
    y1: float
    label: str = ""   # optional subject label, e.g. "car" or "character"


def _clamp01(v: float) -> float:
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return float(v)


def _cardinal(dx: float, dy: float) -> str:
    """Nearest of the 8 cardinal/diagonal directions.

    Note: image y grows downward, so positive dy = down, negative dy = up.
    """
    if dx == 0 and dy == 0:
        return "still"
    angle_deg = math.degrees(math.atan2(dy, dx))  # -180..180
    # Rotate so right = 0, then bucket into 8 x 45°
    bins = [
        (-22.5, 22.5, "right"),
        (22.5, 67.5, "down-right"),
        (67.5, 112.5, "down"),
        (112.5, 157.5, "down-left"),
        (157.5, 180.1, "left"),
        (-180.1, -157.5, "left"),
        (-157.5, -112.5, "up-left"),
        (-112.5, -67.5, "up"),
        (-67.5, -22.5, "up-right"),
    ]
    for lo, hi, name in bins:
        if lo <= angle_deg < hi:
            return name
    return "right"


def _distance_bucket(length: float) -> str:
    """0.0-1.0 stroke length → verbal intensity."""
    if length < 0.08:
        return "subtle"
    if length < 0.25:
        return "moderate"
    return "strong"


def _region(x: float, y: float) -> str:
    """Nine-slice region label from a point."""
    col = "left" if x < 1/3 else ("center" if x < 2/3 else "right")
    row = "top" if y < 1/3 else ("middle" if y < 2/3 else "bottom")
    if row == "middle" and col == "center":
        return "center"
    return f"{row}-{col}".replace("middle-", "")


def interpret_stroke(stroke: MotionStroke) -> dict:
    """Return a dict describing the stroke's motion for the prompt builder."""
    x0 = _clamp01(stroke.x0)
    y0 = _clamp01(stroke.y0)
    x1 = _clamp01(stroke.x1)
    y1 = _clamp01(stroke.y1)
    dx = x1 - x0
    dy = y1 - y0
    length = math.sqrt(dx * dx + dy * dy)
    return {
        "label": stroke.label.strip(),
        "direction": _cardinal(dx, dy),
        "intensity": _distance_bucket(length),
        "origin_region": _region(x0, y0),
        "end_region": _region(x1, y1),
        "length": round(length, 4),
    }


def _render_one(record: dict) -> str:
    """Render one stroke record into a short natural-language fragment."""
    subject = record["label"] or "subject"
    direction = record["direction"]
    intensity = record["intensity"]
    region = record["origin_region"]
    if direction == "still":
        return f"the {subject} stays still in the {region}"
    return (
        f"the {subject} in the {region} moves {intensity}ly to the {direction}"
    )


def format_prompt_suffix(strokes: Iterable[MotionStroke]) -> str:
    """Build a sentence that captures every stroke's intent.

    Returns an empty string when no strokes are passed in, so the caller can
    append unconditionally (`prompt + format_prompt_suffix(strokes)`) and
    produce the plain prompt back when there are no strokes.
    """
    pieces = [_render_one(interpret_stroke(s)) for s in strokes]
    pieces = [p for p in pieces if p]
    if not pieces:
        return ""
    if len(pieces) == 1:
        return ". Motion: " + pieces[0] + "."
    body = "; ".join(pieces[:-1]) + "; and " + pieces[-1]
    return ". Motion: " + body + "."


def dedupe_strokes(strokes: Iterable[MotionStroke],
                   min_length: float = 0.02) -> list[MotionStroke]:
    """Drop near-zero strokes (accidental taps) and exact duplicates.

    Pure utility used by the UI before posting - a canvas click-without-drag
    creates a zero-length stroke that would otherwise pollute the prompt.
    """
    seen: set[tuple] = set()
    out: list[MotionStroke] = []
    for s in strokes:
        dx = _clamp01(s.x1) - _clamp01(s.x0)
        dy = _clamp01(s.y1) - _clamp01(s.y0)
        length = math.sqrt(dx * dx + dy * dy)
        if length < min_length:
            continue
        key = (round(s.x0, 3), round(s.y0, 3), round(s.x1, 3), round(s.y1, 3),
               s.label.strip().lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out
