"""Video collage helpers: hstack / vstack / grid / picture-in-picture.

Pure-Python wrappers around ffmpeg's ``hstack``, ``vstack``, ``xstack`` and
``overlay`` filters, split into argv builders + a dispatcher. Each input
is scaled+padded to a common per-cell size before stacking so clips of
mismatched dimensions tile cleanly without ffmpeg complaining about
"inputs with different SAR".

Covers issue #23 (Video Collage - Grid/split screen layouts).
"""
from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


_VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}


def _run_default(cmd: list) -> Any:
    return subprocess.run(cmd, capture_output=True, text=True)


def _check_ffmpeg(ffmpeg_bin: str) -> Optional[str]:
    if shutil.which(ffmpeg_bin) is None:
        return "ffmpeg not on PATH. Install ffmpeg to build video collages."
    return None


# ---------------------------------------------------------------------------
# Layout definitions
# ---------------------------------------------------------------------------


# Each entry gives: (name, cols, rows, min_inputs, max_inputs).
# ``xstack`` with N cells can be fed N-1 extras padded with a solid color
# when the user provides fewer clips than slots.
LAYOUTS: Dict[str, Tuple[int, int, int, int]] = {
    "hstack_2":  (2, 1, 2, 2),
    "vstack_2":  (1, 2, 2, 2),
    "hstack_3":  (3, 1, 2, 3),
    "vstack_3":  (1, 3, 2, 3),
    "grid_2x2":  (2, 2, 2, 4),
    "grid_3x3":  (3, 3, 2, 9),
    "pip":       (1, 1, 2, 2),   # picture-in-picture: base + corner overlay
}


# PIP corner coordinates: libavfilter expressions plugged into overlay's
# x:y. ``main_w`` / ``main_h`` refer to the base clip; ``overlay_w`` /
# ``overlay_h`` refer to the inset. 20-pixel margin for all corners.
_PIP_CORNERS = {
    "top-left":     "20:20",
    "top-right":    "main_w-overlay_w-20:20",
    "bottom-left":  "20:main_h-overlay_h-20",
    "bottom-right": "main_w-overlay_w-20:main_h-overlay_h-20",
}


@dataclass
class CollageStyle:
    """Collage knobs.

    ``layout`` picks one of ``LAYOUTS`` above. ``cell_width`` /
    ``cell_height`` is the per-cell output size (grids are
    cell * cols by cell * rows). ``background`` is the canvas colour used
    to pad short clips to the grid duration. ``pip_position`` is only
    read for ``layout='pip'``. ``pip_scale`` scales the inset relative to
    the base clip, 0.1..0.5 reasonable.
    """
    layout: str = "hstack_2"
    cell_width: int = 640
    cell_height: int = 360
    fps: int = 30
    background: str = "black"
    pip_position: str = "bottom-right"
    pip_scale: float = 0.28
    audio: str = "first"  # first|mix|drop
    crf: int = 20


# ---------------------------------------------------------------------------
# Filter graph construction
# ---------------------------------------------------------------------------


def _scale_pad_chain(cell_w: int, cell_h: int, fps: int, background: str) -> str:
    """Scale+pad a single input into a (cell_w x cell_h) cell at ``fps``."""
    return (
        f"scale={cell_w}:{cell_h}:force_original_aspect_ratio=decrease,"
        f"pad={cell_w}:{cell_h}:(ow-iw)/2:(oh-ih)/2:color={background},"
        f"fps={fps},setsar=1"
    )


def build_xstack_layout(cols: int, rows: int) -> str:
    """xstack's ``layout`` expression - comma-joined ``x_y`` cells, column-major
    within rows: ``0_0|w0_0|0_h0|w0_h0`` for 2x2.

    Uses the inputs' own widths/heights so the caller doesn't have to
    repeat the cell dimensions here."""
    cells: List[str] = []
    for r in range(rows):
        y = "0" if r == 0 else "+".join(f"h{i}" for i in range(r))
        for c in range(cols):
            x = "0" if c == 0 else "+".join(f"w{i}" for i in range(c))
            cells.append(f"{x}_{y}")
    return "|".join(cells)


def build_stack_filter_complex(
    n_inputs: int,
    *,
    style: CollageStyle,
) -> str:
    """Assemble the filter_complex graph for a stack layout.

    Produces N ``scale+pad+fps`` chains labelled ``[c0]..[cN-1]``, then
    either ``hstack``/``vstack``/``xstack`` into ``[v]``.
    """
    cols, rows, _, _ = LAYOUTS[style.layout]
    slots = cols * rows
    chain = _scale_pad_chain(style.cell_width, style.cell_height, style.fps, style.background)

    graph_parts: List[str] = []
    for i in range(n_inputs):
        graph_parts.append(f"[{i}:v]{chain}[c{i}]")

    # When fewer inputs than slots, pad the remaining slots with a solid
    # background lavfi source. The caller is responsible for adding the
    # matching ``-f lavfi -i color=...`` inputs.
    if style.layout in ("hstack_2", "hstack_3"):
        inputs = "".join(f"[c{i}]" for i in range(min(n_inputs, slots)))
        graph_parts.append(f"{inputs}hstack=inputs={min(n_inputs, slots)}[v]")
    elif style.layout in ("vstack_2", "vstack_3"):
        inputs = "".join(f"[c{i}]" for i in range(min(n_inputs, slots)))
        graph_parts.append(f"{inputs}vstack=inputs={min(n_inputs, slots)}[v]")
    else:
        # xstack grid
        inputs = "".join(f"[c{i}]" for i in range(min(n_inputs, slots)))
        layout = build_xstack_layout(cols, rows)
        graph_parts.append(
            f"{inputs}xstack=inputs={min(n_inputs, slots)}:layout={layout}:"
            f"fill={style.background}[v]"
        )
    return ";".join(graph_parts)


def build_pip_filter_complex(style: CollageStyle) -> str:
    """Base clip [0:v] + inset clip [1:v] -> [v]. Inset scales to
    ``pip_scale`` of base width."""
    pos = _PIP_CORNERS.get(style.pip_position, _PIP_CORNERS["bottom-right"])
    scale = max(0.05, min(0.5, float(style.pip_scale)))
    return (
        f"[0:v]fps={style.fps},setsar=1[base];"
        f"[1:v]scale=iw*{scale:.3f}:-2,fps={style.fps},setsar=1[pip];"
        f"[base][pip]overlay={pos}:shortest=1[v]"
    )


def _audio_map(n_inputs: int, mode: str) -> List[str]:
    """Return the ``-map`` / ``-filter_complex`` fragments to express the
    chosen audio behaviour."""
    if mode == "drop" or n_inputs == 0:
        return ["-an"]
    if mode == "mix" and n_inputs > 1:
        # Amix into [a]
        amix = "".join(f"[{i}:a]" for i in range(n_inputs))
        amix += f"amix=inputs={n_inputs}:duration=longest[a]"
        return ["-filter_complex_audio", amix, "-map", "[a]"]
    # "first" -> map the first input's audio track if any
    return ["-map", "0:a?"]


def build_collage_cmd(
    srcs: List[str],
    dst: str,
    *,
    style: CollageStyle,
    ffmpeg_bin: str = "ffmpeg",
) -> list:
    cmd = [ffmpeg_bin, "-y"]
    for s in srcs:
        cmd += ["-i", s]

    if style.layout == "pip":
        graph = build_pip_filter_complex(style)
    else:
        graph = build_stack_filter_complex(len(srcs), style=style)

    # Audio: mix handling is limited when the layout re-uses filter_complex,
    # so we fold audio mapping into the same graph for "mix".
    audio_parts: List[str]
    if style.audio == "mix" and len(srcs) > 1 and style.layout != "pip":
        amix = "".join(f"[{i}:a]" for i in range(len(srcs)))
        amix += f"amix=inputs={len(srcs)}:duration=longest[a]"
        full = graph + ";" + amix
        cmd += ["-filter_complex", full, "-map", "[v]", "-map", "[a]"]
        audio_parts = []
    elif style.audio == "drop":
        cmd += ["-filter_complex", graph, "-map", "[v]"]
        audio_parts = ["-an"]
    else:
        cmd += ["-filter_complex", graph, "-map", "[v]", "-map", "0:a?"]
        audio_parts = []

    cmd += audio_parts
    cmd += [
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", str(max(0, min(51, int(style.crf)))),
        "-preset", "medium",
    ]
    if style.audio != "drop":
        cmd += ["-c:a", "aac", "-b:a", "192k"]
    cmd += ["-movflags", "+faststart", dst]
    return cmd


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def build_collage(
    srcs: List[str],
    dst: str,
    *,
    style: Optional[CollageStyle] = None,
    ffmpeg_bin: str = "ffmpeg",
    runner: Callable[[list], Any] = _run_default,
) -> Dict[str, Any]:
    style = style or CollageStyle()
    if style.layout not in LAYOUTS:
        return {"ok": False, "error": f"Unknown layout: {style.layout}"}

    _cols, _rows, min_in, max_in = LAYOUTS[style.layout]
    if len(srcs) < min_in:
        return {"ok": False, "error": f"{style.layout} needs at least {min_in} clips"}
    if len(srcs) > max_in:
        return {"ok": False, "error": f"{style.layout} accepts at most {max_in} clips"}
    for s in srcs:
        if not os.path.isfile(s):
            return {"ok": False, "error": f"Source not found: {s}"}

    hint = _check_ffmpeg(ffmpeg_bin)
    if hint:
        return {"ok": False, "error": "ffmpeg missing", "install_hint": hint}

    cmd = build_collage_cmd(srcs, dst, style=style, ffmpeg_bin=ffmpeg_bin)
    proc = runner(cmd)
    rc = getattr(proc, "returncode", 1)
    if rc != 0:
        err = getattr(proc, "stderr", "") or ""
        last = err.strip().splitlines()[-1] if err else "ffmpeg collage failed"
        return {"ok": False, "error": last}
    if not os.path.isfile(dst) or os.path.getsize(dst) == 0:
        return {"ok": False, "error": "ffmpeg produced no output"}
    return {
        "ok": True,
        "backend": f"ffmpeg:{style.layout}",
        "output_path": dst,
        "layout": style.layout,
        "cells": LAYOUTS[style.layout][0] * LAYOUTS[style.layout][1],
    }
