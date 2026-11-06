"""Image-to-video slideshow helpers.

Pure-Python wrappers around ffmpeg that turn an ordered list of images
into a single mp4 at a fixed resolution/fps, with:
- per-image display duration (same for all, or per-image override)
- an optional transition between adjacent images (xfade: fade, wipeleft,
  slideleft, dissolve, pixelize, circleopen)
- an optional slow Ken Burns zoom on each still (gives static slideshows
  a cinematic drift at zero model cost)
- an optional background music track that's trimmed/padded to the full
  slideshow length

ffmpeg argv assembly is pulled out as pure functions so the exact command
is testable without actually invoking ffmpeg; the top-level
``generate_slideshow`` takes an injectable ``runner`` that defaults to
``subprocess.run`` - tests pass a fake.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


# xfade transitions supported here. The full list upstream is much longer
# but these cover 95% of slideshow use and keep the UI picker short.
_XFADE_TRANSITIONS = {
    "fade", "wipeleft", "wiperight", "slideleft", "slideright",
    "dissolve", "pixelize", "circleopen", "radial", "smoothleft",
}

# Ken Burns direction codes. Kept small and intuitive.
_KENBURNS_MOVES = {"none", "zoom-in", "zoom-out", "pan-left", "pan-right"}


@dataclass
class SlideshowStyle:
    """All the dials for one slideshow render.

    Field defaults match a sensible "upload this to YouTube" preset:
    1920x1080 at 30fps, 3 seconds per still, a 0.5s crossfade, no
    Ken Burns. Overrides let callers go fancier."""
    width: int = 1920
    height: int = 1080
    fps: int = 30
    per_image_duration: float = 3.0
    transition: str = "fade"           # "none" disables xfade
    transition_duration: float = 0.5
    kenburns: str = "none"             # see _KENBURNS_MOVES
    audio_volume: float = 1.0          # 0..1 multiplier on background music


# ---------------------------------------------------------------------------
# Ken Burns (per-image zoompan clause)
# ---------------------------------------------------------------------------


def kenburns_expr(move: str, frames: int, fps: int) -> str:
    """Return the ``z=...:x=...:y=...:d=N`` clause for one image.

    The expressions are deliberately tame - slideshows look natural when
    the motion is barely visible. Same shape as filmmaker/agents.py's
    helper but kept local so slideshow has no cross-module dep."""
    span = max(frames - 1, 1)
    m = (move or "none").lower()
    if m == "zoom-in":
        return (
            f"z='min(zoom+0.0010,1.20)':"
            f"x='iw/2-(iw/zoom/2)':"
            f"y='ih/2-(ih/zoom/2)':d={frames}"
        )
    if m == "zoom-out":
        return (
            f"z='if(lte(zoom,1.0),1.20,max(1.001,zoom-0.0010))':"
            f"x='iw/2-(iw/zoom/2)':"
            f"y='ih/2-(ih/zoom/2)':d={frames}"
        )
    if m == "pan-left":
        return (
            f"z='1.15':"
            f"x='(iw-iw/zoom)*(1-on/{span})':"
            f"y='(ih-ih/zoom)/2':d={frames}"
        )
    if m == "pan-right":
        return (
            f"z='1.15':"
            f"x='(iw-iw/zoom)*on/{span}':"
            f"y='(ih-ih/zoom)/2':d={frames}"
        )
    # "none" - a static hold on the image, no pan or zoom.
    return (
        f"z='1.0':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={frames}"
    )


# ---------------------------------------------------------------------------
# Per-image clip construction
# ---------------------------------------------------------------------------


def build_image_clip_cmd(
    src: str,
    dst: str,
    *,
    duration: float,
    style: SlideshowStyle,
    ffmpeg_bin: str = "ffmpeg",
) -> List[str]:
    """Render one image as a dst.mp4 of ``duration`` seconds at the style's
    resolution and fps. Ken Burns is off by default (static image + scale).

    Using H.264 CRF 20 yuv420p so every clip concatenates / cross-fades
    without a re-encode mismatch."""
    fps = int(style.fps)
    frames = max(1, int(round(float(duration) * fps)))
    # Scale + pad into the target frame so mixed-aspect source images don't
    # distort. Letterbox colour stays black by default.
    pre = (
        f"scale={style.width}:{style.height}:force_original_aspect_ratio=decrease:flags=lanczos,"
        f"pad={style.width}:{style.height}:(ow-iw)/2:(oh-ih)/2,setsar=1"
    )
    if (style.kenburns or "none").lower() != "none":
        # zoompan wants a large source so its crop keeps edges sharp.
        pre = (
            f"scale={style.width*2}:{style.height*2}:force_original_aspect_ratio=increase:flags=lanczos,"
            f"crop={style.width*2}:{style.height*2},setsar=1,"
            f"zoompan={kenburns_expr(style.kenburns, frames, fps)}:"
            f"s={style.width}x{style.height}:fps={fps}"
        )
    vf = f"{pre},fps={fps}"
    return [
        ffmpeg_bin, "-y",
        "-loop", "1", "-i", src,
        "-vf", vf,
        "-t", f"{duration:.3f}",
        "-fps_mode", "cfr",
        "-pix_fmt", "yuv420p",
        "-c:v", "libx264", "-preset", "medium", "-crf", "20",
        dst,
    ]


# ---------------------------------------------------------------------------
# Transitions / concat
# ---------------------------------------------------------------------------


def build_xfade_graph(
    clip_paths: List[str],
    *,
    style: SlideshowStyle,
    durations: List[float],
) -> str:
    """Build an xfade filter_complex graph that chains every clip through
    a transition. The i-th transition starts at
    ``sum(durations[0..i]) - transition_duration``, which is the moment
    clip i reaches its last ``transition_duration`` seconds."""
    n = len(clip_paths)
    if n < 2:
        return ""  # single-clip slideshow doesn't need xfade
    td = float(style.transition_duration)
    xf = style.transition
    chain: List[str] = []
    prev = "[0:v]"
    accum = 0.0
    for i in range(1, n):
        accum += durations[i - 1] - td
        out_label = f"[v{i}]"
        chain.append(
            f"{prev}[{i}:v]xfade=transition={xf}:duration={td:.3f}:offset={accum:.3f}{out_label}"
        )
        prev = out_label
    return ";".join(chain)


def total_duration(durations: List[float], style: SlideshowStyle) -> float:
    """Final slideshow length given per-image durations and whether the
    transition shortens overlap. Each xfade overlaps the two neighbouring
    clips by ``transition_duration`` seconds."""
    if not durations:
        return 0.0
    if style.transition == "none" or len(durations) < 2:
        return sum(durations)
    return sum(durations) - (len(durations) - 1) * float(style.transition_duration)


# ---------------------------------------------------------------------------
# Full assembly: image clips -> xfade (or concat) -> optional audio mux
# ---------------------------------------------------------------------------


def _default_runner(cmd: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def _run(cmd, runner):
    try:
        proc = (runner or _default_runner)(cmd)
    except FileNotFoundError:
        return None, {
            "ok": False,
            "error": "ffmpeg not found",
            "install_hint": "Install ffmpeg and ensure it is on PATH.",
        }
    except OSError as e:
        return None, {"ok": False, "error": f"ffmpeg launch failed: {e}"}
    return proc, None


def _stderr_tail(proc):
    stderr = (getattr(proc, "stderr", "") or "").strip()
    return stderr.splitlines()[-1] if stderr else f"exit {proc.returncode}"


def generate_slideshow(
    images: List[str],
    out_path: str,
    *,
    style: Optional[SlideshowStyle] = None,
    durations: Optional[List[float]] = None,
    audio_path: Optional[str] = None,
    work_dir: Optional[str] = None,
    ffmpeg_bin: str = "ffmpeg",
    runner: Optional[Callable[[List[str]], subprocess.CompletedProcess]] = None,
) -> Dict[str, Any]:
    """Build a slideshow mp4 from ordered image paths.

    Returns ``{ok, backend, output_path, duration_sec, image_count}`` on
    success or ``{ok: False, error, install_hint?}`` on any failure.
    Never raises.
    """
    if not images:
        return {"ok": False, "error": "at least one image is required"}
    for p in images:
        if not os.path.isfile(p):
            return {"ok": False, "error": f"image not found: {p}"}
    if audio_path and not os.path.isfile(audio_path):
        return {"ok": False, "error": f"audio not found: {audio_path}"}

    style = style or SlideshowStyle()
    if style.transition != "none" and style.transition not in _XFADE_TRANSITIONS:
        return {"ok": False,
                "error": f"unknown transition: {style.transition}"}
    if (style.kenburns or "none").lower() not in _KENBURNS_MOVES:
        return {"ok": False,
                "error": f"unknown kenburns mode: {style.kenburns}"}

    # Pad/truncate per-image durations to match images.
    durs: List[float] = list(durations or [])
    while len(durs) < len(images):
        durs.append(float(style.per_image_duration))
    durs = durs[: len(images)]
    # Transition must fit inside the shorter of the two neighbouring clips,
    # or ffmpeg's xfade rejects the graph.
    if style.transition != "none" and len(images) > 1:
        min_dur = min(durs)
        if style.transition_duration >= min_dur:
            return {"ok": False,
                    "error": (f"transition_duration {style.transition_duration}s "
                              f"is >= the shortest image duration {min_dur}s")}

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    work = work_dir or os.path.join(os.path.dirname(out_path) or ".",
                                     "_slideshow_work")
    os.makedirs(work, exist_ok=True)
    ff = shutil.which(ffmpeg_bin) or ffmpeg_bin

    # 1) Render each image to a per-clip mp4 at the target style.
    clip_paths: List[str] = []
    for i, (img, dur) in enumerate(zip(images, durs)):
        dst = os.path.join(work, f"clip_{i:03d}.mp4")
        cmd = build_image_clip_cmd(img, dst, duration=dur, style=style,
                                    ffmpeg_bin=ff)
        proc, err = _run(cmd, runner)
        if err is not None:
            return err
        if proc.returncode != 0:
            return {"ok": False,
                    "error": f"image clip {i} failed: {_stderr_tail(proc)}"}
        if not os.path.exists(dst):
            return {"ok": False,
                    "error": f"image clip {i} produced no output"}
        clip_paths.append(dst)

    # 2) Combine with xfade or plain concat demuxer.
    muxed_video = os.path.join(work, "muxed.mp4")
    if style.transition == "none" or len(clip_paths) < 2:
        # Plain concat demuxer. Stream-copies since all clips share codecs.
        concat_txt = os.path.join(work, "concat.txt")
        with open(concat_txt, "w", encoding="utf-8") as f:
            for p in clip_paths:
                f.write(f"file '{p.replace(os.sep, '/')}'\n")
        cmd = [
            ff, "-y",
            "-f", "concat", "-safe", "0",
            "-i", concat_txt,
            "-c", "copy",
            muxed_video,
        ]
    else:
        graph = build_xfade_graph(clip_paths, style=style, durations=durs)
        cmd = [ff, "-y"]
        for p in clip_paths:
            cmd += ["-i", p]
        # The last label in build_xfade_graph is v{n-1}.
        last_label = f"[v{len(clip_paths)-1}]"
        cmd += [
            "-filter_complex", graph,
            "-map", last_label,
            "-fps_mode", "cfr",
            "-pix_fmt", "yuv420p",
            "-c:v", "libx264", "-preset", "medium", "-crf", "20",
            muxed_video,
        ]
    proc, err = _run(cmd, runner)
    if err is not None:
        return err
    if proc.returncode != 0:
        return {"ok": False, "error": f"concat failed: {_stderr_tail(proc)}"}
    if not os.path.exists(muxed_video):
        return {"ok": False, "error": "concat produced no output"}

    # 3) If no audio, we're done - just move the muxed clip into place.
    total_sec = total_duration(durs, style)
    if not audio_path:
        try:
            shutil.copyfile(muxed_video, out_path)
        except OSError as e:
            return {"ok": False, "error": f"copy failed: {e}"}
        return {
            "ok": True,
            "backend": "ffmpeg",
            "output_path": out_path,
            "duration_sec": round(total_sec, 3),
            "image_count": len(images),
        }

    # 4) Mux the background music. Trim audio to the slideshow length so a
    # 10-minute song on a 30-second slideshow doesn't make the output
    # longer than the picture track.
    vol = max(0.0, min(2.0, float(style.audio_volume)))
    af = f"volume={vol:.3f}" if vol != 1.0 else None
    cmd = [
        ff, "-y",
        "-i", muxed_video,
        "-i", audio_path,
    ]
    if af:
        cmd += ["-filter_complex", f"[1:a]{af}[a]", "-map", "0:v", "-map", "[a]"]
    else:
        cmd += ["-map", "0:v", "-map", "1:a"]
    cmd += [
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        "-t", f"{total_sec:.3f}",
        out_path,
    ]
    proc, err = _run(cmd, runner)
    if err is not None:
        return err
    if proc.returncode != 0:
        return {"ok": False, "error": f"audio mux failed: {_stderr_tail(proc)}"}
    if not os.path.exists(out_path):
        return {"ok": False, "error": "audio mux produced no output"}

    return {
        "ok": True,
        "backend": "ffmpeg",
        "output_path": out_path,
        "duration_sec": round(total_sec, 3),
        "image_count": len(images),
        "has_audio": True,
    }
