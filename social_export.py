"""One-click Social Export: reshape a finished video for Reels/Shorts/TikTok.

Pure argv builders for the ffmpeg steps and one orchestrator that runs them.
Keeping the argv builders as pure functions makes the pipeline testable without
ffmpeg installed - each builder returns the exact argv list, tests assert on
it, and the orchestrator feeds it to a ``runner`` callable that the real API
replaces with ``subprocess.run``.

Pipeline stages (any may be disabled):
1. Pad or crop to the target aspect, with an optional blurred backdrop for
   portrait-from-landscape so bars don't show solid black.
2. Burn subtitles inside the format's safe zone.
3. Loudness-normalize to the platform target LUFS.
4. Export as H.264 + AAC MP4 under the platform's bitrate cap.
"""

from __future__ import annotations

import os
import shlex
import subprocess
from dataclasses import dataclass
from typing import Callable, Optional

from video_formats import FormatPreset, resolve_format, safe_zone_px


@dataclass
class SocialExportStyle:
    """Knobs the API/UI expose; everything else comes from the format preset."""
    format_id: str = "reels"
    fit_mode: str = "pad_blur"    # "pad_blur", "pad_black", "crop", "stretch"
    subtitle_path: str = ""       # SRT/VTT to burn in; "" to skip
    normalize_loudness: bool = True
    max_duration_s: Optional[float] = None  # None = platform cap
    caption_font: str = "Arial"
    caption_font_size_pct: float = 0.045  # fraction of height
    caption_color: str = "white"
    caption_box_color: str = "black@0.5"
    audio_bitrate_kbps: int = 192
    ffmpeg_bin: str = "ffmpeg"
    ffprobe_bin: str = "ffprobe"


def _fit_filter(fit_mode: str, w: int, h: int) -> str:
    """Return the first-stage video filter chain string for the fit mode."""
    if fit_mode == "stretch":
        return f"scale={w}:{h}:flags=lanczos"
    if fit_mode == "crop":
        # Scale up to cover, then center-crop to exact dims
        return (
            f"scale={w}:{h}:force_original_aspect_ratio=increase:flags=lanczos,"
            f"crop={w}:{h}"
        )
    if fit_mode == "pad_black":
        return (
            f"scale={w}:{h}:force_original_aspect_ratio=decrease:flags=lanczos,"
            f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=black"
        )
    # pad_blur: duplicate the stream, blur the copy, use it as the backdrop
    # [v] = main centered, [bg] = blurred stretched background
    return (
        f"split[orig][bg];"
        f"[bg]scale={w}:{h}:flags=lanczos,boxblur=luma_radius=40:luma_power=1[bg2];"
        f"[orig]scale={w}:{h}:force_original_aspect_ratio=decrease:flags=lanczos[fg];"
        f"[bg2][fg]overlay=(W-w)/2:(H-h)/2"
    )


def _subtitle_filter(subtitle_path: str, preset: FormatPreset, style: SocialExportStyle) -> str:
    """Burn SRT/VTT with a force_style that respects the safe zone."""
    if not subtitle_path:
        return ""
    # libass force_style tokens (comma-separated). MarginV uses script px not %
    top, right, bottom, left = safe_zone_px(preset)
    font_size = max(18, int(round(preset.height * style.caption_font_size_pct)))
    # Escape the path for ffmpeg's subtitles filter - on Windows the drive
    # colon needs an extra backslash so ffmpeg doesn't read it as a key:value.
    esc = subtitle_path.replace("\\", "/").replace(":", "\\:")
    force_style = (
        f"FontName={style.caption_font},"
        f"FontSize={font_size},"
        f"PrimaryColour=&H00FFFFFF,"
        f"OutlineColour=&H00000000,"
        f"BorderStyle=3,"       # opaque box
        f"Outline=2,"
        f"Shadow=0,"
        f"Alignment=2,"         # bottom-center
        f"MarginV={bottom + 10}"  # sit above the safe-zone cutoff
    )
    return f"subtitles=filename='{esc}':force_style='{force_style}'"


def build_video_filter_chain(preset: FormatPreset, style: SocialExportStyle) -> str:
    """Return the full -vf argument. Caller passes through -filter_complex if blur is used."""
    chain = _fit_filter(style.fit_mode, preset.width, preset.height)
    if style.subtitle_path:
        sub = _subtitle_filter(style.subtitle_path, preset, style)
        if sub:
            # For pad_blur we're already in filter_complex land; tack on with comma
            chain = f"{chain},{sub}"
    return chain


def build_audio_filter(preset: FormatPreset, style: SocialExportStyle) -> str:
    """Return the -af argument. EBU-R128 loudnorm with the format's target LUFS."""
    if not style.normalize_loudness:
        return "anull"
    return (
        f"loudnorm=I={preset.audio_lufs}:LRA=11:TP=-1.5:"
        f"linear=true:print_format=summary"
    )


def build_export_argv(input_path: str, output_path: str, style: SocialExportStyle) -> list[str]:
    """Build the full ffmpeg command. Pure: no disk access, no ffmpeg invocation."""
    preset = resolve_format(style.format_id)
    vf = build_video_filter_chain(preset, style)
    af = build_audio_filter(preset, style)

    duration = style.max_duration_s if style.max_duration_s is not None else preset.max_duration_s
    # Platform bitrate cap converted to an ffmpeg -b:v / -maxrate. We go about
    # 75% of the cap for the video stream so audio + container overhead fits.
    target_v_kbps = int(preset.max_bitrate_mbps * 1000 * 0.75)

    argv = [
        style.ffmpeg_bin, "-y",
        "-i", input_path,
        "-t", f"{duration:.3f}",
    ]

    # pad_blur needs filter_complex (two inputs from one via split); the other
    # modes are single-chain and fit -vf.
    if style.fit_mode == "pad_blur":
        argv += ["-filter_complex", vf]
    else:
        argv += ["-vf", vf]

    argv += [
        "-r", str(preset.target_fps),
        "-c:v", "libx264",
        "-preset", "medium",
        "-profile:v", "high",
        "-level", "4.2",
        "-pix_fmt", "yuv420p",
        "-b:v", f"{target_v_kbps}k",
        "-maxrate", f"{target_v_kbps}k",
        "-bufsize", f"{target_v_kbps * 2}k",
        "-af", af,
        "-c:a", "aac",
        "-b:a", f"{style.audio_bitrate_kbps}k",
        "-ar", "48000",
        "-ac", "2",
        "-movflags", "+faststart",
        output_path,
    ]
    return argv


def probe_duration(path: str, ffprobe_bin: str = "ffprobe",
                   runner: Optional[Callable] = None) -> Optional[float]:
    """Return ``path``'s duration in seconds via ffprobe, or None if unreadable."""
    import json
    argv = [
        ffprobe_bin, "-v", "quiet", "-print_format", "json",
        "-show_format", path,
    ]
    r = runner if runner is not None else (
        lambda a: subprocess.run(a, capture_output=True, text=True, check=False)
    )
    try:
        proc = r(argv)
    except (OSError, FileNotFoundError):
        return None
    if getattr(proc, "returncode", 1) != 0:
        return None
    try:
        data = json.loads(proc.stdout or "{}")
        return float(data.get("format", {}).get("duration", 0.0)) or None
    except (json.JSONDecodeError, ValueError, TypeError):
        return None


def run_export(input_path: str, output_path: str, style: SocialExportStyle,
               runner: Optional[Callable] = None) -> dict:
    """Execute the pipeline end-to-end. Returns a summary dict.

    ``runner`` is injectable: tests pass a fake that records the argv without
    invoking ffmpeg. In production the default calls ``subprocess.run``.
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input not found: {input_path}")

    preset = resolve_format(style.format_id)

    # Honor platform duration cap even if the caller forgot to pass one
    if style.max_duration_s is None:
        src_dur = probe_duration(input_path, style.ffprobe_bin, runner)
        if src_dur is not None:
            style = SocialExportStyle(
                **{**style.__dict__, "max_duration_s": min(src_dur, preset.max_duration_s)}
            )

    argv = build_export_argv(input_path, output_path, style)
    r = runner if runner is not None else (
        lambda a: subprocess.run(a, capture_output=True, text=True, check=False)
    )
    proc = r(argv)

    ok = getattr(proc, "returncode", 1) == 0
    return {
        "ok": ok,
        "output_path": output_path if ok else "",
        "format": {
            "id": preset.id,
            "label": preset.label,
            "width": preset.width,
            "height": preset.height,
            "fps": preset.target_fps,
            "aspect": preset.aspect,
        },
        "argv": argv,
        "stderr": getattr(proc, "stderr", "") if not ok else "",
    }
