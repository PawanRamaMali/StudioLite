"""Audio-effect helpers: denoise, equalizer, background music mix.

Pure-Python wrappers around ffmpeg filters, split into argv builders and
dispatchers. Keeps argv assembly unit-testable without running ffmpeg.

Covers three issues:
- #13 Noise reduction (``afftdn`` frequency-domain, ``anlmdn`` time-domain)
- #14 Audio equalizer (10-band graphic EQ + presets)
- #12 Background music mix (amix with optional sidechain ducking + loop)
"""
from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


# ---------------------------------------------------------------------------
# Shared
# ---------------------------------------------------------------------------


_AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".opus", ".webm"}
_AUDIO_VIDEO_EXTS = _AUDIO_EXTS | {".mp4", ".mov", ".mkv", ".avi"}


def _run_default(cmd: list) -> Any:
    return subprocess.run(cmd, capture_output=True, text=True)


def _check_ffmpeg(ffmpeg_bin: str) -> Optional[str]:
    if shutil.which(ffmpeg_bin) is None:
        return "ffmpeg not on PATH. Install ffmpeg to use audio effects."
    return None


# ---------------------------------------------------------------------------
# #13 Noise Reduction
# ---------------------------------------------------------------------------


@dataclass
class DenoiseStyle:
    """``method='afftdn'`` is FFT-based broadband noise reduction with a
    tunable noise floor in dB. ``method='anlmdn'`` is time-domain
    non-local-means; slower, better for stationary hiss."""
    method: str = "afftdn"
    # afftdn: noise reduction in dB (negative), typical -20..-30
    noise_reduction_db: float = -25.0
    # afftdn: noise floor (dB). Audio below this is treated as noise.
    noise_floor_db: float = -40.0
    # anlmdn: smoothing strength (seconds, 0.001..0.01)
    smoothing: float = 0.002
    # Output codec choice
    output_format: str = "wav"  # wav|mp3


def build_denoise_filter(style: DenoiseStyle) -> str:
    """Return the -af filter string for the chosen method."""
    method = style.method if style.method in ("afftdn", "anlmdn") else "afftdn"
    if method == "afftdn":
        # Clamp inputs. nr is the attenuation amount (0..97), nf is noise floor.
        nr = max(0.0, min(97.0, abs(float(style.noise_reduction_db))))
        nf = max(-80.0, min(-20.0, float(style.noise_floor_db)))
        return f"afftdn=nr={nr:.1f}:nf={nf:.1f}"
    # anlmdn uses sigma (similarity) and smoothing (s/p params).
    s = max(0.0001, min(0.05, float(style.smoothing)))
    return f"anlmdn=s={s:.5f}:p=0.002:r=0.002"


def build_denoise_cmd(
    src: str,
    dst: str,
    style: DenoiseStyle,
    *,
    ffmpeg_bin: str = "ffmpeg",
) -> list:
    af = build_denoise_filter(style)
    cmd = [ffmpeg_bin, "-y", "-i", src, "-af", af]
    if style.output_format == "mp3":
        cmd += ["-c:a", "libmp3lame", "-b:a", "192k"]
    else:
        cmd += ["-c:a", "pcm_s16le"]
    cmd += [dst]
    return cmd


def denoise_audio(
    src: str,
    dst: str,
    *,
    style: Optional[DenoiseStyle] = None,
    ffmpeg_bin: str = "ffmpeg",
    runner: Callable[[list], Any] = _run_default,
) -> Dict[str, Any]:
    style = style or DenoiseStyle()
    if not os.path.isfile(src):
        return {"ok": False, "error": f"Source not found: {src}"}
    hint = _check_ffmpeg(ffmpeg_bin)
    if hint:
        return {"ok": False, "error": "ffmpeg missing", "install_hint": hint}

    cmd = build_denoise_cmd(src, dst, style, ffmpeg_bin=ffmpeg_bin)
    proc = runner(cmd)
    rc = getattr(proc, "returncode", 1)
    if rc != 0:
        err = getattr(proc, "stderr", "") or ""
        last = err.strip().splitlines()[-1] if err else "ffmpeg denoise failed"
        return {"ok": False, "error": last}
    if not os.path.isfile(dst) or os.path.getsize(dst) == 0:
        return {"ok": False, "error": "ffmpeg produced no output"}
    return {"ok": True, "backend": f"ffmpeg:{style.method}", "output_path": dst}


# ---------------------------------------------------------------------------
# #14 Audio Equalizer
# ---------------------------------------------------------------------------


# Classic 10-band frequencies (Hz) - cover audible range with musical spacing.
EQ_BANDS_HZ: List[int] = [31, 62, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]


# Presets are 10 gain values in dB, one per band in EQ_BANDS_HZ.
# "flat" means no change; presets emphasise/attenuate a region.
EQ_PRESETS: Dict[str, List[float]] = {
    "flat":      [0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
    "bass_boost":  [8,  7,  5,  3,  0,  0,  0,  0,  0,  0],
    "bass_cut":    [-10, -8, -5, -2, 0, 0, 0, 0, 0, 0],
    "vocal_boost": [-2, -2, 0,  2,  4,  5,  4,  2,  0,  0],
    "treble_boost": [0, 0, 0, 0, 0, 2, 4, 6, 7, 7],
    "treble_cut":  [0,  0,  0,  0,  0, -2, -5, -7, -9, -10],
    "rock":        [4,  3,  2,  1,  -1, -1, 2, 4, 5, 5],
    "pop":         [-2, -1, 0,  2,  4,  4, 2, 0, -1, -2],
    "podcast":     [-6, -4, -2, 0,  3,  4, 3, 1, 0, -2],
    "radio":       [-4, -2, 0, 1, 2, 3, 2, 1, 0, -4],
}


@dataclass
class EqStyle:
    preset: str = "flat"
    # per-band dB gains. When non-empty, overrides preset.
    gains_db: List[float] = field(default_factory=list)
    output_format: str = "wav"  # wav|mp3


def _resolve_gains(style: EqStyle) -> List[float]:
    if style.gains_db:
        src = list(style.gains_db)
    else:
        src = list(EQ_PRESETS.get(style.preset, EQ_PRESETS["flat"]))
    # Pad/truncate to 10 bands. Clamp each gain to [-18, +18] dB for safety.
    out: List[float] = []
    for i in range(len(EQ_BANDS_HZ)):
        g = float(src[i]) if i < len(src) else 0.0
        out.append(max(-18.0, min(18.0, g)))
    return out


def build_equalizer_filter(style: EqStyle) -> str:
    """Chain one ``equalizer=f=X:t=q:w=1:g=Y`` per band. Bands with zero
    gain are dropped so the graph stays shorter and ffmpeg logs cleaner."""
    gains = _resolve_gains(style)
    parts: List[str] = []
    for hz, g in zip(EQ_BANDS_HZ, gains):
        if abs(g) < 0.05:
            continue
        parts.append(f"equalizer=f={hz}:t=q:w=1:g={g:.2f}")
    return ",".join(parts) if parts else "anull"


def build_equalizer_cmd(
    src: str,
    dst: str,
    style: EqStyle,
    *,
    ffmpeg_bin: str = "ffmpeg",
) -> list:
    af = build_equalizer_filter(style)
    cmd = [ffmpeg_bin, "-y", "-i", src, "-af", af]
    if style.output_format == "mp3":
        cmd += ["-c:a", "libmp3lame", "-b:a", "192k"]
    else:
        cmd += ["-c:a", "pcm_s16le"]
    cmd += [dst]
    return cmd


def equalize_audio(
    src: str,
    dst: str,
    *,
    style: Optional[EqStyle] = None,
    ffmpeg_bin: str = "ffmpeg",
    runner: Callable[[list], Any] = _run_default,
) -> Dict[str, Any]:
    style = style or EqStyle()
    if not os.path.isfile(src):
        return {"ok": False, "error": f"Source not found: {src}"}
    hint = _check_ffmpeg(ffmpeg_bin)
    if hint:
        return {"ok": False, "error": "ffmpeg missing", "install_hint": hint}

    cmd = build_equalizer_cmd(src, dst, style, ffmpeg_bin=ffmpeg_bin)
    proc = runner(cmd)
    rc = getattr(proc, "returncode", 1)
    if rc != 0:
        err = getattr(proc, "stderr", "") or ""
        last = err.strip().splitlines()[-1] if err else "ffmpeg equalizer failed"
        return {"ok": False, "error": last}
    if not os.path.isfile(dst) or os.path.getsize(dst) == 0:
        return {"ok": False, "error": "ffmpeg produced no output"}
    return {"ok": True, "backend": "ffmpeg:equalizer", "bands_hz": EQ_BANDS_HZ,
            "gains_db": _resolve_gains(style), "output_path": dst}


# ---------------------------------------------------------------------------
# #12 Background Music mix
# ---------------------------------------------------------------------------


@dataclass
class BackgroundMusicStyle:
    """Mix background music under a primary (voice) track.

    ``music_volume`` / ``voice_volume`` are linear multipliers; ``duck``
    enables sidechain-compressor ducking so the music dips when voice is
    present. ``fade_in``/``fade_out`` are in seconds and applied to the
    music bed only. ``loop_music`` tiles the bed to match voice length."""
    music_volume: float = 0.3
    voice_volume: float = 1.0
    duck: bool = True
    duck_threshold: float = 0.05   # sidechaincompress threshold (linear)
    duck_ratio: float = 8.0        # compression ratio when voice present
    fade_in: float = 1.0
    fade_out: float = 2.0
    loop_music: bool = True
    output_format: str = "wav"     # wav|mp3


def _vol_clamp(v: float, lo: float = 0.0, hi: float = 4.0) -> float:
    return max(lo, min(hi, float(v)))


def build_background_music_filter_complex(
    *,
    voice_duration: Optional[float],
    style: BackgroundMusicStyle,
) -> str:
    """Build the -filter_complex graph for mixing [0:a] voice + [1:a] music.

    Inputs:
      [0:a] -> voice track
      [1:a] -> music track (possibly looped via -stream_loop)
    Output label: [mix]
    """
    v = _vol_clamp(style.voice_volume)
    m = _vol_clamp(style.music_volume)
    fin = max(0.0, float(style.fade_in))
    fout = max(0.0, float(style.fade_out))

    voice_chain = [f"volume={v:.3f}"]
    music_chain = [f"volume={m:.3f}"]

    if voice_duration is not None and voice_duration > 0:
        # Apply fades to music and optionally trim so amix doesn't run long.
        if fin > 0:
            music_chain.append(f"afade=t=in:st=0:d={fin:.2f}")
        if fout > 0:
            fade_start = max(0.0, voice_duration - fout)
            music_chain.append(f"afade=t=out:st={fade_start:.2f}:d={fout:.2f}")
        music_chain.append(f"atrim=0:{voice_duration:.3f}")
    else:
        if fin > 0:
            music_chain.append(f"afade=t=in:st=0:d={fin:.2f}")

    graph_parts: List[str] = []
    graph_parts.append(f"[0:a]{','.join(voice_chain)}[v0]")
    graph_parts.append(f"[1:a]{','.join(music_chain)}[m0]")

    if style.duck:
        # sidechaincompress: compressor on [m0], keyed by [v0]. When voice
        # exceeds threshold, music dips by ratio. [v0] is split so we also
        # feed it to the mix unchanged.
        thr = max(0.001, min(1.0, float(style.duck_threshold)))
        ratio = max(1.0, min(20.0, float(style.duck_ratio)))
        graph_parts.append("[v0]asplit=2[va][vb]")
        graph_parts.append(
            f"[m0][vb]sidechaincompress=threshold={thr:.3f}:ratio={ratio:.1f}:"
            "attack=20:release=250[md]"
        )
        graph_parts.append("[va][md]amix=inputs=2:duration=first:dropout_transition=0[mix]")
    else:
        graph_parts.append("[v0][m0]amix=inputs=2:duration=first:dropout_transition=0[mix]")

    return ";".join(graph_parts)


def build_background_music_cmd(
    voice_path: str,
    music_path: str,
    dst: str,
    *,
    voice_duration: Optional[float],
    style: BackgroundMusicStyle,
    ffmpeg_bin: str = "ffmpeg",
) -> list:
    cmd = [ffmpeg_bin, "-y", "-i", voice_path]
    if style.loop_music:
        cmd = [ffmpeg_bin, "-y", "-i", voice_path, "-stream_loop", "-1", "-i", music_path]
    else:
        cmd += ["-i", music_path]
    graph = build_background_music_filter_complex(
        voice_duration=voice_duration, style=style
    )
    cmd += ["-filter_complex", graph, "-map", "[mix]"]
    if style.output_format == "mp3":
        cmd += ["-c:a", "libmp3lame", "-b:a", "192k"]
    else:
        cmd += ["-c:a", "pcm_s16le"]
    if voice_duration is not None and voice_duration > 0:
        cmd += ["-t", f"{voice_duration:.3f}"]
    else:
        cmd += ["-shortest"]
    cmd += [dst]
    return cmd


def probe_duration(
    path: str,
    *,
    ffprobe_bin: str = "ffprobe",
    runner: Callable[[list], Any] = _run_default,
) -> Optional[float]:
    if shutil.which(ffprobe_bin) is None:
        return None
    try:
        proc = runner([
            ffprobe_bin, "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            path,
        ])
        if getattr(proc, "returncode", 1) != 0:
            return None
        out = (getattr(proc, "stdout", "") or "").strip()
        return float(out) if out else None
    except (ValueError, TypeError):
        return None


def mix_background_music(
    voice_path: str,
    music_path: str,
    dst: str,
    *,
    style: Optional[BackgroundMusicStyle] = None,
    ffmpeg_bin: str = "ffmpeg",
    ffprobe_bin: str = "ffprobe",
    runner: Callable[[list], Any] = _run_default,
) -> Dict[str, Any]:
    style = style or BackgroundMusicStyle()
    if not os.path.isfile(voice_path):
        return {"ok": False, "error": f"Voice track not found: {voice_path}"}
    if not os.path.isfile(music_path):
        return {"ok": False, "error": f"Music track not found: {music_path}"}
    hint = _check_ffmpeg(ffmpeg_bin)
    if hint:
        return {"ok": False, "error": "ffmpeg missing", "install_hint": hint}

    duration = probe_duration(voice_path, ffprobe_bin=ffprobe_bin, runner=runner)
    cmd = build_background_music_cmd(
        voice_path, music_path, dst,
        voice_duration=duration, style=style, ffmpeg_bin=ffmpeg_bin,
    )
    proc = runner(cmd)
    rc = getattr(proc, "returncode", 1)
    if rc != 0:
        err = getattr(proc, "stderr", "") or ""
        last = err.strip().splitlines()[-1] if err else "ffmpeg background-music mix failed"
        return {"ok": False, "error": last}
    if not os.path.isfile(dst) or os.path.getsize(dst) == 0:
        return {"ok": False, "error": "ffmpeg produced no output"}
    return {
        "ok": True,
        "backend": "ffmpeg:amix" + (":ducked" if style.duck else ""),
        "output_path": dst,
        "duration": duration,
    }
