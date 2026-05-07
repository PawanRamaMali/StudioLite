"""
Audio Studio for StudioLite.

Enhanced audio pipeline: TTS, sound effects, voice isolation,
lip sync preparation, and multi-language dubbing support.
"""

import os
import subprocess
import numpy as np
from uuid import uuid4
from typing import Callable, Optional

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(ROOT_DIR, ".mp")


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================
# Voice Isolation (Vocal Separation)
# =============================================

def isolate_voice(audio_path: str, output_path: str = None) -> dict:
    """
    Separate vocals from background noise/music.
    Uses FFmpeg's built-in audio filters as a lightweight approach.
    For better quality, install demucs: pip install demucs

    Returns dict with 'vocals' and 'background' paths.
    """
    ensure_output_dir()
    base = os.path.splitext(os.path.basename(audio_path))[0]
    vocals_path = output_path or os.path.join(OUTPUT_DIR, f"vocals_{uuid4()}.wav")
    bg_path = os.path.join(OUTPUT_DIR, f"background_{uuid4()}.wav")

    # Try demucs first (best quality)
    try:
        import demucs.separate
        demucs.separate.main([
            "--two-stems", "vocals",
            "-o", OUTPUT_DIR,
            audio_path,
        ])
        # Demucs outputs to OUTPUT_DIR/htdemucs/basename/vocals.wav
        demucs_dir = os.path.join(OUTPUT_DIR, "htdemucs", base)
        if os.path.exists(os.path.join(demucs_dir, "vocals.wav")):
            import shutil
            shutil.move(os.path.join(demucs_dir, "vocals.wav"), vocals_path)
            shutil.move(os.path.join(demucs_dir, "no_vocals.wav"), bg_path)
            return {"vocals": vocals_path, "background": bg_path, "method": "demucs"}
    except (ImportError, Exception):
        pass

    # Fallback: FFmpeg high-pass filter to isolate voice frequencies
    try:
        subprocess.run([
            "ffmpeg", "-y", "-i", audio_path,
            "-af", "highpass=f=200,lowpass=f=3000",
            vocals_path,
        ], capture_output=True, check=True)

        subprocess.run([
            "ffmpeg", "-y", "-i", audio_path,
            "-af", "lowpass=f=200",
            bg_path,
        ], capture_output=True, check=True)

        return {"vocals": vocals_path, "background": bg_path, "method": "ffmpeg_filter"}
    except Exception as e:
        return {"error": str(e)}


# =============================================
# Sound Effects Generation
# =============================================

# Built-in SFX library (procedural)
SFX_LIBRARY = {
    "whoosh": "Fast air movement sound",
    "explosion": "Dramatic explosion",
    "footsteps": "Walking footsteps",
    "rain": "Rain falling",
    "thunder": "Thunder crack",
    "wind": "Blowing wind",
    "door_close": "Door closing",
    "heartbeat": "Rhythmic heartbeat",
    "clock_tick": "Ticking clock",
    "typing": "Keyboard typing",
    "ocean_waves": "Ocean waves crashing",
    "fire_crackling": "Campfire crackling",
    "bird_chirp": "Birds chirping",
    "car_engine": "Car engine running",
    "glass_break": "Glass shattering",
}


SFX_ENGINE_VERSION = "procedural-v2"

# Sample rate used by the procedural generators below. 44.1 kHz so stereo
# effects (rain, wind, ocean, footsteps) sound spacious and don't alias.
_SR = 44100


def _butter_lowpass(audio: np.ndarray, cutoff: float, sr: int = _SR, order: int = 4) -> np.ndarray:
    from scipy.signal import butter, filtfilt
    nyq = 0.5 * sr
    b, a = butter(order, min(cutoff / nyq, 0.99), btype="low")
    return filtfilt(b, a, audio).astype(np.float32)


def _butter_highpass(audio: np.ndarray, cutoff: float, sr: int = _SR, order: int = 4) -> np.ndarray:
    from scipy.signal import butter, filtfilt
    nyq = 0.5 * sr
    b, a = butter(order, max(cutoff / nyq, 0.01), btype="high")
    return filtfilt(b, a, audio).astype(np.float32)


def _butter_bandpass(audio: np.ndarray, low: float, high: float, sr: int = _SR, order: int = 4) -> np.ndarray:
    from scipy.signal import butter, filtfilt
    nyq = 0.5 * sr
    b, a = butter(order, [max(low / nyq, 0.01), min(high / nyq, 0.99)], btype="band")
    return filtfilt(b, a, audio).astype(np.float32)


def _stereo(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    n = min(len(left), len(right))
    return np.stack([left[:n], right[:n]], axis=1).astype(np.float32)


def _rain(duration: float, sr: int) -> np.ndarray:
    # Stereo: pink-ish ambient hiss + decorrelated droplets per channel.
    n = int(duration * sr)
    rng = np.random.default_rng()

    def channel():
        # Ambient: low-pass filtered noise with slight density variation
        ambient = _butter_lowpass(rng.standard_normal(n).astype(np.float32), 4500.0, sr) * 0.35
        ambient *= (0.85 + 0.15 * np.sin(2 * np.pi * 0.4 * np.linspace(0, duration, n)))
        # Droplets: random short transients with high-frequency content
        drops = np.zeros(n, dtype=np.float32)
        n_drops = int(duration * 60)
        for _ in range(n_drops):
            pos = rng.integers(0, n)
            length = int(sr * 0.012)
            end = min(pos + length, n)
            t = np.arange(end - pos) / sr
            freq = rng.uniform(2500, 5500)
            drops[pos:end] += (rng.uniform(0.3, 0.9) *
                               np.sin(2 * np.pi * freq * t) *
                               np.exp(-t * 220)).astype(np.float32)
        return ambient + drops * 0.6

    return _stereo(channel(), channel())


def _thunder(duration: float, sr: int) -> np.ndarray:
    n = int(duration * sr)
    rng = np.random.default_rng()
    t = np.linspace(0, duration, n, endpoint=False)

    # Initial sharp crack (broadband, very short)
    crack_n = int(0.08 * sr)
    crack = rng.standard_normal(crack_n).astype(np.float32)
    crack *= np.exp(-np.linspace(0, 1, crack_n) * 18)
    crack = _butter_bandpass(crack, 800, 8000, sr) * 1.2

    # Deep rumble: sub-bass + low-pass noise with slow undulation
    noise = rng.standard_normal(n).astype(np.float32)
    rumble_noise = _butter_lowpass(noise, 200.0, sr) * 0.9
    sub = (np.sin(2 * np.pi * 28 * t + np.cumsum(rng.standard_normal(n)) * 0.005) * 0.55).astype(np.float32)
    rumble = (rumble_noise + sub) * np.exp(-t * (1.4 / max(duration, 0.6)))

    out = rumble.copy()
    out[:crack_n] += crack
    return out.astype(np.float32)


def _wind(duration: float, sr: int) -> np.ndarray:
    n = int(duration * sr)
    rng = np.random.default_rng()
    t = np.linspace(0, duration, n, endpoint=False)

    def channel(phase_offset: float):
        base = rng.standard_normal(n).astype(np.float32)
        # Resonant moving-bandpass: simulate gusts by mixing two filtered streams
        low_hum = _butter_bandpass(base, 80, 400, sr) * 0.7
        high_whistle = _butter_bandpass(rng.standard_normal(n).astype(np.float32), 800, 2200, sr) * 0.3
        # Slow gust modulation
        gust = 0.55 + 0.45 * np.sin(2 * np.pi * 0.18 * t + phase_offset)
        gust *= 0.7 + 0.3 * np.sin(2 * np.pi * 0.07 * t + 1.7 + phase_offset)
        return (low_hum + high_whistle * gust) * gust

    return _stereo(channel(0.0), channel(0.9))


def _ocean_waves(duration: float, sr: int) -> np.ndarray:
    n = int(duration * sr)
    rng = np.random.default_rng()
    t = np.linspace(0, duration, n, endpoint=False)

    def channel(phase: float):
        # Multi-period swell envelope (a wave cycle is roughly 4-7s)
        swell = (np.sin(2 * np.pi * 0.18 * t + phase) +
                 0.5 * np.sin(2 * np.pi * 0.23 * t + phase + 1.3))
        envelope = (0.45 + 0.55 * np.maximum(0, swell) ** 1.4)
        # Filtered noise = water rush, plus high-frequency foam at peaks
        rush = _butter_bandpass(rng.standard_normal(n).astype(np.float32), 200, 1800, sr)
        foam = _butter_highpass(rng.standard_normal(n).astype(np.float32), 4000, sr) * 0.3
        return (rush + foam * np.maximum(0, swell)) * envelope * 0.8

    return _stereo(channel(0.0), channel(0.7))


def _heartbeat(duration: float, sr: int) -> np.ndarray:
    bpm = 70
    period = 60.0 / bpm
    n = int(duration * sr)
    out = np.zeros(n, dtype=np.float32)
    t_beat = 0.0
    rng = np.random.default_rng()
    while t_beat < duration:
        # First thump (lub): 60 Hz sine with sharp envelope
        for offset, freq, amp, decay in ((0.0, 60.0, 1.0, 22.0), (0.18, 50.0, 0.7, 28.0)):
            pos = int((t_beat + offset) * sr)
            if pos >= n:
                break
            length = min(int(0.18 * sr), n - pos)
            tt = np.arange(length) / sr
            # Slight pitch wobble for organic feel
            wobble = 1 + 0.04 * np.sin(2 * np.pi * 7 * tt)
            beat = amp * np.sin(2 * np.pi * freq * tt * wobble) * np.exp(-tt * decay)
            out[pos:pos + length] += beat.astype(np.float32)
        # Tiny variability in interval
        t_beat += period * (1 + rng.uniform(-0.04, 0.04))
    # Low-pass to soften
    return _butter_lowpass(out, 800.0, sr)


def _explosion(duration: float, sr: int) -> np.ndarray:
    n = int(duration * sr)
    rng = np.random.default_rng()
    t = np.linspace(0, duration, n, endpoint=False)
    # Bass boom + broadband noise + sub thump
    noise = rng.standard_normal(n).astype(np.float32)
    boom = _butter_lowpass(noise, 250.0, sr) * np.exp(-t * (2.5 / max(duration, 0.5)))
    sub = (np.sin(2 * np.pi * 45 * t) * np.exp(-t * 6)).astype(np.float32)
    high = _butter_bandpass(noise, 1500, 6000, sr) * np.exp(-t * 18)
    return (boom * 1.4 + sub * 1.0 + high * 0.5).astype(np.float32)


def _whoosh(duration: float, sr: int) -> np.ndarray:
    n = int(duration * sr)
    rng = np.random.default_rng()
    t = np.linspace(0, duration, n, endpoint=False)
    # Frequency sweep (low->high->low) with noise body
    sweep_freq = 200 + 1500 * np.sin(np.pi * t / duration)
    phase = 2 * np.pi * np.cumsum(sweep_freq) / sr
    tone = np.sin(phase) * 0.4
    body = _butter_bandpass(rng.standard_normal(n).astype(np.float32), 300, 3000, sr) * 0.7
    envelope = np.sin(np.pi * t / duration) ** 1.5  # bell-shaped
    return ((tone + body) * envelope).astype(np.float32)


def _clock_tick(duration: float, sr: int) -> np.ndarray:
    n = int(duration * sr)
    out = np.zeros(n, dtype=np.float32)
    rng = np.random.default_rng()
    period = 1.0  # 1 tick per second
    t_pos = 0.05
    is_tick = True
    while t_pos < duration:
        pos = int(t_pos * sr)
        # tick (high) vs tock (slightly lower)
        freq = 4200 if is_tick else 3100
        length = int(0.025 * sr)
        end = min(pos + length, n)
        tt = np.arange(end - pos) / sr
        click = (rng.standard_normal(end - pos).astype(np.float32) * 0.3 +
                 np.sin(2 * np.pi * freq * tt) * 0.7) * np.exp(-tt * 240)
        out[pos:end] += click * 0.85
        t_pos += period
        is_tick = not is_tick
    return out


def _bird_chirp(duration: float, sr: int) -> np.ndarray:
    n = int(duration * sr)
    out = np.zeros(n, dtype=np.float32)
    rng = np.random.default_rng()
    # Random chirps every 0.4-1.5s
    t_pos = 0.1
    while t_pos < duration:
        pos = int(t_pos * sr)
        chirp_dur = rng.uniform(0.06, 0.18)
        chirp_n = min(int(chirp_dur * sr), n - pos)
        if chirp_n <= 0:
            break
        tt = np.arange(chirp_n) / sr
        f0, f1 = rng.uniform(2000, 3500), rng.uniform(3500, 6500)
        # Frequency-modulated chirp (FM)
        freq = f0 + (f1 - f0) * (tt / chirp_dur)
        phase = 2 * np.pi * np.cumsum(freq) / sr
        chirp = np.sin(phase) * np.exp(-((tt - chirp_dur / 2) ** 2) / (chirp_dur * 0.15) ** 2)
        out[pos:pos + chirp_n] += chirp.astype(np.float32) * 0.6
        # Sometimes a quick repeat
        if rng.random() < 0.4:
            t_pos += chirp_dur + 0.03
        else:
            t_pos += rng.uniform(0.4, 1.4)
    # Tiny ambient hiss
    out += _butter_highpass(rng.standard_normal(n).astype(np.float32), 8000, sr) * 0.02
    return out


def _fire_crackling(duration: float, sr: int) -> np.ndarray:
    n = int(duration * sr)
    rng = np.random.default_rng()
    # Continuous low rumble (the fire's air movement)
    rumble = _butter_lowpass(rng.standard_normal(n).astype(np.float32), 300, sr) * 0.18
    # Sharp pops (crackles) at random intervals
    pops = np.zeros(n, dtype=np.float32)
    n_pops = int(duration * 18)
    for _ in range(n_pops):
        pos = rng.integers(0, n)
        length = int(rng.uniform(0.005, 0.025) * sr)
        end = min(pos + length, n)
        tt = np.arange(end - pos) / sr
        freq = rng.uniform(900, 4000)
        pop = np.sin(2 * np.pi * freq * tt) * np.exp(-tt * 150) * rng.uniform(0.4, 1.0)
        pops[pos:end] += pop.astype(np.float32)
    return rumble + pops * 0.7


def _typing(duration: float, sr: int) -> np.ndarray:
    n = int(duration * sr)
    rng = np.random.default_rng()
    out = np.zeros(n, dtype=np.float32)
    # Random keystrokes ~ 6 keys/sec with small natural variation
    t_pos = 0.05
    while t_pos < duration:
        pos = int(t_pos * sr)
        length = int(0.03 * sr)
        end = min(pos + length, n)
        tt = np.arange(end - pos) / sr
        # Two-component click: high-freq snap + small body
        freq_a = rng.uniform(2500, 4500)
        freq_b = rng.uniform(800, 1500)
        click = (np.sin(2 * np.pi * freq_a * tt) * np.exp(-tt * 250) * 0.6 +
                 np.sin(2 * np.pi * freq_b * tt) * np.exp(-tt * 90) * 0.4 +
                 rng.standard_normal(end - pos).astype(np.float32) * 0.15 * np.exp(-tt * 200))
        out[pos:end] += click.astype(np.float32) * rng.uniform(0.4, 0.85)
        t_pos += rng.uniform(0.10, 0.22)
    return out


def _footsteps(duration: float, sr: int) -> np.ndarray:
    n = int(duration * sr)
    rng = np.random.default_rng()
    # Stereo: alternate L/R channels for walking
    left = np.zeros(n, dtype=np.float32)
    right = np.zeros(n, dtype=np.float32)
    period = 0.55
    t_pos = 0.1
    is_left = True
    while t_pos < duration:
        pos = int(t_pos * sr)
        length = int(0.18 * sr)
        end = min(pos + length, n)
        tt = np.arange(end - pos) / sr
        # Mid-frequency thump with quick decay + scuff component
        thump = np.sin(2 * np.pi * 90 * tt) * np.exp(-tt * 22) * 0.7
        scuff = rng.standard_normal(end - pos).astype(np.float32) * np.exp(-tt * 30) * 0.25
        step = (thump + scuff).astype(np.float32)
        target = left if is_left else right
        target[pos:end] += step
        is_left = not is_left
        t_pos += period * (1 + rng.uniform(-0.08, 0.08))
    # Light low-pass to soften the impact
    left = _butter_lowpass(left, 1800, sr)
    right = _butter_lowpass(right, 1800, sr)
    return _stereo(left, right)


def _car_engine(duration: float, sr: int) -> np.ndarray:
    n = int(duration * sr)
    rng = np.random.default_rng()
    t = np.linspace(0, duration, n, endpoint=False)
    # Idle base tone at ~50-60 Hz with harmonics + slow throttle modulation
    throttle = 0.85 + 0.15 * np.sin(2 * np.pi * 0.4 * t + np.sin(2 * np.pi * 0.13 * t))
    base = np.sin(2 * np.pi * 55 * t * throttle) * 0.5
    h2 = np.sin(2 * np.pi * 110 * t * throttle) * 0.3
    h3 = np.sin(2 * np.pi * 165 * t * throttle) * 0.15
    # Combustion turbulence
    turb = _butter_bandpass(rng.standard_normal(n).astype(np.float32), 80, 600, sr) * 0.35
    return ((base + h2 + h3 + turb) * (0.7 + 0.3 * throttle)).astype(np.float32)


def _door_close(duration: float, sr: int) -> np.ndarray:
    n = int(duration * sr)
    rng = np.random.default_rng()
    # Initial thud (low) + latch click (high) then short reverb tail
    thud_n = int(0.25 * sr)
    thud_t = np.arange(thud_n) / sr
    thud = (np.sin(2 * np.pi * 75 * thud_t) * np.exp(-thud_t * 14) +
            rng.standard_normal(thud_n).astype(np.float32) * np.exp(-thud_t * 30) * 0.4)
    click_pos = int(0.04 * sr)
    click_n = int(0.025 * sr)
    click_t = np.arange(click_n) / sr
    click = np.sin(2 * np.pi * 3500 * click_t) * np.exp(-click_t * 280) * 0.5
    out = np.zeros(n, dtype=np.float32)
    out[:thud_n] += thud.astype(np.float32) * 1.1
    if click_pos + click_n < n:
        out[click_pos:click_pos + click_n] += click.astype(np.float32)
    # Subtle reverb-like tail via low-pass on a delayed copy
    tail = _butter_lowpass(out * 0.35, 800, sr)
    delay = int(0.05 * sr)
    if delay + n - delay > 0:
        out[delay:] += tail[: n - delay] * 0.6
    return out


def _glass_break(duration: float, sr: int) -> np.ndarray:
    n = int(duration * sr)
    rng = np.random.default_rng()
    # Initial impact (broadband transient) + falling shard cascade (bandpassed glittery noise)
    impact_n = int(0.06 * sr)
    impact = rng.standard_normal(impact_n).astype(np.float32)
    impact *= np.exp(-np.linspace(0, 1, impact_n) * 12)
    impact = _butter_bandpass(impact, 2000, 9000, sr)

    cascade_n = n - impact_n
    cascade = rng.standard_normal(cascade_n).astype(np.float32)
    cascade = _butter_highpass(cascade, 3500, sr)
    # Random pings: short high-freq sines randomly placed
    pings = np.zeros(cascade_n, dtype=np.float32)
    n_pings = int((duration - 0.06) * 60)
    for _ in range(n_pings):
        pos = rng.integers(0, cascade_n)
        length = int(rng.uniform(0.01, 0.06) * sr)
        end = min(pos + length, cascade_n)
        tt = np.arange(end - pos) / sr
        freq = rng.uniform(3500, 9000)
        ping = np.sin(2 * np.pi * freq * tt) * np.exp(-tt * 60) * rng.uniform(0.3, 1.0)
        pings[pos:end] += ping.astype(np.float32)
    # Decay envelope on the cascade
    cascade_env = np.exp(-np.linspace(0, 1, cascade_n) * 3)

    out = np.zeros(n, dtype=np.float32)
    out[:impact_n] += impact * 1.4
    out[impact_n:] += (cascade * 0.4 + pings * 0.6) * cascade_env
    return out


_PROCEDURAL_GENERATORS = {
    "rain": (_rain, "Layered ambient hiss + decorrelated stereo droplets, low-pass shaped"),
    "thunder": (_thunder, "Broadband crack + sub-bass rumble with low-pass tail"),
    "wind": (_wind, "Stereo bandpass noise with slow gust modulation"),
    "ocean_waves": (_ocean_waves, "Multi-period swell envelope, foam at wave peaks"),
    "heartbeat": (_heartbeat, "Lub-dub thump pair @ 70 BPM with slight wobble"),
    "explosion": (_explosion, "Bass boom + sub thump + transient mid-high noise"),
    "whoosh": (_whoosh, "Frequency-swept tone over bell-envelope bandpass noise"),
    "clock_tick": (_clock_tick, "Alternating tick/tock with envelope decay"),
    "bird_chirp": (_bird_chirp, "FM-chirped sine bursts with random spacing"),
    "fire_crackling": (_fire_crackling, "Low rumble bed + random short crackle pops"),
    "typing": (_typing, "Two-component click + noise burst, ~6 keys/sec"),
    "footsteps": (_footsteps, "Alternating L/R thump + scuff transient"),
    "car_engine": (_car_engine, "Harmonic stack with throttle modulation + combustion turbulence"),
    "door_close": (_door_close, "Low thud + latch click + subtle reverb tail"),
    "glass_break": (_glass_break, "Broadband impact + high-freq shard cascade with random pings"),
}


# Keyword map for matching arbitrary text prompts to a procedural preset.
_KEYWORD_MAP = [
    (("rain", "raindrop", "drizzle", "shower"), "rain"),
    (("thunder", "lightning", "storm"), "thunder"),
    (("wind", "breeze", "gale", "howl"), "wind"),
    (("ocean", "wave", "sea", "surf", "beach"), "ocean_waves"),
    (("heart", "beat", "pulse"), "heartbeat"),
    (("explos", "blast", "boom", "bomb"), "explosion"),
    (("whoosh", "swoosh", "swipe", "swish"), "whoosh"),
    (("tick", "clock", "watch"), "clock_tick"),
    (("bird", "chirp", "tweet", "sparrow"), "bird_chirp"),
    (("fire", "crackl", "campfire", "flame"), "fire_crackling"),
    (("typ", "keyboard", "key click"), "typing"),
    (("footstep", "walk", "step"), "footsteps"),
    (("engine", "motor", "car ", "truck"), "car_engine"),
    (("door", "slam", "close"), "door_close"),
    (("glass", "shatter", "break", "smash"), "glass_break"),
]


def match_prompt_to_sfx(prompt: str) -> Optional[str]:
    """Return the closest known SFX type for a free-text prompt, or None."""
    p = (prompt or "").lower()
    for keywords, sfx_type in _KEYWORD_MAP:
        for kw in keywords:
            if kw in p:
                return sfx_type
    return None


def generate_sfx_procedural(sfx_type: str, duration: float = 2.0, output_path: str = None) -> str:
    """
    Generate a procedural sound effect using numpy + scipy filters and write it
    to ``output_path``. Returns the file path (kept for backward compat).

    For richer metadata (engine name, algorithm description), call
    :func:`generate_sfx` instead.
    """
    info = generate_sfx(sfx_type=sfx_type, duration=duration, output_path=output_path)
    return info["path"]


def generate_sfx(
    sfx_type: str = "",
    prompt: str = "",
    duration: float = 2.0,
    output_path: Optional[str] = None,
) -> dict:
    """
    Generate a sound effect. If ``sfx_type`` is given, run the matching
    procedural synthesizer. Otherwise, infer one from ``prompt`` via keyword
    matching. Returns a dict::

        {"path": str, "engine": str, "method": str, "sfx_type": str,
         "duration": float, "sample_rate": int, "channels": int, "details": str}
    """
    import soundfile as sf

    ensure_output_dir()
    duration = max(0.2, min(float(duration), 30.0))

    resolved_type = sfx_type.strip().lower()
    if not resolved_type and prompt:
        match = match_prompt_to_sfx(prompt)
        if match:
            resolved_type = match
    if resolved_type not in _PROCEDURAL_GENERATORS:
        # Fallback: noise burst with envelope so it never silently fails.
        resolved_type = "whoosh"

    output_path = output_path or os.path.join(OUTPUT_DIR, f"sfx_{resolved_type}_{uuid4()}.wav")

    fn, details = _PROCEDURAL_GENERATORS[resolved_type]
    audio = fn(duration, _SR)

    # Peak normalize (preserve stereo balance if 2D)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0:
        audio = audio * (0.92 / peak)

    sf.write(output_path, audio.astype(np.float32), _SR)

    return {
        "path": output_path,
        "engine": SFX_ENGINE_VERSION,
        "method": "Procedural synthesis",
        "sfx_type": resolved_type,
        "duration": float(duration),
        "sample_rate": _SR,
        "channels": 2 if audio.ndim == 2 else 1,
        "details": details,
    }


def list_sfx_types() -> list:
    """Names of all directly callable procedural SFX presets."""
    return list(_PROCEDURAL_GENERATORS.keys())


# =============================================
# TTS Personas + audio post-processing
# =============================================

# Each persona maps to: (default_voice, speed_multiplier, pitch_semitones,
# volume_gain, description). The voice can be overridden by the user.
TTS_PERSONAS = {
    "narrator": {
        "label": "Narrator",
        "voice": "Lessac",
        "speed": 0.95,
        "pitch": -1,
        "volume": 1.0,
        "description": "Calm, authoritative voice for documentaries and explainers.",
    },
    "audiobook": {
        "label": "Audiobook",
        "voice": "Amy",
        "speed": 1.0,
        "pitch": 0,
        "volume": 1.0,
        "description": "Warm and expressive, paced for long-form listening.",
    },
    "news_anchor": {
        "label": "News Anchor",
        "voice": "Ryan",
        "speed": 1.05,
        "pitch": 0,
        "volume": 1.05,
        "description": "Clear, professional delivery with broadcast pacing.",
    },
    "storyteller": {
        "label": "Storyteller",
        "voice": "Jenny",
        "speed": 0.9,
        "pitch": 1,
        "volume": 1.0,
        "description": "Dramatic, expressive cadence for fiction or kids' content.",
    },
    "documentary": {
        "label": "Documentary",
        "voice": "Alan",
        "speed": 0.88,
        "pitch": -1,
        "volume": 1.0,
        "description": "Slow, deliberate UK-English voice for serious topics.",
    },
    "excited": {
        "label": "Excited Host",
        "voice": "Kathleen",
        "speed": 1.18,
        "pitch": 2,
        "volume": 1.05,
        "description": "Fast, upbeat energy for adverts and social shorts.",
    },
    "casual": {
        "label": "Casual",
        "voice": "Amy",
        "speed": 1.0,
        "pitch": 0,
        "volume": 1.0,
        "description": "Conversational and approachable, vlog-style.",
    },
    "deep_voice": {
        "label": "Deep Voice",
        "voice": "Northern",
        "speed": 0.92,
        "pitch": -2,
        "volume": 1.05,
        "description": "Deep, gravelly trailer voice. Pairs well with longer text.",
    },
    "british_lady": {
        "label": "British Lady",
        "voice": "Alba",
        "speed": 1.0,
        "pitch": 0,
        "volume": 1.0,
        "description": "Refined UK-English female voice.",
    },
    "robot": {
        "label": "Robot / AI",
        "voice": "Libritts",
        "speed": 1.0,
        "pitch": -3,
        "volume": 1.0,
        "description": "Lower pitch with flat dynamics for AI / sci-fi readouts.",
    },
    "whisper": {
        "label": "Whisper",
        "voice": "Amy",
        "speed": 0.95,
        "pitch": -1,
        "volume": 0.55,
        "description": "Soft, intimate delivery for voiceovers and ASMR.",
    },
}


def list_personas() -> list:
    """Return persona metadata for the UI."""
    return [
        {
            "id": pid,
            "label": p["label"],
            "voice": p["voice"],
            "speed": p["speed"],
            "pitch": p["pitch"],
            "volume": p["volume"],
            "description": p["description"],
        }
        for pid, p in TTS_PERSONAS.items()
    ]


def list_piper_voices() -> list:
    """Return all available Piper voices with metadata."""
    try:
        from mpv2.classes.PiperTts import PIPER_VOICES
    except Exception:
        return []
    return [
        {
            "id": name,
            "model": info.get("model"),
            "quality": info.get("quality"),
            "gender": info.get("gender"),
            # Heuristic: model id starts with en_GB -> UK, en_US -> US.
            "accent": (info.get("model", "")[:5] if info.get("model") else "").replace("_", "-"),
        }
        for name, info in PIPER_VOICES.items()
    ]


def _ffmpeg_run(cmd: list) -> None:
    """Run ffmpeg and raise with stderr on failure."""
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr.strip().splitlines()[-1] if proc.stderr else 'unknown'}")


def _atempo_chain(factor: float) -> str:
    """ffmpeg's atempo only accepts 0.5..2.0 — chain filters for larger ranges."""
    f = max(0.25, min(4.0, factor))
    if 0.5 <= f <= 2.0:
        return f"atempo={f:.4f}"
    # Decompose: 0.4 = 0.5 * 0.8 ; 2.5 = 2.0 * 1.25
    if f < 0.5:
        return f"atempo=0.5,atempo={f / 0.5:.4f}"
    return f"atempo=2.0,atempo={f / 2.0:.4f}"


def apply_audio_effects(
    input_path: str,
    output_path: Optional[str] = None,
    speed: float = 1.0,
    pitch_semitones: float = 0.0,
    volume: float = 1.0,
    output_format: str = "wav",
) -> str:
    """
    Apply speed, pitch, and volume adjustments to an audio file via ffmpeg.

    speed: 0.5..2.0 typical (clamped to 0.25..4.0)
    pitch_semitones: -12..+12 typical (uses asetrate trick + tempo correction)
    volume: linear gain (1.0 = unchanged)
    output_format: 'wav' or 'mp3'

    Returns the output path.
    """
    ensure_output_dir()
    if output_format not in ("wav", "mp3"):
        output_format = "wav"
    output_path = output_path or os.path.join(OUTPUT_DIR, f"tts_proc_{uuid4()}.{output_format}")
    speed = max(0.25, min(4.0, float(speed)))
    pitch = max(-12.0, min(12.0, float(pitch_semitones)))
    volume = max(0.0, min(3.0, float(volume)))

    # If everything is identity AND output format matches, copy and return.
    if abs(speed - 1.0) < 0.005 and abs(pitch) < 0.05 and abs(volume - 1.0) < 0.005:
        if input_path.lower().endswith(f".{output_format}"):
            try:
                import shutil
                if os.path.abspath(input_path) != os.path.abspath(output_path):
                    shutil.copyfile(input_path, output_path)
                else:
                    output_path = input_path
                return output_path
            except OSError:
                pass

    # Probe sample rate so the asetrate-based pitch shift can preserve it.
    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=sample_rate",
             "-of", "csv=p=0", input_path],
            capture_output=True, text=True,
        )
        sr = int(probe.stdout.strip() or "44100")
    except Exception:
        sr = 44100

    filters = []
    pitch_factor = 2 ** (pitch / 12.0) if pitch else 1.0
    if abs(pitch) >= 0.05:
        # Pitch shift via asetrate (changes both pitch and tempo) then compensate tempo.
        filters.append(f"asetrate={int(sr * pitch_factor)}")
        # After asetrate, audio sample rate is non-standard — set back.
        filters.append(f"aresample={sr}")
        # Compensating tempo so pitch shift doesn't change speed: divide by pitch_factor
        comp = 1.0 / pitch_factor
        if speed != 1.0:
            comp *= speed
        filters.append(_atempo_chain(comp))
    elif abs(speed - 1.0) >= 0.005:
        filters.append(_atempo_chain(speed))
    if abs(volume - 1.0) >= 0.005:
        filters.append(f"volume={volume:.3f}")

    af = ",".join(filters) if filters else "anull"

    cmd = ["ffmpeg", "-y", "-i", input_path, "-af", af]
    if output_format == "mp3":
        cmd += ["-c:a", "libmp3lame", "-b:a", "192k", output_path]
    else:
        cmd += ["-c:a", "pcm_s16le", output_path]
    _ffmpeg_run(cmd)
    return output_path


def generate_sfx_ai(prompt: str, duration: float = 3.0, output_path: str = None) -> str:
    """
    Generate sound effects from text description using AudioLDM.
    Requires: pip install audioldm

    Falls back to procedural if AudioLDM not available.
    """
    ensure_output_dir()
    output_path = output_path or os.path.join(OUTPUT_DIR, f"sfx_ai_{uuid4()}.wav")

    try:
        from audioldm import build_model, text_to_audio
        model = build_model()
        waveform = text_to_audio(model, prompt, duration=duration)
        import soundfile as sf
        sf.write(output_path, waveform.squeeze().numpy(), 16000)
        return output_path
    except ImportError:
        pass

    # Fallback: match prompt to closest procedural SFX
    prompt_lower = prompt.lower()
    for sfx_name in SFX_LIBRARY:
        if sfx_name.replace("_", " ") in prompt_lower or sfx_name in prompt_lower:
            return generate_sfx_procedural(sfx_name, duration, output_path)

    return generate_sfx_procedural("whoosh", duration, output_path)


# =============================================
# Audio Mixing & Effects
# =============================================

def normalize_audio(audio_path: str, output_path: str = None, target_db: float = -3.0) -> str:
    """Normalize audio to target dB level."""
    import soundfile as sf
    ensure_output_dir()
    output_path = output_path or os.path.join(OUTPUT_DIR, f"normalized_{uuid4()}.wav")

    audio, sr = sf.read(audio_path)
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        target_amp = 10 ** (target_db / 20)
        audio = audio * (target_amp / max_val)

    sf.write(output_path, audio, sr)
    return output_path


def mix_audio_tracks(tracks: list, output_path: str = None) -> str:
    """
    Mix multiple audio tracks together.

    Args:
        tracks: List of dicts: {"path": str, "volume": float, "start_time": float}
        output_path: Output file path

    Returns:
        Path to mixed audio file
    """
    import soundfile as sf
    ensure_output_dir()
    output_path = output_path or os.path.join(OUTPUT_DIR, f"mixed_{uuid4()}.wav")

    # Find total duration and sample rate
    max_duration = 0
    sample_rate = 22050

    for track in tracks:
        audio, sr = sf.read(track["path"])
        sample_rate = sr
        start = track.get("start_time", 0)
        duration = len(audio) / sr + start
        max_duration = max(max_duration, duration)

    # Create output buffer
    total_samples = int(max_duration * sample_rate)
    mixed = np.zeros(total_samples, dtype=np.float32)

    for track in tracks:
        audio, sr = sf.read(track["path"])
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        volume = track.get("volume", 1.0)
        start_sample = int(track.get("start_time", 0) * sample_rate)

        end_sample = min(start_sample + len(audio), total_samples)
        audio_len = end_sample - start_sample
        mixed[start_sample:end_sample] += audio[:audio_len] * volume

    # Normalize to prevent clipping
    max_val = np.max(np.abs(mixed))
    if max_val > 0.99:
        mixed = mixed * 0.95 / max_val

    sf.write(output_path, mixed, sample_rate)
    return output_path


def add_fade(audio_path: str, fade_in: float = 0.5, fade_out: float = 0.5,
             output_path: str = None) -> str:
    """Add fade in/out to audio."""
    import soundfile as sf
    ensure_output_dir()
    output_path = output_path or os.path.join(OUTPUT_DIR, f"faded_{uuid4()}.wav")

    audio, sr = sf.read(audio_path)
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    # Fade in
    fade_in_samples = int(fade_in * sr)
    if fade_in_samples > 0 and fade_in_samples < len(audio):
        fade_in_curve = np.linspace(0, 1, fade_in_samples)
        audio[:fade_in_samples] *= fade_in_curve

    # Fade out
    fade_out_samples = int(fade_out * sr)
    if fade_out_samples > 0 and fade_out_samples < len(audio):
        fade_out_curve = np.linspace(1, 0, fade_out_samples)
        audio[-fade_out_samples:] *= fade_out_curve

    sf.write(output_path, audio, sr)
    return output_path


def change_speed(audio_path: str, speed: float = 1.0, output_path: str = None) -> str:
    """Change audio playback speed without changing pitch."""
    ensure_output_dir()
    output_path = output_path or os.path.join(OUTPUT_DIR, f"speed_{uuid4()}.wav")

    try:
        subprocess.run([
            "ffmpeg", "-y", "-i", audio_path,
            "-filter:a", f"atempo={speed}",
            output_path,
        ], capture_output=True, check=True)
        return output_path
    except Exception as e:
        raise RuntimeError(f"Speed change failed: {e}")


def extract_audio(video_path: str, output_path: str = None) -> str:
    """Extract audio track from video."""
    ensure_output_dir()
    output_path = output_path or os.path.join(OUTPUT_DIR, f"extracted_{uuid4()}.wav")

    try:
        subprocess.run([
            "ffmpeg", "-y", "-i", video_path,
            "-vn", "-acodec", "pcm_s16le",
            output_path,
        ], capture_output=True, check=True)
        if os.path.exists(output_path) and os.path.getsize(output_path) > 100:
            return output_path
        return None
    except Exception:
        return None


# =============================================
# Lip Sync Preparation
# =============================================

def prepare_lip_sync_data(audio_path: str) -> dict:
    """
    Extract phoneme timing from audio for lip sync.
    Returns timing data that can be used to drive character mouth movements.
    """
    try:
        import soundfile as sf
        audio, sr = sf.read(audio_path)
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        # Simple energy-based mouth open/close detection
        frame_size = int(sr * 0.033)  # 33ms frames (~30fps)
        hop = frame_size
        frames = []

        for i in range(0, len(audio) - frame_size, hop):
            frame = audio[i:i + frame_size]
            energy = np.sqrt(np.mean(frame ** 2))
            frames.append({
                "time": i / sr,
                "energy": float(energy),
                "mouth_open": float(min(energy * 10, 1.0)),  # 0-1 normalized
            })

        return {
            "frames": frames,
            "duration": len(audio) / sr,
            "fps": sr / hop,
            "total_frames": len(frames),
        }

    except Exception as e:
        return {"error": str(e)}


def list_sfx_library() -> dict:
    """List all available sound effects."""
    return dict(SFX_LIBRARY)
