"""Beat and BPM detection for timeline sync.

Primary path uses librosa's onset-aware beat tracker. If librosa isn't
available we fall back to an onset-envelope + autocorrelation implementation
on top of scipy + soundfile, so callers get *some* answer even on a minimal
install.

The output shape is stable across both paths:

    {
        "bpm": 118.3,
        "beats_s": [0.42, 0.93, 1.44, ...],
        "downbeats_s": [0.42, 2.46, 4.50, ...],  # every 4th beat by default
        "confidence": 0.78,                       # 0-1, higher = clearer pulse
        "source": "librosa" | "fallback",
    }

The Timeline UI feeds ``beats_s`` to its snap-to-beat cut tool; the rest is
there for the user-facing BPM readout.
"""

from __future__ import annotations

import os
from typing import Optional


def _load_mono_audio(path: str, target_sr: int = 22050) -> tuple[Optional["np.ndarray"], int]:  # type: ignore[name-defined]
    """Read ``path`` as mono float32 PCM. Returns (samples, sr) or (None, 0)."""
    try:
        import numpy as np
        import soundfile as sf
    except Exception:
        return None, 0
    try:
        data, sr = sf.read(path, always_2d=False)
    except Exception:
        return None, 0
    if data is None or len(data) == 0:
        return None, 0
    arr = data.astype("float32", copy=False)
    if arr.ndim == 2:
        arr = arr.mean(axis=1).astype("float32")
    if sr != target_sr and arr.size:
        # Simple linear resample; good enough for beat tracking.
        new_len = int(round(arr.size * target_sr / sr))
        if new_len > 1:
            x_old = np.linspace(0.0, 1.0, arr.size, endpoint=False)
            x_new = np.linspace(0.0, 1.0, new_len, endpoint=False)
            arr = np.interp(x_new, x_old, arr).astype("float32")
            sr = target_sr
    return arr, sr


def _detect_with_librosa(path: str) -> Optional[dict]:
    """Try librosa first - it's the gold standard here."""
    try:
        import librosa  # type: ignore
        import numpy as np
    except Exception:
        return None
    try:
        y, sr = librosa.load(path, sr=22050, mono=True)
        if y.size < sr // 2:
            return None  # too short to track reliably
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units="frames")
        # librosa may return tempo as a 0-d array in newer versions
        tempo = float(np.asarray(tempo).reshape(-1)[0]) if hasattr(tempo, "shape") else float(tempo)
        beats_s = librosa.frames_to_time(beat_frames, sr=sr).tolist()
        # Confidence = ratio of beat-synchronous onset energy to total onset energy
        env = librosa.onset.onset_strength(y=y, sr=sr)
        total = float(np.sum(env)) + 1e-9
        hit = float(np.sum(env[beat_frames.astype(int)])) if len(beat_frames) else 0.0
        confidence = max(0.0, min(1.0, hit / total * len(env) / max(1, len(beat_frames))))
        return {
            "bpm": round(tempo, 2),
            "beats_s": [round(t, 4) for t in beats_s],
            "downbeats_s": [round(t, 4) for t in beats_s[::4]],
            "confidence": round(confidence, 3),
            "source": "librosa",
        }
    except Exception:
        return None


def _onset_envelope(y: "np.ndarray", sr: int, hop: int = 512) -> "np.ndarray":  # type: ignore[name-defined]
    """Minimal spectral-flux onset envelope. Pure numpy."""
    import numpy as np
    win = 1024
    n_frames = max(0, 1 + (len(y) - win) // hop)
    if n_frames < 2:
        return np.zeros(1, dtype="float32")
    # Hann window for each frame
    hann = np.hanning(win).astype("float32")
    flux = np.zeros(n_frames, dtype="float32")
    prev_mag = None
    for i in range(n_frames):
        frame = y[i * hop:i * hop + win] * hann
        spec = np.abs(np.fft.rfft(frame))
        if prev_mag is not None:
            diff = spec - prev_mag
            diff[diff < 0] = 0  # positive flux only
            flux[i] = diff.sum()
        prev_mag = spec
    # Remove slow trend so autocorrelation finds the pulse
    mean = np.mean(flux)
    flux = flux - mean
    flux[flux < 0] = 0
    return flux


def _estimate_bpm_from_envelope(env: "np.ndarray", sr: int, hop: int = 512,  # type: ignore[name-defined]
                                 bpm_min: float = 60.0, bpm_max: float = 180.0) -> float:
    """Autocorrelate the onset envelope and pick the BPM-shaped peak."""
    import numpy as np
    n = len(env)
    if n < 8:
        return 0.0
    ac = np.correlate(env, env, mode="full")[n - 1:]
    # Hunt in the lag window that corresponds to the BPM range
    frame_rate = sr / hop  # frames per second
    lag_min = max(1, int(round(frame_rate * 60.0 / bpm_max)))
    lag_max = min(n - 1, int(round(frame_rate * 60.0 / bpm_min)))
    if lag_max <= lag_min + 1:
        return 0.0
    segment = ac[lag_min:lag_max]
    peak = int(np.argmax(segment)) + lag_min
    bpm = 60.0 * frame_rate / max(1, peak)
    return float(bpm)


def _detect_with_fallback(path: str) -> dict:
    """Numpy/scipy-only path. Lower quality than librosa but robust."""
    import numpy as np
    y, sr = _load_mono_audio(path)
    if y is None or y.size < sr:
        return {
            "bpm": 0.0, "beats_s": [], "downbeats_s": [],
            "confidence": 0.0, "source": "fallback",
        }
    hop = 512
    env = _onset_envelope(y, sr, hop=hop)
    bpm = _estimate_bpm_from_envelope(env, sr, hop=hop)
    if bpm < 20.0:
        return {
            "bpm": 0.0, "beats_s": [], "downbeats_s": [],
            "confidence": 0.1, "source": "fallback",
        }
    # Synthesize beat times from BPM + first onset peak as phase anchor.
    frame_rate = sr / hop
    first_peak = int(np.argmax(env[: int(frame_rate * 3)]))  # within first 3s
    anchor_s = first_peak / frame_rate
    period = 60.0 / bpm
    duration = y.size / sr
    beats = []
    t = anchor_s
    # Step back to catch any leading beat before the anchor
    while t - period > 0:
        t -= period
    while t < duration:
        beats.append(round(t, 4))
        t += period
    # Confidence: ratio of the chosen autocorr peak to the envelope mean
    confidence = min(1.0, float(np.max(env) / (np.mean(env) + 1e-6)) / 30.0)
    return {
        "bpm": round(bpm, 2),
        "beats_s": beats,
        "downbeats_s": beats[::4],
        "confidence": round(confidence, 3),
        "source": "fallback",
    }


def detect_beats(audio_path: str) -> dict:
    """Return beat positions and BPM for ``audio_path``.

    Never raises for a readable file - callers always get the stable schema,
    with empty beats + bpm=0 on failure. Missing files DO raise so the API
    can 404/422 cleanly.
    """
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio not found: {audio_path}")
    r = _detect_with_librosa(audio_path)
    if r is not None:
        return r
    return _detect_with_fallback(audio_path)


def snap_to_beats(times_s: list[float], beats_s: list[float],
                  tolerance_s: float = 0.15) -> list[float]:
    """Nudge each cut in ``times_s`` to the nearest beat within ``tolerance_s``.

    Cuts outside the tolerance are left untouched. Pure function - the
    Timeline UI's "snap selected cuts" button is this call on the client's
    cut list after loading ``beats_s`` for the current music track.
    """
    if not beats_s:
        return list(times_s)
    out = []
    for t in times_s:
        # Binary search the nearest beat
        lo, hi = 0, len(beats_s) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if beats_s[mid] < t:
                lo = mid + 1
            else:
                hi = mid
        candidates = [beats_s[lo]]
        if lo > 0:
            candidates.append(beats_s[lo - 1])
        if lo + 1 < len(beats_s):
            candidates.append(beats_s[lo + 1])
        nearest = min(candidates, key=lambda b: abs(b - t))
        if abs(nearest - t) <= tolerance_s:
            out.append(round(nearest, 4))
        else:
            out.append(round(t, 4))
    return out
