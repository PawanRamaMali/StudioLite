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


def generate_sfx_procedural(sfx_type: str, duration: float = 2.0, output_path: str = None) -> str:
    """
    Generate basic sound effects procedurally using numpy.
    For better quality, use AudioLDM model.
    """
    import soundfile as sf
    ensure_output_dir()
    output_path = output_path or os.path.join(OUTPUT_DIR, f"sfx_{sfx_type}_{uuid4()}.wav")

    sample_rate = 22050
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio = np.zeros_like(t)

    if sfx_type == "whoosh":
        # Rising frequency sweep
        freq = np.linspace(100, 2000, len(t))
        audio = 0.5 * np.sin(2 * np.pi * freq * t / sample_rate) * np.exp(-t / duration * 3)
        # Add noise
        audio += 0.2 * np.random.randn(len(t)) * np.exp(-t / duration * 2)

    elif sfx_type == "explosion":
        audio = np.random.randn(len(t))
        envelope = np.exp(-t / (duration * 0.15))
        audio = audio * envelope
        # Low-pass effect via simple averaging
        kernel_size = 50
        audio = np.convolve(audio, np.ones(kernel_size) / kernel_size, mode='same')

    elif sfx_type == "rain":
        audio = 0.3 * np.random.randn(len(t))
        # Add occasional droplet hits
        for _ in range(int(duration * 20)):
            pos = np.random.randint(0, len(t))
            drop_len = min(200, len(t) - pos)
            drop_t = np.arange(drop_len) / sample_rate
            audio[pos:pos + drop_len] += 0.5 * np.sin(2 * np.pi * 4000 * drop_t) * np.exp(-drop_t * 40)

    elif sfx_type == "heartbeat":
        for beat in range(int(duration * 1.2)):
            beat_pos = int(beat * sample_rate / 1.2)
            if beat_pos < len(t):
                beat_len = min(int(0.15 * sample_rate), len(t) - beat_pos)
                beat_t = np.arange(beat_len) / sample_rate
                audio[beat_pos:beat_pos + beat_len] += np.sin(2 * np.pi * 60 * beat_t) * np.exp(-beat_t * 20)
                # Second thump
                offset = int(0.2 * sample_rate)
                if beat_pos + offset + beat_len <= len(t):
                    audio[beat_pos + offset:beat_pos + offset + beat_len] += (
                        0.7 * np.sin(2 * np.pi * 50 * beat_t) * np.exp(-beat_t * 25)
                    )

    elif sfx_type == "wind":
        audio = 0.4 * np.random.randn(len(t))
        # Slow modulation
        modulation = 0.5 + 0.5 * np.sin(2 * np.pi * 0.3 * t)
        audio = audio * modulation
        # Smooth with convolution
        kernel = np.ones(200) / 200
        audio = np.convolve(audio, kernel, mode='same')

    elif sfx_type == "thunder":
        audio = np.random.randn(len(t))
        envelope = np.exp(-t / (duration * 0.4))
        # Rumble
        rumble = 0.5 * np.sin(2 * np.pi * 30 * t + np.random.randn(len(t)) * 0.5)
        audio = (audio * 0.5 + rumble) * envelope

    elif sfx_type == "clock_tick":
        for tick in range(int(duration * 2)):
            pos = int(tick * sample_rate / 2)
            if pos < len(t):
                tick_len = min(int(0.02 * sample_rate), len(t) - pos)
                tick_t = np.arange(tick_len) / sample_rate
                audio[pos:pos + tick_len] += np.sin(2 * np.pi * 3000 * tick_t) * np.exp(-tick_t * 200)

    elif sfx_type == "ocean_waves":
        # Multiple wave layers
        wave1 = np.sin(2 * np.pi * 0.15 * t)  # Slow wave
        wave2 = np.sin(2 * np.pi * 0.25 * t + 1.5)  # Offset wave
        envelope = 0.5 + 0.5 * (wave1 + 0.5 * wave2) / 1.5
        audio = 0.3 * np.random.randn(len(t)) * envelope
        kernel = np.ones(300) / 300
        audio = np.convolve(audio, kernel, mode='same')

    else:
        # Generic white noise with envelope
        audio = 0.3 * np.random.randn(len(t)) * np.exp(-t / duration)

    # Normalize
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio * 0.9 / max_val

    sf.write(output_path, audio.astype(np.float32), sample_rate)
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
