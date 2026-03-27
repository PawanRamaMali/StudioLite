"""
AudioMixer - Mix narration with background music.

Features:
- Mix voice narration with background music
- Adjustable music volume
- Auto-ducking: reduce music volume when speech plays
- Loop music to match narration duration
"""
import os
import numpy as np
import soundfile as sf
from pathlib import Path

from .config import ROOT_DIR


def get_music_dir():
    """Get directory for background music files."""
    music_dir = Path(ROOT_DIR) / "music"
    music_dir.mkdir(parents=True, exist_ok=True)
    return music_dir


def list_background_music():
    """List available background music files."""
    music_dir = get_music_dir()
    extensions = {".mp3", ".wav", ".ogg", ".flac", ".m4a"}
    files = []
    for f in music_dir.iterdir():
        if f.suffix.lower() in extensions:
            files.append(f.name)
    return sorted(files)


class AudioMixer:
    """
    Mix narration audio with background music.

    Features:
    - Volume control for music
    - Auto-ducking (reduce music when speech is present)
    - Loop music to match narration length
    """

    def __init__(self, music_volume=0.15, duck_enabled=True, duck_factor=0.3):
        """
        Initialize AudioMixer.

        Args:
            music_volume: Base volume for background music (0.0 - 1.0)
            duck_enabled: Whether to reduce music volume during speech
            duck_factor: How much to reduce music during speech (0 = silence)
        """
        self.music_volume = music_volume
        self.duck_enabled = duck_enabled
        self.duck_factor = duck_factor

    def _load_audio(self, audio_path):
        """Load audio file and return data + sample rate."""
        data, sr = sf.read(audio_path)
        # Convert to mono if stereo
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)
        return data, sr

    def _resample(self, audio, from_sr, to_sr):
        """Resample audio to target sample rate."""
        if from_sr == to_sr:
            return audio
        from scipy import signal
        num_samples = int(len(audio) * to_sr / from_sr)
        return signal.resample(audio, num_samples)

    def _loop_to_length(self, audio, target_length):
        """Loop audio to match target length."""
        if len(audio) >= target_length:
            return audio[:target_length]

        # Repeat audio to fill target length
        repeats = int(np.ceil(target_length / len(audio)))
        looped = np.tile(audio, repeats)
        return looped[:target_length]

    def _detect_speech(self, narration, threshold=0.02, window_size=4410):
        """
        Detect speech regions in narration.

        Returns array of same length where 1 = speech, 0 = silence.
        """
        # Use RMS energy to detect speech
        speech_mask = np.zeros(len(narration))

        for i in range(0, len(narration), window_size):
            end = min(i + window_size, len(narration))
            segment = narration[i:end]
            rms = np.sqrt(np.mean(segment ** 2))

            if rms > threshold:
                speech_mask[i:end] = 1

        # Smooth the mask
        from scipy.ndimage import uniform_filter1d
        speech_mask = uniform_filter1d(speech_mask, size=window_size * 2)
        speech_mask = (speech_mask > 0.3).astype(float)

        return speech_mask

    def mix(self, narration_path, music_path, output_path):
        """
        Mix narration with background music.

        Args:
            narration_path: Path to narration audio (WAV)
            music_path: Path to background music
            output_path: Path to write mixed audio

        Returns:
            output_path on success
        """
        # Load narration
        narration, narr_sr = self._load_audio(narration_path)

        # Load and prepare music
        music, music_sr = self._load_audio(music_path)

        # Resample music to match narration sample rate
        if music_sr != narr_sr:
            music = self._resample(music, music_sr, narr_sr)

        # Loop music to match narration length
        music = self._loop_to_length(music, len(narration))

        # Apply base volume to music
        music = music * self.music_volume

        # Apply ducking if enabled
        if self.duck_enabled:
            speech_mask = self._detect_speech(narration)
            # Where speech is detected, reduce music volume
            duck_amount = 1.0 - (speech_mask * (1.0 - self.duck_factor))
            music = music * duck_amount

        # Mix narration and music
        mixed = narration + music

        # Normalize to prevent clipping
        max_val = np.max(np.abs(mixed))
        if max_val > 0.99:
            mixed = mixed * 0.95 / max_val

        # Write output
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sf.write(output_path, mixed, narr_sr)

        return output_path


def mix_audio_simple(narration_path, music_path, output_path, music_volume=0.15):
    """
    Simple function to mix narration with background music.

    Args:
        narration_path: Path to narration audio
        music_path: Path to background music
        output_path: Path to write mixed audio
        music_volume: Volume for background music (0.0 - 1.0)

    Returns:
        output_path on success
    """
    mixer = AudioMixer(music_volume=music_volume, duck_enabled=True)
    return mixer.mix(narration_path, music_path, output_path)
