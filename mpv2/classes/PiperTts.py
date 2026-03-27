"""
Piper TTS - High-quality offline neural text-to-speech.

Piper uses ONNX models for fast, natural-sounding speech synthesis.
Models are downloaded automatically from Hugging Face on first use.
"""
import os
import subprocess
import wave
import numpy as np
import soundfile as sf
from pathlib import Path

from ..config import ROOT_DIR

PIPER_SAMPLE_RATE = 22050  # Piper outputs 22.05kHz audio

# Voice models available for download
# Format: "display_name": {"model": "model_name", "quality": "low/medium/high"}
PIPER_VOICES = {
    # US English voices
    "Amy": {"model": "en_US-amy-medium", "quality": "medium", "gender": "female"},
    "Ryan": {"model": "en_US-ryan-high", "quality": "high", "gender": "male"},
    "Lessac": {"model": "en_US-lessac-high", "quality": "high", "gender": "male"},
    "Libritts": {"model": "en_US-libritts_r-medium", "quality": "medium", "gender": "neutral"},
    "Arctic": {"model": "en_US-arctic-medium", "quality": "medium", "gender": "male"},
    "Danny": {"model": "en_US-danny-low", "quality": "low", "gender": "male"},
    "Kathleen": {"model": "en_US-kathleen-low", "quality": "low", "gender": "female"},
    # UK English voices
    "Alan": {"model": "en_GB-alan-medium", "quality": "medium", "gender": "male"},
    "Alba": {"model": "en_GB-alba-medium", "quality": "medium", "gender": "female"},
    "Aru": {"model": "en_GB-aru-medium", "quality": "medium", "gender": "male"},
    "Jenny": {"model": "en_GB-jenny_dioco-medium", "quality": "medium", "gender": "female"},
    "Northern": {"model": "en_GB-northern_english_male-medium", "quality": "medium", "gender": "male"},
    "Semaine": {"model": "en_GB-semaine-medium", "quality": "medium", "gender": "female"},
    "Southern": {"model": "en_GB-southern_english_female-low", "quality": "low", "gender": "female"},
    "Vctk": {"model": "en_GB-vctk-medium", "quality": "medium", "gender": "multi"},
}

# Default voice if not specified
DEFAULT_VOICE = "Amy"


def get_piper_models_dir():
    """Get directory for Piper voice models."""
    models_dir = Path(ROOT_DIR) / "piper_models"
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


def get_piper_executable():
    """Get path to piper executable, downloading if needed."""
    # Check if piper is in PATH
    try:
        result = subprocess.run(["piper", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            return "piper"
    except FileNotFoundError:
        pass

    # Check for local piper installation
    local_piper = Path(ROOT_DIR) / "piper" / "piper"
    if local_piper.exists():
        return str(local_piper)

    # Try pip-installed piper-tts
    try:
        import piper
        return None  # Will use Python API instead
    except ImportError:
        pass

    return None


def download_piper_model(voice_name):
    """
    Download a Piper voice model from Hugging Face.

    Returns path to the .onnx model file.
    """
    from huggingface_hub import hf_hub_download

    if voice_name not in PIPER_VOICES:
        raise ValueError(f"Unknown voice: {voice_name}. Available: {list(PIPER_VOICES.keys())}")

    voice_info = PIPER_VOICES[voice_name]
    model_name = voice_info["model"]
    quality = voice_info["quality"]

    models_dir = get_piper_models_dir()

    # Piper models are on rhasspy/piper-voices repo
    repo_id = "rhasspy/piper-voices"

    # Model files follow pattern: voices/{lang}/{model_name}.onnx
    lang_code = model_name.split("-")[0] + "_" + model_name.split("-")[1]  # e.g., en_US
    model_file = f"{model_name}.onnx"
    config_file = f"{model_name}.onnx.json"

    # Download model and config
    model_path = models_dir / model_file
    config_path = models_dir / config_file

    if not model_path.exists():
        print(f"Downloading Piper voice model: {voice_name} ({model_name})...")
        try:
            # Download from HuggingFace
            downloaded = hf_hub_download(
                repo_id=repo_id,
                filename=f"{lang_code}/{quality}/{model_file}",
                local_dir=str(models_dir),
                local_dir_use_symlinks=False,
            )
            # Move to models dir root
            if Path(downloaded).parent != models_dir:
                import shutil
                shutil.move(downloaded, model_path)
        except Exception as e:
            print(f"Failed to download model: {e}")
            raise

    if not config_path.exists():
        try:
            downloaded = hf_hub_download(
                repo_id=repo_id,
                filename=f"{lang_code}/{quality}/{config_file}",
                local_dir=str(models_dir),
                local_dir_use_symlinks=False,
            )
            if Path(downloaded).parent != models_dir:
                import shutil
                shutil.move(downloaded, config_path)
        except Exception as e:
            print(f"Warning: Could not download config file: {e}")

    return str(model_path)


class PiperTTS:
    """
    High-quality TTS using Piper (ONNX neural voices).

    Features:
    - Natural sounding speech (neural network based)
    - Fast inference (ONNX runtime)
    - Multiple voice options
    - Offline operation (models downloaded once)
    - 22.05kHz output sample rate
    """

    def __init__(self, voice_name=None):
        """
        Initialize PiperTTS.

        Args:
            voice_name: Voice to use (from PIPER_VOICES). Defaults to config setting.
        """
        from ..config import get_tts_voice

        if voice_name is None:
            voice_name = get_tts_voice()

        # Map KittenTTS voices to Piper equivalents
        kitten_to_piper = {
            "Jasper": "Ryan",
            "Luna": "Amy",
            "Marcus": "Lessac",
            "Elena": "Jenny",
            "Thomas": "Alan",
            "Sofia": "Alba",
            "Alex": "Arctic",
            "Emma": "Kathleen",
        }

        if voice_name in kitten_to_piper:
            voice_name = kitten_to_piper[voice_name]

        if voice_name not in PIPER_VOICES:
            voice_name = DEFAULT_VOICE

        self._voice_name = voice_name
        self._model_path = None
        self._piper_voice = None
        self._sample_rate = PIPER_SAMPLE_RATE

    def _ensure_model(self):
        """Ensure voice model is downloaded and loaded."""
        if self._model_path is None:
            self._model_path = download_piper_model(self._voice_name)

    def _synthesize_with_cli(self, text, output_path):
        """Synthesize using piper CLI."""
        piper_exe = get_piper_executable()
        if piper_exe is None:
            raise RuntimeError("Piper executable not found. Install with: pip install piper-tts")

        self._ensure_model()

        cmd = [
            piper_exe,
            "--model", self._model_path,
            "--output_file", output_path,
        ]

        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        stdout, stderr = process.communicate(input=text)

        if process.returncode != 0:
            raise RuntimeError(f"Piper synthesis failed: {stderr}")

        return output_path

    def _synthesize_with_python(self, text, output_path):
        """Synthesize using piper-tts Python package."""
        try:
            from piper import PiperVoice
        except ImportError:
            raise RuntimeError("piper-tts not installed. Run: pip install piper-tts")

        self._ensure_model()

        if self._piper_voice is None:
            self._piper_voice = PiperVoice.load(self._model_path)

        # Generate audio
        audio_data = []
        for audio_bytes in self._piper_voice.synthesize_stream_raw(text):
            audio_data.append(np.frombuffer(audio_bytes, dtype=np.int16))

        if not audio_data:
            raise RuntimeError("No audio generated")

        # Concatenate and convert to float
        audio = np.concatenate(audio_data).astype(np.float32) / 32768.0

        # Write to file
        sf.write(output_path, audio, self._sample_rate)

        return output_path

    def synthesize(self, text, output_path):
        """
        Synthesize text to speech.

        Args:
            text: Text to synthesize
            output_path: Path to write WAV file

        Returns:
            output_path on success
        """
        if not text or not text.strip():
            text = "Content generated."

        # Clean text for better synthesis
        text = text.strip()

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Try Python API first (faster), fall back to CLI
        try:
            return self._synthesize_with_python(text, output_path)
        except (ImportError, RuntimeError) as e:
            print(f"Python API failed, trying CLI: {e}")

        try:
            return self._synthesize_with_cli(text, output_path)
        except Exception as e:
            raise RuntimeError(f"Piper TTS synthesis failed: {e}")

    @property
    def sample_rate(self):
        """Return the output sample rate."""
        return self._sample_rate

    @staticmethod
    def list_voices():
        """List available voice names."""
        return list(PIPER_VOICES.keys())

    @staticmethod
    def get_voice_info(voice_name):
        """Get info about a voice."""
        return PIPER_VOICES.get(voice_name)
