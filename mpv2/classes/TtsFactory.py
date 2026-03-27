"""
TTS Factory - Creates the appropriate TTS instance based on configuration.

Supports:
- piper: High-quality neural TTS (default)
- kitten: KittenTTS (lightweight, faster)
"""
from ..config import get_tts_engine


def get_tts_instance(engine=None):
    """
    Get a TTS instance based on the configured or specified engine.

    Args:
        engine: Optional override. One of "piper" or "kitten".
                If None, uses config setting.

    Returns:
        TTS instance with synthesize(text, output_path) method
    """
    if engine is None:
        engine = get_tts_engine()

    engine = engine.lower() if engine else "piper"

    if engine == "piper":
        try:
            from .PiperTts import PiperTTS
            return PiperTTS()
        except ImportError as e:
            print(f"Failed to load PiperTTS: {e}")
            print("Falling back to KittenTTS...")
            engine = "kitten"

    if engine == "kitten":
        try:
            from .Tts import TTS
            return TTS()
        except ImportError as e:
            raise RuntimeError(f"No TTS engine available: {e}")

    # Default fallback
    try:
        from .Tts import TTS
        return TTS()
    except ImportError:
        pass

    try:
        from .PiperTts import PiperTTS
        return PiperTTS()
    except ImportError:
        pass

    raise RuntimeError("No TTS engine available. Install piper-tts or kittentts.")


def list_available_engines():
    """List TTS engines that are currently available."""
    available = []

    try:
        from .PiperTts import PiperTTS
        available.append({
            "id": "piper",
            "name": "Piper TTS",
            "description": "High-quality neural TTS (offline)",
            "sample_rate": 22050,
        })
    except ImportError:
        pass

    try:
        from .Tts import TTS
        available.append({
            "id": "kitten",
            "name": "KittenTTS",
            "description": "Lightweight TTS (faster, lower quality)",
            "sample_rate": 24000,
        })
    except ImportError:
        pass

    return available


def get_voices_for_engine(engine):
    """Get available voices for a TTS engine."""
    engine = engine.lower() if engine else "piper"

    if engine == "piper":
        try:
            from .PiperTts import PIPER_VOICES
            return list(PIPER_VOICES.keys())
        except ImportError:
            return []

    if engine == "kitten":
        try:
            from .Tts import VOICE_MAP
            return list(VOICE_MAP.keys())
        except ImportError:
            return []

    return []
