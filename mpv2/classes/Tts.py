import os
import re
import numpy as np
import soundfile as sf
from kittentts import KittenTTS as KittenModel

from ..config import ROOT_DIR, get_tts_voice

KITTEN_SAMPLE_RATE = 24000
MAX_CHUNK_LENGTH = 200  # Characters per chunk to avoid model issues

# Voice mapping from user-friendly names to kittentts internal names
VOICE_MAP = {
    "Jasper": "expr-voice-2-m",
    "Luna": "expr-voice-2-f",
    "Marcus": "expr-voice-3-m",
    "Elena": "expr-voice-3-f",
    "Thomas": "expr-voice-4-m",
    "Sofia": "expr-voice-4-f",
    "Alex": "expr-voice-5-m",
    "Emma": "expr-voice-5-f",
}

class TTS:
    def __init__(self) -> None:
        # Use default model (kitten-tts-nano-0.1) - no args needed
        self._model = KittenModel()
        voice_name = get_tts_voice()
        self._voice = VOICE_MAP.get(voice_name, "expr-voice-2-m")

    def _clean_text(self, text: str) -> str:
        """Clean text for TTS - remove problematic characters."""
        # Keep only ASCII letters, numbers, spaces, and basic punctuation
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove non-ASCII
        text = re.sub(r'[#@$%^&*()_+=\[\]{}|\\<>/~`"]+', ' ', text)  # Remove special chars
        text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
        return text if text else "Hello"

    def _split_into_chunks(self, text: str) -> list:
        """Split text into manageable chunks for TTS."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current = ""
        for s in sentences:
            if len(current) + len(s) < MAX_CHUNK_LENGTH:
                current += (" " + s if current else s)
            else:
                if current:
                    chunks.append(current.strip())
                current = s
        if current:
            chunks.append(current.strip())
        return chunks if chunks else ["Hello"]

    def synthesize(self, text, output_file=os.path.join(ROOT_DIR, ".mp", "audio.wav")):
        text = self._clean_text(text)
        chunks = self._split_into_chunks(text)

        audio_parts = []
        for chunk in chunks:
            if not chunk.strip():
                continue
            try:
                audio = self._model.generate(chunk, voice=self._voice)
                audio_parts.append(audio)
            except Exception as e:
                print(f"TTS chunk error: {e}, chunk: {chunk[:50]}...")
                continue

        if not audio_parts:
            # Fallback: generate simple audio
            audio_parts.append(self._model.generate("Content generated.", voice=self._voice))

        # Concatenate all audio parts
        full_audio = np.concatenate(audio_parts)
        sf.write(output_file, full_audio, KITTEN_SAMPLE_RATE)
        return output_file
