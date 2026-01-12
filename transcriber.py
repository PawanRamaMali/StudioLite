#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audio/Video Transcription using faster-whisper (local models).

Supports transcribing audio from video files and audio files
using OpenAI's Whisper model running locally via faster-whisper.
"""

import os
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Supported languages (ISO 639-1 codes)
LANGUAGES = {
    "Auto-detect": None,
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Italian": "it",
    "Portuguese": "pt",
    "Dutch": "nl",
    "Russian": "ru",
    "Chinese": "zh",
    "Japanese": "ja",
    "Korean": "ko",
    "Arabic": "ar",
    "Hindi": "hi",
    "Turkish": "tr",
    "Polish": "pl",
    "Vietnamese": "vi",
    "Thai": "th",
    "Indonesian": "id",
    "Swedish": "sv",
    "Danish": "da",
    "Norwegian": "no",
    "Finnish": "fi",
    "Greek": "el",
    "Hebrew": "he",
    "Czech": "cs",
    "Romanian": "ro",
    "Hungarian": "hu",
    "Ukrainian": "uk",
}

# Model sizes available
MODEL_SIZES = ["tiny", "base", "small", "medium", "large-v2", "large-v3"]

# Compute types
COMPUTE_TYPES = ["int8", "float16", "float32"]

# Output formats
OUTPUT_FORMATS = ["txt", "srt", "vtt", "json", "tsv"]


def check_whisperx_installed() -> tuple[bool, str]:
    """Check if faster-whisper is installed."""
    try:
        from faster_whisper import WhisperModel
        import torch
        device = "CUDA" if torch.cuda.is_available() else "CPU"
        return True, f"faster-whisper ready ({device})"
    except ImportError as e:
        return False, f"Missing dependency: {e}. Run: pip install faster-whisper"


def get_device() -> str:
    """Get the best available device (cuda or cpu)."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


@dataclass
class TranscriptionConfig:
    """Configuration for transcription."""
    model_size: str = "base"
    language: Optional[str] = None  # None for auto-detect
    compute_type: str = "float16"
    batch_size: int = 16
    translate_to_english: bool = False


class Transcriber:
    """Handles audio/video transcription using faster-whisper."""

    def __init__(self, config: Optional[TranscriptionConfig] = None):
        """
        Initialize the transcriber.

        Args:
            config: Transcription configuration
        """
        self.config = config or TranscriptionConfig()
        self._model = None
        self._segments = []

    def transcribe(
        self,
        input_path: str,
        output_formats: Optional[List[str]] = None,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Transcribe audio/video file.

        Args:
            input_path: Path to audio or video file
            output_formats: List of output formats (txt, srt, vtt, json, tsv)
            progress_callback: Optional callback for progress updates

        Returns:
            Dict with 'success', 'text', 'segments', 'language', 'error' keys
        """
        result = {
            "success": False,
            "text": None,
            "segments": None,
            "language": None,
            "error": None
        }

        if not os.path.exists(input_path):
            result["error"] = f"File not found: {input_path}"
            return result

        try:
            from faster_whisper import WhisperModel

            device = get_device()
            compute_type = self.config.compute_type

            # Adjust compute type for CPU
            if device == "cpu" and compute_type == "float16":
                compute_type = "int8"
                logger.info("Switched to int8 compute type for CPU")

            if progress_callback:
                progress_callback("Loading model...")

            # Load model
            self._model = WhisperModel(
                self.config.model_size,
                device=device,
                compute_type=compute_type
            )

            if progress_callback:
                progress_callback("Transcribing...")

            # Transcribe
            task = "translate" if self.config.translate_to_english else "transcribe"

            segments_generator, info = self._model.transcribe(
                input_path,
                task=task,
                language=self.config.language,
                beam_size=5,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500),
            )

            # Collect segments
            self._segments = []
            text_parts = []
            segment_count = 0

            if progress_callback:
                progress_callback("Processing segments... (this may take a while for long files)")

            for segment in segments_generator:
                seg_dict = {
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text.strip()
                }
                self._segments.append(seg_dict)
                text_parts.append(segment.text.strip())
                segment_count += 1

                # Update progress every 10 segments
                if progress_callback and segment_count % 10 == 0:
                    minutes = int(segment.end // 60)
                    seconds = int(segment.end % 60)
                    progress_callback(f"Transcribing... {segment_count} segments ({minutes}:{seconds:02d} processed)")

            text = " ".join(text_parts)
            detected_language = info.language

            if progress_callback:
                progress_callback(f"Completed! {segment_count} segments transcribed.")

            result["success"] = True
            result["text"] = text
            result["segments"] = self._segments
            result["language"] = detected_language
            logger.info(f"Transcription complete: {segment_count} segments, language: {detected_language}")

        except Exception as e:
            import traceback
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            result["error"] = str(e)
            logger.error(f"Transcription error: {error_msg}")

        return result

    def generate_srt(self, segments: List[Dict]) -> str:
        """
        Generate SRT subtitle content from segments.

        Args:
            segments: List of transcription segments with start, end, text

        Returns:
            SRT formatted string
        """
        srt_content = []

        for i, segment in enumerate(segments, 1):
            start = segment.get("start", 0)
            end = segment.get("end", start + 1)
            text = segment.get("text", "").strip()

            start_time = self._format_timestamp_srt(start)
            end_time = self._format_timestamp_srt(end)

            srt_content.append(f"{i}")
            srt_content.append(f"{start_time} --> {end_time}")
            srt_content.append(text)
            srt_content.append("")

        return "\n".join(srt_content)

    def generate_vtt(self, segments: List[Dict]) -> str:
        """
        Generate VTT subtitle content from segments.

        Args:
            segments: List of transcription segments with start, end, text

        Returns:
            VTT formatted string
        """
        vtt_content = ["WEBVTT", ""]

        for segment in segments:
            start = segment.get("start", 0)
            end = segment.get("end", start + 1)
            text = segment.get("text", "").strip()

            start_time = self._format_timestamp_vtt(start)
            end_time = self._format_timestamp_vtt(end)

            vtt_content.append(f"{start_time} --> {end_time}")
            vtt_content.append(text)
            vtt_content.append("")

        return "\n".join(vtt_content)

    @staticmethod
    def _format_timestamp_srt(seconds: float) -> str:
        """Format seconds to SRT timestamp (HH:MM:SS,mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    @staticmethod
    def _format_timestamp_vtt(seconds: float) -> str:
        """Format seconds to VTT timestamp (HH:MM:SS.mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def get_language_list() -> List[str]:
    """Return list of available languages."""
    return list(LANGUAGES.keys())


def get_language_code(language_name: str) -> Optional[str]:
    """Get language code from language name."""
    return LANGUAGES.get(language_name)
