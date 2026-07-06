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

# Workaround: faster_whisper.utils.disabled_tqdm is a tqdm subclass that never
# keeps `_lock` set on the class, so tqdm.contrib.concurrent.ensure_lock crashes
# on its cleanup `del tqdm_class._lock` during huggingface_hub.snapshot_download.
# Pre-seeding `_lock` makes ensure_lock take the "restore" branch (set_lock(old))
# instead of attempting the delete.
try:
    from faster_whisper.utils import disabled_tqdm as _fw_disabled_tqdm
    from tqdm.std import TqdmDefaultWriteLock as _TqdmDefaultWriteLock
    if getattr(_fw_disabled_tqdm, "_lock", None) is None:
        _fw_disabled_tqdm._lock = _TqdmDefaultWriteLock()
except Exception:
    pass

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


# ---------------------------------------------------------------------------
# Live (streaming) transcription
# ---------------------------------------------------------------------------

LIVE_SAMPLE_RATE = 16000
LIVE_WINDOW_SECONDS = 30          # max audio kept in the rolling buffer
LIVE_TAIL_SECONDS = 4.0           # last N seconds of buffer treated as unstable / partial
LIVE_DECODE_INTERVAL_SECONDS = 2.5  # re-decode after this much new audio
LIVE_MIN_DECODE_SECONDS = 1.5     # don't decode buffers shorter than this

# Process-wide model cache so connections share weights
_LIVE_MODEL_CACHE: Dict[tuple, Any] = {}


def _get_live_model(model_size: str, device: str, compute_type: str):
    """Lazy-load and cache a faster-whisper model keyed by (size, device, compute)."""
    from faster_whisper import WhisperModel
    key = (model_size, device, compute_type)
    if key not in _LIVE_MODEL_CACHE:
        logger.info(f"Loading live whisper model: {key}")
        _LIVE_MODEL_CACHE[key] = WhisperModel(model_size, device=device, compute_type=compute_type)
    return _LIVE_MODEL_CACHE[key]


class LiveTranscriber:
    """
    Sliding-window live transcription over a stream of PCM int16 16kHz mono chunks.

    Producers call `feed_pcm16` with raw bytes; whenever enough new audio has
    accumulated, `maybe_decode` is called to run faster-whisper across the
    whole buffer. Segments older than the unstable-tail threshold become
    "final" and are persisted to disk + returned; newer ones are "partial"
    and may change on the next pass.

    Files written incrementally to .mp/transcripts/<session>.{txt,srt,json}.
    """

    def __init__(
        self,
        session_id: str,
        transcripts_dir: str,
        model_size: str = "base",
        language: Optional[str] = None,
        translate_to_english: bool = False,
    ):
        import numpy as np

        self.session_id = session_id
        self.transcripts_dir = transcripts_dir
        os.makedirs(transcripts_dir, exist_ok=True)

        self.model_size = model_size
        self.language = language
        self.task = "translate" if translate_to_english else "transcribe"

        self.device = get_device()
        self.compute_type = "int8" if self.device == "cpu" else "float16"

        self._np = np
        self._buffer = np.zeros(0, dtype=np.float32)
        self._buffer_start_time = 0.0          # absolute offset of buffer[0]
        self._committed_through = 0.0          # absolute time before which segments are final
        self._last_decode_samples = 0          # sample count at last decode
        self._final_segments: List[Dict[str, Any]] = []

        self.txt_path = os.path.join(transcripts_dir, f"{session_id}.txt")
        self.srt_path = os.path.join(transcripts_dir, f"{session_id}.srt")
        self.json_path = os.path.join(transcripts_dir, f"{session_id}.json")
        self._txt_fh = open(self.txt_path, "w", encoding="utf-8")
        self._srt_fh = open(self.srt_path, "w", encoding="utf-8")
        self._srt_index = 0

        self._model = None  # loaded on first decode

    def feed_pcm16(self, pcm_bytes: bytes) -> None:
        """Append int16 little-endian PCM bytes (16kHz mono) to the buffer."""
        if not pcm_bytes:
            return
        ints = self._np.frombuffer(pcm_bytes, dtype=self._np.int16)
        floats = ints.astype(self._np.float32) / 32768.0
        self._buffer = self._np.concatenate([self._buffer, floats])

    def _ensure_model(self):
        if self._model is None:
            self._model = _get_live_model(self.model_size, self.device, self.compute_type)

    def maybe_decode(self) -> Optional[Dict[str, Any]]:
        """
        If enough new audio has accumulated, decode and return
        {"final": [...], "partial": [...]} with absolute timestamps.
        Otherwise return None.
        """
        new_samples = len(self._buffer) - self._last_decode_samples
        if new_samples < LIVE_DECODE_INTERVAL_SECONDS * LIVE_SAMPLE_RATE:
            return None
        if len(self._buffer) < LIVE_MIN_DECODE_SECONDS * LIVE_SAMPLE_RATE:
            return None
        return self._decode(force_final=False)

    def flush(self) -> Optional[Dict[str, Any]]:
        """Decode any remaining audio, treating everything as final. Called on stop."""
        if len(self._buffer) < 0.4 * LIVE_SAMPLE_RATE:
            return self._finalize_files()
        result = self._decode(force_final=True)
        self._finalize_files()
        return result

    def _decode(self, force_final: bool) -> Dict[str, Any]:
        self._ensure_model()
        segments_gen, _info = self._model.transcribe(
            self._buffer,
            task=self.task,
            language=self.language,
            beam_size=1,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=400),
            condition_on_previous_text=False,
        )

        buffer_duration = len(self._buffer) / LIVE_SAMPLE_RATE
        stable_cutoff = buffer_duration if force_final else max(0.0, buffer_duration - LIVE_TAIL_SECONDS)

        new_finals: List[Dict[str, Any]] = []
        partials: List[Dict[str, Any]] = []

        for seg in segments_gen:
            text = (seg.text or "").strip()
            if not text:
                continue
            abs_start = self._buffer_start_time + float(seg.start)
            abs_end = self._buffer_start_time + float(seg.end)
            if seg.end <= stable_cutoff:
                if abs_start + 0.05 >= self._committed_through:
                    new_finals.append({"start": abs_start, "end": abs_end, "text": text})
            else:
                partials.append({"start": abs_start, "end": abs_end, "text": text})

        for fs in new_finals:
            self._final_segments.append(fs)
            self._srt_index += 1
            self._txt_fh.write(fs["text"] + "\n")
            self._srt_fh.write(self._format_srt_entry(self._srt_index, fs))
            self._committed_through = max(self._committed_through, fs["end"])
        if new_finals:
            self._txt_fh.flush()
            self._srt_fh.flush()
            import json as _json
            with open(self.json_path, "w", encoding="utf-8") as f:
                _json.dump({"session": self.session_id, "segments": self._final_segments}, f, indent=2)

        # Trim buffer up to the last committed point so we don't re-decode it
        if new_finals:
            keep_from_relative = self._committed_through - self._buffer_start_time
            keep_from_samples = int(keep_from_relative * LIVE_SAMPLE_RATE)
            if keep_from_samples > 0:
                self._buffer = self._buffer[keep_from_samples:]
                self._buffer_start_time = self._committed_through

        # Cap absolute buffer length so we don't drift
        max_samples = int(LIVE_WINDOW_SECONDS * LIVE_SAMPLE_RATE)
        if len(self._buffer) > max_samples:
            drop = len(self._buffer) - max_samples
            self._buffer = self._buffer[drop:]
            self._buffer_start_time += drop / LIVE_SAMPLE_RATE

        self._last_decode_samples = len(self._buffer)
        return {"final": new_finals, "partial": partials}

    def _finalize_files(self) -> Dict[str, Any]:
        try:
            self._txt_fh.close()
        except Exception:
            pass
        try:
            self._srt_fh.close()
        except Exception:
            pass
        return {"final": [], "partial": [], "complete": True}

    def transcript_urls(self, base_url: str = "") -> Dict[str, str]:
        """Return URLs (or paths) for the persisted transcript files."""
        return {
            "txt": f"{base_url}/{self.session_id}.txt",
            "srt": f"{base_url}/{self.session_id}.srt",
            "json": f"{base_url}/{self.session_id}.json",
        }

    @staticmethod
    def _format_srt_entry(index: int, seg: Dict[str, Any]) -> str:
        start = LiveTranscriber._format_srt_timestamp(seg["start"])
        end = LiveTranscriber._format_srt_timestamp(seg["end"])
        return f"{index}\n{start} --> {end}\n{seg['text']}\n\n"

    @staticmethod
    def _format_srt_timestamp(seconds: float) -> str:
        if seconds < 0:
            seconds = 0.0
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds - int(seconds)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
