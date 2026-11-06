"""AudioLDM2-based ambient and foley generation.

Local text-to-audio: "crickets and distant traffic", "footsteps on gravel",
"wind through a forest". Produces a WAV at a requested duration, which the
Timeline can place under a scene as its own layer, or Social Export can mix
in alongside narration.

Backends:
- AudioLDM2 (primary): https://huggingface.co/cvssp/audioldm2
  Diffusers has a native pipeline; weights ~2GB, CPU-runnable but slow.
- Stable Audio Open (optional, higher quality):
  https://huggingface.co/stabilityai/stable-audio-open-1.0
  Needs HF auth; 24GB works, 12GB tight. Falls back to AudioLDM2 if not loaded.

Pattern follows ``lipsync.py`` / ``talking_head.py``: ``available()`` gate,
pure config builder, injectable ``runner``.
"""

from __future__ import annotations

import logging
import os
import wave
from dataclasses import dataclass
from typing import Callable, Optional

logger = logging.getLogger(__name__)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIOLDM2_REPO_ID = "cvssp/audioldm2"
STABLE_AUDIO_REPO_ID = "stabilityai/stable-audio-open-1.0"

# Cache dir for HF snapshots when the model has been downloaded locally.
AUDIOLDM2_LOCAL = os.path.join(ROOT_DIR, ".models", "audioldm2")
STABLE_AUDIO_LOCAL = os.path.join(ROOT_DIR, ".models", "stable-audio-open")


def _has_diffusers() -> bool:
    try:
        import importlib
        importlib.import_module("diffusers")
        return True
    except ImportError:
        return False


def _diffusers_has(class_name: str) -> bool:
    """diffusers ships AudioLDM2Pipeline in >= 0.21. StableAudio in >= 0.27."""
    if not _has_diffusers():
        return False
    try:
        import diffusers
        return hasattr(diffusers, class_name)
    except Exception:
        return False


def audioldm2_available() -> bool:
    """True iff diffusers knows about AudioLDM2Pipeline."""
    return _diffusers_has("AudioLDM2Pipeline")


def stable_audio_available() -> bool:
    """True iff diffusers knows about StableAudioPipeline."""
    return _diffusers_has("StableAudioPipeline")


def available_backends() -> list[str]:
    out = []
    if audioldm2_available():
        out.append("audioldm2")
    if stable_audio_available():
        out.append("stable_audio")
    return out


def install_hint() -> str:
    lines = ["Ambient/foley generation needs ONE of:"]
    if not _has_diffusers():
        lines.append(" - pip install 'diffusers>=0.27'")
    if not audioldm2_available():
        lines.append(
            f" - AudioLDM2: ships with diffusers >= 0.21; "
            f"first call downloads {AUDIOLDM2_REPO_ID} (~2GB)"
        )
    if not stable_audio_available():
        lines.append(
            f" - Stable Audio Open (higher quality): diffusers >= 0.27 + "
            f"HF auth to download {STABLE_AUDIO_REPO_ID}"
        )
    if available_backends():
        return "Ambient audio generation is set up."
    return "\n".join(lines)


@dataclass
class AmbientConfig:
    prompt: str
    duration_s: float = 5.0
    negative_prompt: str = "low quality, distorted, noise, silence"
    num_inference_steps: int = 50          # 20 for draft, 50 for final
    guidance_scale: float = 3.5
    sample_rate: int = 48000               # social delivery target
    seed: Optional[int] = None
    output_path: str = ""                  # writer picks a default if blank


# Pure validation / shaping helpers --------------------------------------


def _resolve_output_path(cfg: AmbientConfig) -> str:
    if cfg.output_path:
        return cfg.output_path
    tag = f"{abs(hash((cfg.prompt, cfg.seed, cfg.duration_s))) & 0xFFFF:04x}"
    return os.path.join(ROOT_DIR, ".mp", f"ambient_{tag}.wav")


def _validate_config(cfg: AmbientConfig) -> None:
    if not cfg.prompt or not cfg.prompt.strip():
        raise ValueError("prompt is required")
    if cfg.duration_s <= 0 or cfg.duration_s > 60:
        raise ValueError(f"duration_s must be 0-60 (got {cfg.duration_s})")
    if cfg.sample_rate not in (16000, 22050, 24000, 44100, 48000):
        raise ValueError(f"sample_rate must be a standard rate (got {cfg.sample_rate})")
    if cfg.num_inference_steps < 4 or cfg.num_inference_steps > 200:
        raise ValueError(
            f"num_inference_steps must be 4-200 (got {cfg.num_inference_steps})"
        )


def pick_backend(preferred: str = "auto") -> str:
    """Return the backend id to use. Raises when none are installed."""
    backends = available_backends()
    if not backends:
        raise RuntimeError(install_hint())
    if preferred == "auto":
        # Prefer Stable Audio when available - it produces higher-fidelity output
        return "stable_audio" if "stable_audio" in backends else "audioldm2"
    if preferred not in backends:
        raise RuntimeError(
            f"Backend {preferred!r} is not available. Installed: {backends}. "
            f"{install_hint()}"
        )
    return preferred


# Orchestration ----------------------------------------------------------


def generate_ambient(cfg: AmbientConfig, *, backend: str = "auto",
                     progress_callback: Optional[Callable] = None,
                     runner: Optional[Callable] = None) -> str:
    """Generate a WAV of ambient audio matching ``cfg.prompt``.

    ``runner`` is injectable. Tests pass a fake that writes a silent WAV so
    we don't need the model weights at test time. Production passes
    ``_real_runner`` which calls the diffusers pipeline.
    """
    _validate_config(cfg)
    out_path = _resolve_output_path(cfg)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    picked = pick_backend(backend)
    if progress_callback:
        progress_callback(5, 100, f"Loading {picked}...")

    r = runner if runner is not None else _real_runner
    r(cfg, out_path, picked, progress_callback)

    if not os.path.isfile(out_path) or os.path.getsize(out_path) < 1024:
        raise RuntimeError(f"Ambient generation produced no usable output at {out_path}")
    if progress_callback:
        progress_callback(100, 100, "Done.")
    return out_path


def write_silent_wav(path: str, duration_s: float, sample_rate: int = 48000) -> None:
    """Write a silent WAV header + body. Used by tests + as a graceful fallback."""
    n_frames = int(round(duration_s * sample_rate))
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * max(1, n_frames))


def _real_runner(cfg: AmbientConfig, out_path: str, backend: str,
                 progress_callback: Optional[Callable]) -> None:
    """The actual diffusers call. Imported lazily to keep tests light."""
    import torch
    import soundfile as sf

    if backend == "stable_audio":
        from diffusers import StableAudioPipeline  # type: ignore
        pipe = StableAudioPipeline.from_pretrained(
            STABLE_AUDIO_LOCAL if os.path.isdir(STABLE_AUDIO_LOCAL) else STABLE_AUDIO_REPO_ID,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        pipe.enable_model_cpu_offload() if torch.cuda.is_available() else None
    else:
        from diffusers import AudioLDM2Pipeline  # type: ignore
        pipe = AudioLDM2Pipeline.from_pretrained(
            AUDIOLDM2_LOCAL if os.path.isdir(AUDIOLDM2_LOCAL) else AUDIOLDM2_REPO_ID,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        if torch.cuda.is_available():
            pipe.to("cuda")

    generator = torch.Generator(device="cpu")
    if cfg.seed is not None:
        generator.manual_seed(int(cfg.seed))

    def _step_cb(_pipe, step_index, _timestep, _kwargs):
        if progress_callback:
            pct = 15 + int((step_index / cfg.num_inference_steps) * 75)
            progress_callback(pct, 100, f"step {step_index + 1}/{cfg.num_inference_steps}")
        return _kwargs

    out = pipe(
        prompt=cfg.prompt,
        negative_prompt=cfg.negative_prompt,
        audio_length_in_s=cfg.duration_s,
        num_inference_steps=cfg.num_inference_steps,
        guidance_scale=cfg.guidance_scale,
        generator=generator,
        callback_on_step_end=_step_cb,
    )
    audio = out.audios[0] if hasattr(out, "audios") else out["audios"][0]
    sf.write(out_path, audio, cfg.sample_rate)
