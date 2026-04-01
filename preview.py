"""
Real-Time Preview / Draft Mode for StudioLite.

Generates quick low-resolution previews before committing to full-quality renders.
Allows rapid iteration on prompts and compositions.
"""

import os
from dataclasses import dataclass
from typing import Optional
from uuid import uuid4

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


@dataclass
class PreviewConfig:
    """Reduced settings for fast preview generation."""
    width: int = 320
    height: int = 192
    num_frames: int = 17
    num_inference_steps: int = 6
    guidance_scale: float = 3.0
    fps: int = 8


@dataclass
class FullQualityConfig:
    """Standard settings for full quality render."""
    width: int = 832
    height: int = 480
    num_frames: int = 49
    num_inference_steps: int = 30
    guidance_scale: float = 5.0
    fps: int = 16


# Preview presets
PREVIEW_PRESETS = {
    "ultra_fast": PreviewConfig(width=256, height=144, num_frames=9, num_inference_steps=4, fps=8),
    "fast": PreviewConfig(width=320, height=192, num_frames=17, num_inference_steps=6, fps=8),
    "balanced": PreviewConfig(width=480, height=272, num_frames=25, num_inference_steps=10, fps=12),
}


def get_preview_config(preset: str = "fast") -> PreviewConfig:
    """Get preview configuration by preset name."""
    return PREVIEW_PRESETS.get(preset, PREVIEW_PRESETS["fast"])


def estimate_preview_time(engine: str = "wan", preset: str = "fast") -> float:
    """Estimate preview generation time in seconds."""
    base_times = {"wan": 30, "ltx": 15, "cogvideox": 20, "hunyuan": 45}
    preset_multipliers = {"ultra_fast": 0.3, "fast": 0.5, "balanced": 0.8}
    base = base_times.get(engine, 30)
    mult = preset_multipliers.get(preset, 0.5)
    return round(base * mult)


def estimate_full_time(engine: str = "wan", num_frames: int = 49, steps: int = 30) -> float:
    """Estimate full quality generation time in seconds."""
    base = {"wan": 300, "ltx": 120, "cogvideox": 180, "hunyuan": 600}
    time_per_frame = base.get(engine, 300) / 49
    step_factor = steps / 30
    return round(time_per_frame * num_frames * step_factor)


def apply_preview_settings(gen_config, preview_preset: str = "fast"):
    """
    Apply preview settings to a VideoGenConfig object.
    Returns a modified copy with reduced settings for fast preview.
    """
    from dataclasses import replace
    preview = get_preview_config(preview_preset)

    # Create modified config
    config_copy = replace(gen_config)
    config_copy.num_frames = preview.num_frames
    config_copy.num_inference_steps = preview.num_inference_steps
    config_copy.guidance_scale = preview.guidance_scale
    config_copy.fps = preview.fps

    # Adjust resolution based on engine
    if hasattr(config_copy, 'wan_resolution'):
        config_copy.wan_resolution = "480p"  # Force lowest resolution
    if hasattr(config_copy, 'width'):
        config_copy.width = preview.width
    if hasattr(config_copy, 'height'):
        config_copy.height = preview.height

    return config_copy


def generate_batch_previews(
    generator,
    prompt: str,
    negative_prompt: str = None,
    num_variations: int = 4,
    preview_preset: str = "fast",
    progress_callback=None,
):
    """
    Generate multiple preview variations of the same prompt with different seeds.

    Returns list of (video_path, seed) tuples.
    """
    import torch
    import random

    results = []
    for i in range(num_variations):
        seed = random.randint(0, 2**31)
        generator.config.seed = seed

        if progress_callback:
            progress_callback(i, num_variations, f"Preview {i+1}/{num_variations} (seed: {seed})")

        try:
            vid_path = generator.generate_text2video(
                prompt,
                negative_prompt=negative_prompt,
                progress_callback=None,
            )
            results.append({"path": vid_path, "seed": seed, "index": i})
        except Exception as e:
            results.append({"path": None, "seed": seed, "index": i, "error": str(e)})

    return results
