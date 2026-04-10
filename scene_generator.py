"""
Scene Generator - Image-first video generation pipeline.

Pipeline: SDXL (scene image with IP-Adapter) → SVD-XT (animate to video)

This ensures characters actually appear in the video because they're rendered
in the source image first, then that image is animated.

Requires:
- diffusers (pip install diffusers)
- transformers (for CLIP/IP-Adapter)
- torch with CUDA
"""

import os
import gc
import logging
import torch
import numpy as np
from uuid import uuid4
from typing import Optional, List, Callable
from PIL import Image

logger = logging.getLogger(__name__)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(ROOT_DIR, ".mp")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Global pipeline references (lazy-loaded, one at a time to save VRAM)
_sdxl_pipe = None
_svd_pipe = None


def _unload_all():
    """Unload all pipelines to free VRAM."""
    global _sdxl_pipe, _svd_pipe
    if _sdxl_pipe is not None:
        del _sdxl_pipe
        _sdxl_pipe = None
    if _svd_pipe is not None:
        del _svd_pipe
        _svd_pipe = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _load_sdxl(ip_adapter: bool = False):
    """Load SDXL Turbo pipeline (optionally with IP-Adapter)."""
    global _sdxl_pipe
    if _sdxl_pipe is not None:
        return _sdxl_pipe

    # Unload SVD first to free VRAM
    global _svd_pipe
    if _svd_pipe is not None:
        del _svd_pipe
        _svd_pipe = None
        gc.collect()
        torch.cuda.empty_cache()

    from diffusers import AutoPipelineForText2Image

    # Check for local SDXL model first
    models_dir = os.path.join(ROOT_DIR, "models")
    local_model = None
    if os.path.isdir(models_dir):
        for f in os.listdir(models_dir):
            if f.endswith(".safetensors"):
                local_model = os.path.join(models_dir, f)
                break

    if local_model:
        from diffusers import StableDiffusionXLPipeline
        _sdxl_pipe = StableDiffusionXLPipeline.from_single_file(
            local_model, torch_dtype=torch.float16
        )
    else:
        _sdxl_pipe = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16"
        )

    _sdxl_pipe.enable_model_cpu_offload()
    _sdxl_pipe.set_progress_bar_config(disable=True)

    # Load IP-Adapter for character consistency
    if ip_adapter:
        try:
            _sdxl_pipe.load_ip_adapter(
                "h94/IP-Adapter",
                subfolder="sdxl_models",
                weight_name="ip-adapter-plus_sdxl_vit-h.safetensors",
            )
            _sdxl_pipe.set_ip_adapter_scale(0.6)
            logger.info("SDXL IP-Adapter loaded for character consistency")
        except Exception as e:
            logger.warning(f"IP-Adapter not available for SDXL: {e}")

    return _sdxl_pipe


def _load_svd():
    """Load Stable Video Diffusion XT pipeline."""
    global _svd_pipe
    if _svd_pipe is not None:
        return _svd_pipe

    # Unload SDXL first to free VRAM
    global _sdxl_pipe
    if _sdxl_pipe is not None:
        del _sdxl_pipe
        _sdxl_pipe = None
        gc.collect()
        torch.cuda.empty_cache()

    from diffusers import StableVideoDiffusionPipeline

    _svd_pipe = StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid-xt",
        torch_dtype=torch.float16,
        variant="fp16",
    )
    _svd_pipe.enable_model_cpu_offload()
    # Memory optimizations
    _svd_pipe.unet.enable_forward_chunking()
    _svd_pipe.set_progress_bar_config(disable=True)

    return _svd_pipe


def generate_scene_image(
    prompt: str,
    negative_prompt: str = "",
    character_ref_images: Optional[List[Image.Image]] = None,
    width: int = 1024,
    height: int = 576,
    steps: int = 30,
    guidance: float = 7.5,
    seed: Optional[int] = None,
) -> str:
    """
    Generate a scene image using SDXL, optionally with IP-Adapter character references.

    Args:
        prompt: Scene description with character details
        negative_prompt: What to avoid
        character_ref_images: PIL Images of characters for IP-Adapter injection
        width, height: Image dimensions (SVD expects 1024x576)
        steps, guidance: SDXL inference parameters
        seed: Optional seed for reproducibility

    Returns:
        Path to generated image
    """
    has_refs = character_ref_images and len(character_ref_images) > 0
    pipe = _load_sdxl(ip_adapter=has_refs)

    generator = torch.Generator(device="cpu")
    if seed is not None:
        generator.manual_seed(seed)

    # Check if this is SDXL Turbo (fewer steps, lower guidance)
    is_turbo = not any(
        f.endswith(".safetensors")
        for f in os.listdir(os.path.join(ROOT_DIR, "models"))
    ) if os.path.isdir(os.path.join(ROOT_DIR, "models")) else True

    if is_turbo:
        steps = min(steps, 8)
        guidance = min(guidance, 2.0)

    kwargs = dict(
        prompt=prompt,
        negative_prompt=negative_prompt or "low quality, blurry, distorted, bad anatomy, worst quality",
        num_inference_steps=steps,
        guidance_scale=guidance,
        width=width,
        height=height,
        generator=generator,
    )

    # Inject IP-Adapter reference images
    if has_refs:
        try:
            # Use first reference for single-character, list for multi
            if len(character_ref_images) == 1:
                kwargs["ip_adapter_image"] = character_ref_images[0]
            else:
                kwargs["ip_adapter_image"] = character_ref_images
            logger.info(f"Injecting {len(character_ref_images)} character ref(s) via IP-Adapter")
        except Exception as e:
            logger.warning(f"IP-Adapter injection failed: {e}")

    result = pipe(**kwargs)
    img = result.images[0]

    path = os.path.join(OUTPUT_DIR, f"scene_img_{uuid4().hex[:8]}.png")
    img.save(path, quality=95)
    logger.info(f"Scene image generated: {path} ({width}x{height})")
    return path


def animate_image(
    image_path: str,
    num_frames: int = 25,
    fps: int = 7,
    decode_chunk_size: int = 4,
    motion_bucket_id: int = 127,
    noise_aug_strength: float = 0.02,
    seed: Optional[int] = None,
    progress_callback: Optional[Callable] = None,
) -> str:
    """
    Animate a still image into video using SVD-XT.

    Args:
        image_path: Path to source image
        num_frames: Number of frames (SVD-XT supports up to 25)
        fps: Frames per second for output
        decode_chunk_size: VAE decode chunk size (lower = less VRAM)
        motion_bucket_id: Motion intensity (1-255, higher = more motion)
        noise_aug_strength: How much noise to add to conditioning
        seed: Optional seed
        progress_callback: fn(step, total, message)

    Returns:
        Path to generated video
    """
    pipe = _load_svd()

    # Load and resize image to SVD's expected dimensions
    image = Image.open(image_path).convert("RGB")
    image = image.resize((1024, 576), Image.LANCZOS)

    generator = torch.Generator(device="cpu")
    if seed is not None:
        generator.manual_seed(seed)

    if progress_callback:
        progress_callback(10, 100, "Running SVD-XT animation...")

    frames = pipe(
        image,
        decode_chunk_size=decode_chunk_size,
        generator=generator,
        num_frames=num_frames,
        motion_bucket_id=motion_bucket_id,
        noise_aug_strength=noise_aug_strength,
    ).frames[0]

    if progress_callback:
        progress_callback(90, 100, "Encoding video...")

    # Export frames to video
    from diffusers.utils import export_to_video
    output_path = os.path.join(OUTPUT_DIR, f"scene_vid_{uuid4().hex[:8]}.mp4")
    export_to_video(frames, output_path, fps=fps)

    if progress_callback:
        progress_callback(100, 100, "Animation complete")

    logger.info(f"Animated video: {output_path} ({num_frames} frames @ {fps}fps)")
    return output_path


def generate_scene_video(
    prompt: str,
    negative_prompt: str = "",
    character_ref_images: Optional[List[Image.Image]] = None,
    target_duration: float = 5.0,
    fps: int = 7,
    motion_bucket_id: int = 127,
    seed: Optional[int] = None,
    steps: int = 30,
    guidance: float = 7.5,
    progress_callback: Optional[Callable] = None,
) -> str:
    """
    Full pipeline: Generate scene image with characters → Animate to video.

    This is the main entry point for the image-first pipeline.

    Args:
        prompt: Scene description (should include character appearances)
        negative_prompt: What to avoid
        character_ref_images: Character reference images for IP-Adapter
        target_duration: Desired video duration in seconds
        fps: Frames per second
        motion_bucket_id: Motion intensity (1-255)
        seed: Seed for reproducibility
        steps: SDXL inference steps
        guidance: SDXL guidance scale
        progress_callback: fn(step, total, message)

    Returns:
        Path to generated video
    """
    if progress_callback:
        progress_callback(5, 100, "Step 1/2: Generating scene image with characters...")

    # Step 1: Generate scene image with SDXL + IP-Adapter
    image_path = generate_scene_image(
        prompt=prompt,
        negative_prompt=negative_prompt,
        character_ref_images=character_ref_images,
        width=1024,
        height=576,  # SVD expects 1024x576
        steps=steps,
        guidance=guidance,
        seed=seed,
    )

    if progress_callback:
        progress_callback(40, 100, "Step 2/2: Animating scene with SVD-XT...")

    # Step 2: Animate with SVD-XT
    # SVD-XT generates 25 frames. At 7fps that's ~3.6s per clip.
    # For longer scenes, we generate multiple clips.
    num_frames = 25  # SVD-XT maximum
    clip_duration = num_frames / fps

    if target_duration <= clip_duration:
        # Single clip is enough
        video_path = animate_image(
            image_path=image_path,
            num_frames=num_frames,
            fps=fps,
            motion_bucket_id=motion_bucket_id,
            seed=seed,
            progress_callback=progress_callback,
        )
    else:
        # Need multiple clips - animate the same image with different seeds
        clips = []
        clips_needed = max(1, int(target_duration / clip_duration) + 1)
        clips_needed = min(clips_needed, 3)  # Cap at 3 clips

        for i in range(clips_needed):
            clip_seed = (seed + i * 1000) if seed else None
            pct = 40 + int((i / clips_needed) * 55)
            if progress_callback:
                progress_callback(pct, 100, f"Animating clip {i+1}/{clips_needed}...")

            vp = animate_image(
                image_path=image_path,
                num_frames=num_frames,
                fps=fps,
                motion_bucket_id=motion_bucket_id,
                seed=clip_seed,
            )
            clips.append(vp)

        # Concatenate clips
        if len(clips) > 1:
            from videogen import concatenate_videos
            video_path = concatenate_videos(clips, fps=fps)
        else:
            video_path = clips[0]

    if progress_callback:
        progress_callback(100, 100, "Scene video complete")

    # Unload SVD to free VRAM for next scene's SDXL pass
    _unload_all()

    return video_path
