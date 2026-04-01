"""
Video Upscaler for StudioLite.

Upscale AI-generated videos (480p) to 1080p/4K using Real-ESRGAN
or OpenCV fallback. Frame-by-frame processing with audio preservation.
"""

import os
import cv2
import subprocess
import numpy as np
from uuid import uuid4
from typing import Callable, Optional

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(ROOT_DIR, ".mp")


UPSCALE_PRESETS = {
    "fast_2x": {
        "name": "Fast 2x",
        "scale": 2,
        "description": "Fast 2x upscale using Lanczos interpolation",
        "method": "lanczos",
    },
    "quality_2x": {
        "name": "Quality 2x",
        "scale": 2,
        "description": "High quality 2x upscale (Real-ESRGAN if available)",
        "method": "realesrgan",
    },
    "ultra_4x": {
        "name": "Ultra 4x",
        "scale": 4,
        "description": "Ultra 4x upscale for near-4K output",
        "method": "realesrgan",
    },
    "anime_4x": {
        "name": "Anime 4x",
        "scale": 4,
        "description": "Optimized for anime/cartoon content",
        "method": "realesrgan_anime",
    },
}


def check_upscaler_available() -> tuple:
    """Check available upscaling backends."""
    backends = []

    # Always available: OpenCV Lanczos
    backends.append("lanczos")

    # Check Real-ESRGAN
    try:
        from realesrgan import RealESRGANer
        backends.append("realesrgan")
    except ImportError:
        pass

    # Check Real-ESRGAN CLI
    try:
        result = subprocess.run(
            ["realesrgan-ncnn-vulkan", "--help"],
            capture_output=True, timeout=5
        )
        backends.append("realesrgan_cli")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return len(backends) > 0, backends


def get_video_info(video_path: str) -> dict:
    """Get video metadata."""
    cap = cv2.VideoCapture(video_path)
    info = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "duration": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / max(cap.get(cv2.CAP_PROP_FPS), 1),
    }
    cap.release()
    return info


def _upscale_frame_lanczos(frame: np.ndarray, scale: int) -> np.ndarray:
    """Upscale a frame using OpenCV Lanczos interpolation."""
    h, w = frame.shape[:2]
    return cv2.resize(frame, (w * scale, h * scale), interpolation=cv2.INTER_LANCZOS4)


def _upscale_frame_realesrgan(frame: np.ndarray, upsampler) -> np.ndarray:
    """Upscale a frame using Real-ESRGAN."""
    output, _ = upsampler.enhance(frame, outscale=upsampler.scale)
    return output


def _get_realesrgan_model(scale: int = 2, anime: bool = False):
    """Load Real-ESRGAN model."""
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer

    if anime and scale == 4:
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
        model_name = "RealESRGAN_x4plus_anime_6B"
    elif scale == 4:
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        model_name = "RealESRGAN_x4plus"
    else:
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        model_name = "RealESRGAN_x2plus"

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    half = device == "cuda"

    upsampler = RealESRGANer(
        scale=scale,
        model_path=None,  # Auto-download
        model=model,
        tile=256,
        tile_pad=10,
        pre_pad=0,
        half=half,
        device=device,
    )
    return upsampler


def upscale_image(image_path: str, scale: int = 2, method: str = "lanczos",
                   output_path: str = None) -> str:
    """Upscale a single image."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = output_path or os.path.join(OUTPUT_DIR, f"upscaled_{uuid4()}.png")

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    if method == "realesrgan":
        try:
            upsampler = _get_realesrgan_model(scale)
            result = _upscale_frame_realesrgan(img, upsampler)
        except (ImportError, Exception):
            result = _upscale_frame_lanczos(img, scale)
    else:
        result = _upscale_frame_lanczos(img, scale)

    cv2.imwrite(output_path, result)
    return output_path


def upscale_video(
    video_path: str,
    scale: int = 2,
    method: str = "lanczos",
    output_path: str = None,
    progress_callback: Callable = None,
) -> str:
    """
    Upscale a video frame-by-frame.

    Args:
        video_path: Input video path
        scale: Upscale factor (2 or 4)
        method: "lanczos", "realesrgan", or "realesrgan_anime"
        output_path: Output path (auto-generated if None)
        progress_callback: callback(current_frame, total_frames, message)

    Returns:
        Path to upscaled video
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = output_path or os.path.join(OUTPUT_DIR, f"upscaled_{uuid4()}.mp4")

    # Get video info
    info = get_video_info(video_path)
    total_frames = info["frame_count"]
    fps = info["fps"]
    new_w = info["width"] * scale
    new_h = info["height"] * scale

    if progress_callback:
        progress_callback(0, total_frames, f"Upscaling {info['width']}x{info['height']} → {new_w}x{new_h}")

    # Load Real-ESRGAN if requested
    upsampler = None
    if "realesrgan" in method:
        try:
            anime = "anime" in method
            upsampler = _get_realesrgan_model(scale, anime)
            if progress_callback:
                progress_callback(0, total_frames, "Real-ESRGAN model loaded")
        except (ImportError, Exception) as e:
            if progress_callback:
                progress_callback(0, total_frames, f"Real-ESRGAN unavailable ({e}), using Lanczos")

    # Process frames
    cap = cv2.VideoCapture(video_path)
    temp_path = output_path + ".tmp.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(temp_path, fourcc, fps, (new_w, new_h))

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if upsampler:
            try:
                upscaled = _upscale_frame_realesrgan(frame, upsampler)
            except Exception:
                upscaled = _upscale_frame_lanczos(frame, scale)
        else:
            upscaled = _upscale_frame_lanczos(frame, scale)

        # Ensure exact output size
        if upscaled.shape[1] != new_w or upscaled.shape[0] != new_h:
            upscaled = cv2.resize(upscaled, (new_w, new_h))

        writer.write(upscaled)
        frame_idx += 1

        if progress_callback and frame_idx % 3 == 0:
            progress_callback(frame_idx, total_frames,
                              f"Upscaling frame {frame_idx}/{total_frames}")

    cap.release()
    writer.release()

    # Re-encode with H.264 and copy audio from original
    try:
        subprocess.run([
            "ffmpeg", "-y",
            "-i", temp_path,
            "-i", video_path,
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-c:a", "aac",
            "-map", "0:v:0", "-map", "1:a:0?",
            "-shortest",
            output_path,
        ], capture_output=True, check=True)
        os.remove(temp_path)
    except Exception:
        os.replace(temp_path, output_path)

    if progress_callback:
        progress_callback(total_frames, total_frames,
                          f"Complete! Output: {new_w}x{new_h}")

    return output_path
