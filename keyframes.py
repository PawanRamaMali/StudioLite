"""
Keyframe Animation System for StudioLite.

Allows users to define start/end states (images or prompts) and generate
smooth video transitions between them.
"""

import os
import numpy as np
from uuid import uuid4
from typing import Callable, Optional, List
from PIL import Image

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(ROOT_DIR, ".mp")


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


# Interpolation curves
EASING_FUNCTIONS = {
    "linear": lambda t: t,
    "ease_in": lambda t: t * t,
    "ease_out": lambda t: 1 - (1 - t) * (1 - t),
    "ease_in_out": lambda t: 3 * t * t - 2 * t * t * t,
    "bounce": lambda t: 1 - abs(np.sin(t * np.pi * 2.5)) * (1 - t),
}


def interpolate_images(
    img_a_path: str,
    img_b_path: str,
    num_frames: int = 30,
    easing: str = "linear",
    output_dir: str = None,
) -> list:
    """
    Create interpolated frames between two images using alpha blending.

    Args:
        img_a_path: Start image path
        img_b_path: End image path
        num_frames: Number of transition frames
        easing: Easing function name

    Returns:
        List of frame file paths
    """
    output_dir = output_dir or OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    img_a = Image.open(img_a_path).convert("RGB")
    img_b = Image.open(img_b_path).convert("RGB")

    # Resize to match
    target_size = img_a.size
    img_b = img_b.resize(target_size, Image.LANCZOS)

    arr_a = np.array(img_a, dtype=np.float32)
    arr_b = np.array(img_b, dtype=np.float32)

    ease_fn = EASING_FUNCTIONS.get(easing, EASING_FUNCTIONS["linear"])
    frame_paths = []

    for i in range(num_frames):
        t = i / max(num_frames - 1, 1)
        alpha = ease_fn(t)

        blended = arr_a * (1 - alpha) + arr_b * alpha
        frame = Image.fromarray(blended.astype(np.uint8))

        frame_path = os.path.join(output_dir, f"kf_{i:04d}.png")
        frame.save(frame_path)
        frame_paths.append(frame_path)

    return frame_paths


def frames_to_video(frame_paths: list, output_path: str = None, fps: int = 24) -> str:
    """Convert a sequence of frame images to a video file."""
    import cv2
    ensure_output_dir()
    output_path = output_path or os.path.join(OUTPUT_DIR, f"keyframe_{uuid4()}.mp4")

    if not frame_paths:
        raise ValueError("No frames provided")

    first = cv2.imread(frame_paths[0])
    h, w = first.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    for fp in frame_paths:
        frame = cv2.imread(fp)
        if frame is not None:
            writer.write(frame)

    writer.release()

    # Re-encode with H.264 for browser compatibility
    h264_path = output_path.replace(".mp4", "_h264.mp4")
    try:
        import subprocess
        subprocess.run([
            "ffmpeg", "-y", "-i", output_path,
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            h264_path,
        ], capture_output=True, check=True)
        os.replace(h264_path, output_path)
    except Exception:
        pass  # Keep original if ffmpeg fails

    return output_path


def generate_keyframe_video(
    start_image: str,
    end_image: str,
    generator=None,
    prompt: str = "",
    num_frames: int = 49,
    fps: int = 16,
    easing: str = "ease_in_out",
    method: str = "blend",
    progress_callback: Callable = None,
) -> str:
    """
    Generate a video transitioning between two keyframe images.

    Methods:
        "blend": Simple alpha blending (fast, no AI)
        "i2v": Use Image-to-Video AI model for start image, guided by prompt
        "interpolate": AI-based frame interpolation (requires RIFE)

    Returns:
        Path to generated video
    """
    ensure_output_dir()

    if method == "blend":
        if progress_callback:
            progress_callback(0, 100, "Generating blend frames...")
        frames = interpolate_images(start_image, end_image, num_frames, easing)
        if progress_callback:
            progress_callback(80, 100, "Encoding video...")
        return frames_to_video(frames, fps=fps)

    elif method == "i2v" and generator:
        if progress_callback:
            progress_callback(0, 100, "Generating I2V from start keyframe...")
        # Use the AI generator to animate the start image
        return generator.generate_image2video(
            start_image, prompt, progress_callback=progress_callback
        )

    elif method == "interpolate":
        # Try RIFE frame interpolation
        try:
            return _rife_interpolate(start_image, end_image, num_frames, fps, progress_callback)
        except ImportError:
            # Fallback to blend
            if progress_callback:
                progress_callback(0, 100, "RIFE not available, using blend...")
            frames = interpolate_images(start_image, end_image, num_frames, easing)
            return frames_to_video(frames, fps=fps)

    else:
        frames = interpolate_images(start_image, end_image, num_frames, easing)
        return frames_to_video(frames, fps=fps)


def _rife_interpolate(start: str, end: str, num_frames: int, fps: int, callback=None) -> str:
    """Frame interpolation using RIFE model."""
    try:
        from rife_ncnn_vulkan_python import Rife
        rife = Rife(gpuid=0)
    except ImportError:
        raise ImportError("RIFE not installed. pip install rife-ncnn-vulkan-python")

    img0 = Image.open(start).convert("RGB")
    img1 = Image.open(end).convert("RGB")
    img1 = img1.resize(img0.size, Image.LANCZOS)

    frames = [np.array(img0)]
    # Recursively interpolate to get desired frame count
    steps = int(np.log2(max(num_frames, 2)))
    for step in range(steps):
        if callback:
            callback(step, steps, f"Interpolating step {step+1}/{steps}")
        new_frames = [frames[0]]
        for i in range(len(frames) - 1):
            mid = rife.process(frames[i], frames[i + 1])
            new_frames.extend([mid, frames[i + 1]])
        frames = new_frames

    # Convert to video
    frame_paths = []
    for i, f in enumerate(frames[:num_frames]):
        path = os.path.join(OUTPUT_DIR, f"rife_{i:04d}.png")
        Image.fromarray(f).save(path)
        frame_paths.append(path)

    return frames_to_video(frame_paths, fps=fps)


def create_keyframe_sequence(keyframes: list, fps: int = 24, easing: str = "ease_in_out") -> str:
    """
    Create a video from multiple keyframes.

    Args:
        keyframes: List of dicts: {"image": path, "duration": seconds}
        fps: Output FPS
        easing: Easing between keyframes

    Returns:
        Path to final video
    """
    from moviepy.editor import VideoFileClip, concatenate_videoclips

    segments = []
    for i in range(len(keyframes) - 1):
        kf_a = keyframes[i]
        kf_b = keyframes[i + 1]
        duration = kf_a.get("duration", 2.0)
        num_frames = int(duration * fps)

        frames = interpolate_images(kf_a["image"], kf_b["image"], num_frames, easing)
        vid_path = frames_to_video(frames, fps=fps)
        segments.append(vid_path)

    if len(segments) == 1:
        return segments[0]

    # Concatenate
    from videogen import concatenate_videos
    return concatenate_videos(segments, fps=fps)
