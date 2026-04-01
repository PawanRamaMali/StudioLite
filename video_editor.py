"""
In-Video Editing for StudioLite.

Edit specific elements within generated videos using text prompts,
region masks, and inpainting techniques.
"""

import os
import cv2
import numpy as np
from uuid import uuid4
from typing import Callable, Optional, List
from PIL import Image, ImageDraw

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(ROOT_DIR, ".mp")


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================
# Frame Extraction & Assembly
# =============================================

def extract_frames(video_path: str, output_dir: str = None) -> list:
    """Extract all frames from a video."""
    output_dir = output_dir or os.path.join(OUTPUT_DIR, f"frames_{uuid4()}")
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    i = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        path = os.path.join(output_dir, f"frame_{i:05d}.png")
        cv2.imwrite(path, frame)
        frames.append(path)
        i += 1

    cap.release()
    return frames, fps


def frames_to_video(frame_paths: list, output_path: str = None,
                     fps: float = 24, audio_from: str = None) -> str:
    """Reassemble frames into video, optionally preserving audio."""
    ensure_output_dir()
    output_path = output_path or os.path.join(OUTPUT_DIR, f"edited_{uuid4()}.mp4")

    if not frame_paths:
        raise ValueError("No frames to assemble")

    first = cv2.imread(frame_paths[0])
    h, w = first.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    for fp in frame_paths:
        frame = cv2.imread(fp)
        if frame is not None:
            if frame.shape[:2] != (h, w):
                frame = cv2.resize(frame, (w, h))
            writer.write(frame)

    writer.release()

    # Add audio from original if available
    if audio_from and os.path.exists(audio_from):
        import subprocess
        temp_path = output_path + ".tmp.mp4"
        try:
            subprocess.run([
                "ffmpeg", "-y",
                "-i", output_path,
                "-i", audio_from,
                "-c:v", "libx264", "-preset", "fast",
                "-c:a", "aac",
                "-map", "0:v:0", "-map", "1:a:0?",
                "-shortest",
                temp_path,
            ], capture_output=True, check=True)
            os.replace(temp_path, output_path)
        except Exception:
            # Re-encode without audio
            try:
                subprocess.run([
                    "ffmpeg", "-y", "-i", output_path,
                    "-c:v", "libx264", "-preset", "fast",
                    temp_path,
                ], capture_output=True, check=True)
                os.replace(temp_path, output_path)
            except Exception:
                pass

    return output_path


# =============================================
# Region-Based Editing
# =============================================

def create_mask_from_rect(width: int, height: int, x: int, y: int,
                          w: int, h: int, feather: int = 10) -> np.ndarray:
    """Create a soft-edged rectangular mask."""
    mask = np.zeros((height, width), dtype=np.float32)
    mask[max(0, y):min(height, y + h), max(0, x):min(width, x + w)] = 1.0

    if feather > 0:
        from scipy.ndimage import gaussian_filter
        mask = gaussian_filter(mask, sigma=feather)

    return mask


def apply_color_change(frame: np.ndarray, mask: np.ndarray,
                        target_color: tuple) -> np.ndarray:
    """Change the color of a masked region."""
    result = frame.copy().astype(np.float32)
    color_overlay = np.full_like(result, target_color, dtype=np.float32)
    mask_3d = np.stack([mask] * 3, axis=-1)
    result = result * (1 - mask_3d * 0.5) + color_overlay * (mask_3d * 0.5)
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_blur(frame: np.ndarray, mask: np.ndarray, strength: int = 21) -> np.ndarray:
    """Apply blur to a masked region."""
    blurred = cv2.GaussianBlur(frame, (strength, strength), 0)
    mask_3d = np.stack([mask] * 3, axis=-1)
    result = frame.astype(np.float32) * (1 - mask_3d) + blurred.astype(np.float32) * mask_3d
    return result.astype(np.uint8)


def apply_brightness(frame: np.ndarray, mask: np.ndarray, factor: float = 1.5) -> np.ndarray:
    """Adjust brightness of a masked region."""
    bright = np.clip(frame.astype(np.float32) * factor, 0, 255)
    mask_3d = np.stack([mask] * 3, axis=-1)
    result = frame.astype(np.float32) * (1 - mask_3d) + bright * mask_3d
    return result.astype(np.uint8)


def remove_object(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Remove an object from a frame using OpenCV inpainting.
    """
    mask_uint8 = (mask * 255).astype(np.uint8)
    result = cv2.inpaint(frame, mask_uint8, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
    return result


# =============================================
# Video Editing Operations
# =============================================

EDIT_OPERATIONS = {
    "remove": "Remove object from region (inpainting)",
    "blur": "Blur/censor a region",
    "brighten": "Brighten a region",
    "darken": "Darken a region",
    "color_shift": "Change color tint of a region",
    "grayscale": "Convert region to grayscale",
}


def edit_video_region(
    video_path: str,
    operation: str,
    x: int, y: int, w: int, h: int,
    start_frame: int = 0,
    end_frame: int = -1,
    progress_callback: Callable = None,
    **kwargs,
) -> str:
    """
    Apply an edit operation to a region of a video.

    Args:
        video_path: Input video path
        operation: One of EDIT_OPERATIONS keys
        x, y, w, h: Region rectangle
        start_frame: First frame to edit (0 = beginning)
        end_frame: Last frame to edit (-1 = all)
        progress_callback: callback(current, total, message)
        **kwargs: Operation-specific params (e.g., color=(255,0,0), factor=1.5)

    Returns:
        Path to edited video
    """
    ensure_output_dir()
    frames, fps = extract_frames(video_path)

    if not frames:
        raise ValueError("No frames extracted from video")

    if end_frame == -1:
        end_frame = len(frames)

    total = len(frames)
    edited_frames = []

    for i, frame_path in enumerate(frames):
        frame = cv2.imread(frame_path)

        if start_frame <= i < end_frame:
            mask = create_mask_from_rect(
                frame.shape[1], frame.shape[0],
                x, y, w, h, feather=kwargs.get("feather", 10)
            )

            if operation == "remove":
                frame = remove_object(frame, mask)
            elif operation == "blur":
                frame = apply_blur(frame, mask, kwargs.get("strength", 21))
            elif operation == "brighten":
                frame = apply_brightness(frame, mask, kwargs.get("factor", 1.5))
            elif operation == "darken":
                frame = apply_brightness(frame, mask, kwargs.get("factor", 0.5))
            elif operation == "color_shift":
                frame = apply_color_change(frame, mask, kwargs.get("color", (100, 100, 255)))
            elif operation == "grayscale":
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                mask_3d = np.stack([mask] * 3, axis=-1)
                frame = (frame.astype(np.float32) * (1 - mask_3d) +
                         gray_3ch.astype(np.float32) * mask_3d).astype(np.uint8)

        edited_path = frame_path.replace(".png", "_edited.png")
        cv2.imwrite(edited_path, frame)
        edited_frames.append(edited_path)

        if progress_callback and i % 5 == 0:
            progress_callback(i, total, f"Editing frame {i+1}/{total}")

    return frames_to_video(edited_frames, fps=fps, audio_from=video_path)


def apply_style_transfer(
    video_path: str,
    style: str = "grayscale",
    progress_callback: Callable = None,
) -> str:
    """
    Apply a visual style to the entire video.

    Styles: grayscale, sepia, invert, warm, cool, vintage, high_contrast
    """
    ensure_output_dir()
    frames, fps = extract_frames(video_path)
    total = len(frames)
    styled_frames = []

    for i, frame_path in enumerate(frames):
        frame = cv2.imread(frame_path)

        if style == "grayscale":
            frame = cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        elif style == "sepia":
            kernel = np.array([[0.272, 0.534, 0.131],
                               [0.349, 0.686, 0.168],
                               [0.393, 0.769, 0.189]])
            frame = cv2.transform(frame, kernel)
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        elif style == "invert":
            frame = 255 - frame
        elif style == "warm":
            frame = frame.astype(np.float32)
            frame[:, :, 2] = np.clip(frame[:, :, 2] * 1.2, 0, 255)  # Red up
            frame[:, :, 0] = np.clip(frame[:, :, 0] * 0.8, 0, 255)  # Blue down
            frame = frame.astype(np.uint8)
        elif style == "cool":
            frame = frame.astype(np.float32)
            frame[:, :, 0] = np.clip(frame[:, :, 0] * 1.2, 0, 255)  # Blue up
            frame[:, :, 2] = np.clip(frame[:, :, 2] * 0.8, 0, 255)  # Red down
            frame = frame.astype(np.uint8)
        elif style == "vintage":
            frame = frame.astype(np.float32)
            frame[:, :, 2] = np.clip(frame[:, :, 2] * 1.1 + 20, 0, 255)
            frame[:, :, 1] = np.clip(frame[:, :, 1] * 0.9, 0, 255)
            frame[:, :, 0] = np.clip(frame[:, :, 0] * 0.85 + 10, 0, 255)
            # Add slight vignette
            rows, cols = frame.shape[:2]
            X = np.arange(cols) - cols / 2
            Y = np.arange(rows) - rows / 2
            X, Y = np.meshgrid(X, Y)
            vignette = 1 - 0.3 * (X**2 + Y**2) / ((cols/2)**2 + (rows/2)**2)
            for c in range(3):
                frame[:, :, c] *= vignette
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        elif style == "high_contrast":
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            frame = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

        styled_path = frame_path.replace(".png", "_styled.png")
        cv2.imwrite(styled_path, frame)
        styled_frames.append(styled_path)

        if progress_callback and i % 5 == 0:
            progress_callback(i, total, f"Styling frame {i+1}/{total}")

    return frames_to_video(styled_frames, fps=fps, audio_from=video_path)


STYLE_FILTERS = ["grayscale", "sepia", "invert", "warm", "cool", "vintage", "high_contrast"]
