"""
Scene Transitions for StudioLite.

Provides crossfade, fade to black/white, wipe, zoom, and glitch
transitions between video clips.
"""

import os
import numpy as np
from uuid import uuid4
from typing import Callable, Optional, List, Tuple
from moviepy.editor import (
    VideoFileClip, CompositeVideoClip, ColorClip,
    concatenate_videoclips, vfx,
)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(ROOT_DIR, ".mp")


TRANSITION_TYPES = [
    "cut",
    "crossfade",
    "fade_to_black",
    "fade_to_white",
    "wipe_left",
    "wipe_right",
    "wipe_up",
    "wipe_down",
    "zoom",
    "glitch",
]

TRANSITION_DESCRIPTIONS = {
    "cut": "Hard cut (no transition)",
    "crossfade": "Gradual blend between scenes",
    "fade_to_black": "Fade out to black, fade in next scene",
    "fade_to_white": "Fade out to white, fade in next scene",
    "wipe_left": "New scene slides in from the right",
    "wipe_right": "New scene slides in from the left",
    "wipe_up": "New scene slides in from the bottom",
    "wipe_down": "New scene slides in from the top",
    "zoom": "Zoom into end, zoom out of start",
    "glitch": "Digital glitch effect between scenes",
}


def apply_transition(
    clip_a: VideoFileClip,
    clip_b: VideoFileClip,
    transition_type: str = "crossfade",
    duration: float = 1.0,
) -> CompositeVideoClip:
    """
    Apply a transition between two MoviePy clips.

    Args:
        clip_a: First clip
        clip_b: Second clip
        transition_type: Type of transition
        duration: Transition duration in seconds

    Returns:
        Combined clip with transition
    """
    duration = min(duration, clip_a.duration * 0.5, clip_b.duration * 0.5)

    if transition_type == "cut" or duration <= 0:
        return concatenate_videoclips([clip_a, clip_b], method="compose")

    elif transition_type == "crossfade":
        clip_b_shifted = clip_b.set_start(clip_a.duration - duration)
        clip_a_faded = clip_a.crossfadeout(duration)
        clip_b_faded = clip_b_shifted.crossfadein(duration)
        return CompositeVideoClip(
            [clip_a_faded, clip_b_faded],
            size=clip_a.size
        ).set_duration(clip_a.duration + clip_b.duration - duration)

    elif transition_type in ("fade_to_black", "fade_to_white"):
        color = (0, 0, 0) if "black" in transition_type else (255, 255, 255)
        half_dur = duration / 2

        clip_a_faded = clip_a.crossfadeout(half_dur)
        clip_b_faded = clip_b.crossfadein(half_dur)

        black = ColorClip(clip_a.size, color=color, duration=duration)
        black = black.set_start(clip_a.duration - half_dur)
        clip_b_shifted = clip_b_faded.set_start(clip_a.duration + half_dur)

        total_dur = clip_a.duration + clip_b.duration + duration - half_dur
        return CompositeVideoClip(
            [clip_a_faded, black, clip_b_shifted],
            size=clip_a.size,
        ).set_duration(total_dur)

    elif transition_type.startswith("wipe_"):
        direction = transition_type.replace("wipe_", "")
        w, h = clip_a.size

        def position_func(t):
            progress = min(t / duration, 1.0)
            if direction == "left":
                return (int(w * (1 - progress)), 0)
            elif direction == "right":
                return (int(-w * (1 - progress)), 0)
            elif direction == "up":
                return (0, int(h * (1 - progress)))
            elif direction == "down":
                return (0, int(-h * (1 - progress)))
            return (0, 0)

        clip_b_moving = clip_b.set_start(clip_a.duration - duration).set_position(position_func)
        total_dur = clip_a.duration + clip_b.duration - duration
        return CompositeVideoClip(
            [clip_a, clip_b_moving],
            size=clip_a.size,
        ).set_duration(total_dur)

    elif transition_type == "zoom":
        half_dur = duration / 2
        # Zoom into end of clip_a
        clip_a_end = clip_a.subclip(max(0, clip_a.duration - half_dur))
        clip_a_zoom = clip_a_end.resize(lambda t: 1 + (t / half_dur) * 0.5)
        clip_a_main = clip_a.subclip(0, max(0, clip_a.duration - half_dur))

        # Zoom out of start of clip_b
        clip_b_start = clip_b.subclip(0, min(half_dur, clip_b.duration))
        clip_b_zoom = clip_b_start.resize(lambda t: 1.5 - (t / half_dur) * 0.5)
        clip_b_rest = clip_b.subclip(min(half_dur, clip_b.duration))

        return concatenate_videoclips(
            [clip_a_main, clip_a_zoom.crossfadeout(0.2),
             clip_b_zoom.crossfadein(0.2), clip_b_rest],
            method="compose",
        )

    elif transition_type == "glitch":
        # Create glitch frames between clips
        from moviepy.editor import ImageClip

        last_frame = clip_a.get_frame(clip_a.duration - 0.01)
        first_frame = clip_b.get_frame(0)

        glitch_frames = []
        num_glitch = int(duration * clip_a.fps) if clip_a.fps else int(duration * 24)
        for i in range(max(num_glitch, 3)):
            t = i / max(num_glitch - 1, 1)
            # Mix frames with random offsets and color shifts
            frame = last_frame.copy() if t < 0.5 else first_frame.copy()
            # Random horizontal slice displacement
            h = frame.shape[0]
            for _ in range(np.random.randint(3, 8)):
                y = np.random.randint(0, h - 10)
                slice_h = np.random.randint(5, min(30, h - y))
                shift = np.random.randint(-20, 20)
                frame[y:y+slice_h] = np.roll(frame[y:y+slice_h], shift, axis=1)
            # Random color channel shift
            if np.random.random() > 0.5:
                channel = np.random.randint(0, 3)
                frame[:, :, channel] = np.roll(frame[:, :, channel], np.random.randint(-10, 10), axis=1)
            glitch_frames.append(frame)

        glitch_dur = duration / max(len(glitch_frames), 1)
        glitch_clips = [
            ImageClip(f).set_duration(glitch_dur) for f in glitch_frames
        ]
        glitch_video = concatenate_videoclips(glitch_clips, method="compose")

        return concatenate_videoclips(
            [clip_a, glitch_video, clip_b],
            method="compose",
        )

    # Default: crossfade
    return apply_transition(clip_a, clip_b, "crossfade", duration)


def concatenate_with_transitions(
    video_paths: list,
    transition_type: str = "crossfade",
    duration: float = 1.0,
    per_scene_transitions: list = None,
    fps: int = None,
    output_path: str = None,
    progress_callback: Callable = None,
) -> str:
    """
    Concatenate multiple videos with transitions.

    Args:
        video_paths: List of video file paths
        transition_type: Default transition type
        duration: Default transition duration
        per_scene_transitions: Optional list of (type, duration) per pair.
                               Length should be len(video_paths) - 1.
        fps: Output FPS (uses first clip's if None)
        output_path: Output file path
        progress_callback: callback(step, total, message)

    Returns:
        Path to output video with transitions
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = output_path or os.path.join(OUTPUT_DIR, f"transition_{uuid4()}.mp4")

    if not video_paths:
        raise ValueError("No video paths provided")

    if len(video_paths) == 1:
        import shutil
        shutil.copy2(video_paths[0], output_path)
        return output_path

    clips = [VideoFileClip(p) for p in video_paths]
    target_fps = fps or clips[0].fps or 24

    try:
        # Build combined clip pair by pair
        result = clips[0]
        for i in range(1, len(clips)):
            if progress_callback:
                progress_callback(i, len(clips), f"Applying transition {i}/{len(clips)-1}")

            t_type = transition_type
            t_dur = duration
            if per_scene_transitions and i - 1 < len(per_scene_transitions):
                pair = per_scene_transitions[i - 1]
                t_type = pair[0] if pair[0] else transition_type
                t_dur = pair[1] if len(pair) > 1 and pair[1] else duration

            result = apply_transition(result, clips[i], t_type, t_dur)

        result.write_videofile(
            output_path,
            fps=target_fps,
            codec="libx264",
            audio_codec="aac" if result.audio else None,
            logger=None,
        )
    finally:
        for c in clips:
            c.close()
        if result:
            result.close()

    return output_path
