"""
Motion Brush for StudioLite.

Allows users to define motion regions with independent motion parameters
for video generation. Converts region masks + motion params into
conditioning prompts and generation settings.
"""

import os
import numpy as np
from uuid import uuid4
from PIL import Image, ImageDraw
from typing import List, Optional

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(ROOT_DIR, ".mp")


MOTION_TYPES = {
    "static": {"prompt": "completely still, frozen, no movement", "strength": 0.0},
    "gentle_sway": {"prompt": "gentle swaying motion, subtle movement", "strength": 0.2},
    "flowing": {"prompt": "smooth flowing motion, fluid movement", "strength": 0.5},
    "fast_motion": {"prompt": "fast dynamic motion, energetic movement", "strength": 0.8},
    "wind": {"prompt": "blowing in the wind, wind-swept motion", "strength": 0.4},
    "water": {"prompt": "flowing water, rippling, wave motion", "strength": 0.6},
    "rotation": {"prompt": "slow rotation, spinning, turning", "strength": 0.5},
    "zoom": {"prompt": "approaching camera, moving forward", "strength": 0.3},
    "vibration": {"prompt": "slight vibration, trembling, shaking", "strength": 0.3},
}

# Colors for up to 5 motion regions
REGION_COLORS = [
    (255, 107, 107, 128),  # Red
    (78, 205, 196, 128),   # Teal
    (69, 183, 209, 128),   # Blue
    (150, 206, 180, 128),  # Green
    (255, 234, 167, 128),  # Yellow
]


def create_motion_region(
    x: int, y: int, width: int, height: int,
    motion_type: str = "gentle_sway",
    direction: str = "none",
    strength: float = 0.5,
) -> dict:
    """Create a motion region definition."""
    return {
        "x": x, "y": y, "width": width, "height": height,
        "motion_type": motion_type,
        "direction": direction,
        "strength": strength,
    }


def regions_to_prompt(base_prompt: str, regions: List[dict], image_width: int, image_height: int) -> str:
    """
    Convert motion regions into an enhanced prompt that describes
    the desired motion for each area of the image.

    Since most video generation models don't support spatial motion masks directly,
    we encode the motion information as descriptive text in the prompt.
    """
    if not regions:
        return base_prompt

    motion_descriptions = []
    for i, region in enumerate(regions):
        # Calculate relative position
        cx = (region["x"] + region["width"] / 2) / image_width
        cy = (region["y"] + region["height"] / 2) / image_height

        # Describe position in natural language
        h_pos = "left" if cx < 0.33 else "right" if cx > 0.66 else "center"
        v_pos = "top" if cy < 0.33 else "bottom" if cy > 0.66 else "middle"
        position = f"{v_pos} {h_pos}" if v_pos != "middle" or h_pos != "center" else "center"

        motion_info = MOTION_TYPES.get(region["motion_type"], MOTION_TYPES["gentle_sway"])
        direction = region.get("direction", "")
        dir_text = f" moving {direction}" if direction and direction != "none" else ""

        motion_descriptions.append(
            f"the {position} area has {motion_info['prompt']}{dir_text}"
        )

    motion_text = ", ".join(motion_descriptions)
    return f"{base_prompt}, where {motion_text}"


def create_motion_mask(
    regions: List[dict],
    image_width: int,
    image_height: int,
    output_path: str = None,
) -> str:
    """
    Create a grayscale motion mask image from regions.
    White = full motion, Black = static.
    Can be used as conditioning for ControlNet motion modules.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = output_path or os.path.join(OUTPUT_DIR, f"motion_mask_{uuid4()}.png")

    mask = Image.new("L", (image_width, image_height), 0)
    draw = ImageDraw.Draw(mask)

    for region in regions:
        strength = region.get("strength", 0.5)
        gray_value = int(strength * 255)
        draw.rectangle(
            [region["x"], region["y"],
             region["x"] + region["width"], region["y"] + region["height"]],
            fill=gray_value,
        )

    mask.save(output_path)
    return output_path


def create_region_overlay(
    base_image_path: str,
    regions: List[dict],
    output_path: str = None,
) -> str:
    """
    Create a visualization of motion regions overlaid on the base image.
    Used for preview in the UI.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = output_path or os.path.join(OUTPUT_DIR, f"region_overlay_{uuid4()}.png")

    base = Image.open(base_image_path).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for i, region in enumerate(regions):
        color = REGION_COLORS[i % len(REGION_COLORS)]
        draw.rectangle(
            [region["x"], region["y"],
             region["x"] + region["width"], region["y"] + region["height"]],
            fill=color,
            outline=color[:3] + (255,),
            width=2,
        )
        # Label
        motion_type = region.get("motion_type", "gentle_sway")
        draw.text((region["x"] + 4, region["y"] + 4), f"R{i+1}: {motion_type}",
                  fill=(255, 255, 255, 255))

    result = Image.alpha_composite(base, overlay)
    result.convert("RGB").save(output_path)
    return output_path


def get_motion_types() -> dict:
    """Get available motion types."""
    return dict(MOTION_TYPES)
