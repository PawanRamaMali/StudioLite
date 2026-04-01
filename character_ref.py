"""
Character Consistency System for StudioLite.

Manages persistent character references across scenes using visual identity
descriptions and reference images for consistent subject appearance.
"""

import os
import json
import shutil
from uuid import uuid4
from pathlib import Path
from typing import Optional

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CHARACTERS_DIR = os.path.join(ROOT_DIR, "characters")


def ensure_characters_dir():
    """Ensure the characters directory exists."""
    os.makedirs(CHARACTERS_DIR, exist_ok=True)


def create_character(
    name: str,
    description: str,
    visual_prompt: str,
    reference_images: list = None,
) -> dict:
    """
    Create a new character profile.

    Args:
        name: Character name
        description: Brief character description
        visual_prompt: Detailed visual description for prompt injection
        reference_images: List of image file paths to copy as references

    Returns:
        Character dict with id, name, description, visual_prompt, images
    """
    ensure_characters_dir()
    char_id = str(uuid4())[:8]
    char_dir = os.path.join(CHARACTERS_DIR, char_id)
    os.makedirs(char_dir, exist_ok=True)

    # Copy reference images
    image_paths = []
    if reference_images:
        for img_path in reference_images:
            if os.path.exists(img_path):
                ext = os.path.splitext(img_path)[1]
                dest = os.path.join(char_dir, f"ref_{len(image_paths)}{ext}")
                shutil.copy2(img_path, dest)
                image_paths.append(dest)

    character = {
        "id": char_id,
        "name": name,
        "description": description,
        "visual_prompt": visual_prompt,
        "images": image_paths,
        "created_at": __import__("time").time(),
    }

    # Save metadata
    meta_path = os.path.join(char_dir, "character.json")
    with open(meta_path, "w") as f:
        json.dump(character, f, indent=2)

    return character


def load_character(char_id: str) -> Optional[dict]:
    """Load a character profile by ID."""
    meta_path = os.path.join(CHARACTERS_DIR, char_id, "character.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            return json.load(f)
    return None


def list_characters() -> list:
    """List all saved character profiles."""
    ensure_characters_dir()
    characters = []
    for entry in os.listdir(CHARACTERS_DIR):
        char_dir = os.path.join(CHARACTERS_DIR, entry)
        if os.path.isdir(char_dir):
            meta = os.path.join(char_dir, "character.json")
            if os.path.exists(meta):
                with open(meta) as f:
                    characters.append(json.load(f))
    return sorted(characters, key=lambda c: c.get("created_at", 0), reverse=True)


def delete_character(char_id: str) -> bool:
    """Delete a character profile."""
    char_dir = os.path.join(CHARACTERS_DIR, char_id)
    if os.path.exists(char_dir):
        shutil.rmtree(char_dir)
        return True
    return False


def update_character(char_id: str, **kwargs) -> Optional[dict]:
    """Update character fields."""
    char = load_character(char_id)
    if not char:
        return None
    char.update(kwargs)
    meta_path = os.path.join(CHARACTERS_DIR, char_id, "character.json")
    with open(meta_path, "w") as f:
        json.dump(char, f, indent=2)
    return char


def inject_character_prompt(prompt: str, character: dict) -> str:
    """
    Inject character visual description into a generation prompt.
    This ensures the character appears consistently in the output.
    """
    visual = character.get("visual_prompt", "")
    if not visual:
        return prompt
    return f"{prompt}, {visual}"


def inject_multiple_characters(prompt: str, characters: list) -> str:
    """Inject multiple character descriptions into a prompt."""
    for char in characters:
        prompt = inject_character_prompt(prompt, char)
    return prompt


def get_character_reference_image(character: dict) -> Optional[str]:
    """Get the primary reference image for a character (for I2V)."""
    images = character.get("images", [])
    for img in images:
        if os.path.exists(img):
            return img
    return None


def generate_character_description(name: str, traits: str, generate_text_fn=None) -> str:
    """
    Use LLM to generate a detailed visual description for a character.

    Args:
        name: Character name
        traits: Brief trait description (e.g., "young woman, red hair, detective")
        generate_text_fn: LLM text generation function

    Returns:
        Detailed visual prompt for consistent generation
    """
    if not generate_text_fn:
        return traits

    prompt = f"""Create a detailed visual description of a character for AI image/video generation.
The description must be specific enough that the character looks the SAME every time it's generated.

Character Name: {name}
Traits: {traits}

Write a single paragraph describing:
- Exact physical appearance (hair color/style, eye color, skin tone, build, height)
- Clothing (specific items, colors, patterns)
- Distinguishing features (scars, accessories, posture)
- Overall vibe/energy

Be very specific with colors and details. Use image generation prompt style.
Return ONLY the visual description paragraph, nothing else."""

    try:
        description = generate_text_fn(prompt)
        return description.strip()
    except Exception:
        return traits
