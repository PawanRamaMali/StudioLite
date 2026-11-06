"""Prompt templates and style presets.

Two preset types:
- ``style``  : visual look. positive_prompt adds cinematic grading /
  lens / grain cues; negative_prompt kills generic failure modes.
- ``genre``  : mood / narrative cues. Tends to pair with a style preset.

Built-in presets ship inside this module as a Python dict; users save
their own as JSON files in ``PRESETS_DIR``. ``PRESETS_DIR`` resolves to
``<cwd>/.mp/user_presets`` (writable, gitignored) by default - override
with ``STUDIOLITE_PRESETS_DIR``.

Covers issue #36.
"""
from __future__ import annotations

import json
import os
import re
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PRESETS_DIR = os.environ.get(
    "STUDIOLITE_PRESETS_DIR",
    os.path.join(ROOT_DIR, ".mp", "user_presets"),
)

_SAFE_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


# ---------------------------------------------------------------------------
# Data shape
# ---------------------------------------------------------------------------


@dataclass
class Preset:
    id: str
    name: str
    category: str   # style | genre | shot | lighting | custom
    description: str = ""
    positive_prompt: str = ""
    negative_prompt: str = ""
    recommended_settings: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    builtin: bool = False

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


# ---------------------------------------------------------------------------
# Built-in library
# ---------------------------------------------------------------------------


_NEG_COMMON = (
    "blurry, low quality, watermark, text, logo, extra fingers, "
    "deformed hands, bad anatomy"
)


_BUILTINS: List[Preset] = [
    # --- Style (visual) ---
    Preset(
        id="style_cinematic",
        name="Cinematic Film",
        category="style",
        description="35mm film grain, anamorphic lens flares, warm color grading",
        positive_prompt="cinematic, 35mm film grain, anamorphic lens flare, "
                        "shallow depth of field, warm color grading, high dynamic range",
        negative_prompt=_NEG_COMMON + ", flat lighting, washed out",
        recommended_settings={"guidance_scale": 7.5, "num_steps": 30},
        tags=["film", "warm", "shallow"],
        builtin=True,
    ),
    Preset(
        id="style_anime",
        name="Anime",
        category="style",
        description="Japanese animation style, vibrant colors, clean lines",
        positive_prompt="anime style, vibrant colors, clean lines, cel shading, "
                        "detailed background, expressive eyes",
        negative_prompt=_NEG_COMMON + ", photorealistic, 3d render",
        recommended_settings={"guidance_scale": 8.0, "num_steps": 28},
        tags=["anime", "illustration"],
        builtin=True,
    ),
    Preset(
        id="style_photorealistic",
        name="Photorealistic",
        category="style",
        description="Ultra-realistic, RAW photo quality",
        positive_prompt="photorealistic, RAW photo, ultra detailed, 8k, "
                        "natural lighting, sharp focus, DSLR",
        negative_prompt=_NEG_COMMON + ", cartoon, painting, illustration, anime",
        recommended_settings={"guidance_scale": 6.5, "num_steps": 35},
        tags=["realistic", "photo"],
        builtin=True,
    ),
    Preset(
        id="style_noir",
        name="Film Noir",
        category="style",
        description="High contrast B&W, dramatic shadows, dutch angles",
        positive_prompt="film noir, high contrast black and white, dramatic chiaroscuro, "
                        "venetian blind shadows, dutch angle, 1940s styling",
        negative_prompt=_NEG_COMMON + ", colorful, bright, modern",
        recommended_settings={"guidance_scale": 7.0, "num_steps": 30},
        tags=["noir", "monochrome"],
        builtin=True,
    ),
    Preset(
        id="style_dreamlike",
        name="Dreamlike",
        category="style",
        description="Soft focus, ethereal lighting, pastel palette",
        positive_prompt="dreamlike, soft focus, ethereal lighting, pastel palette, "
                        "bokeh, glowing edges, hazy atmosphere",
        negative_prompt=_NEG_COMMON + ", sharp, high contrast",
        recommended_settings={"guidance_scale": 7.5, "num_steps": 32},
        tags=["soft", "ethereal"],
        builtin=True,
    ),
    Preset(
        id="style_vintage80s",
        name="Vintage 80s",
        category="style",
        description="VHS grain, saturated colors, retro palette",
        positive_prompt="80s aesthetic, VHS grain, saturated colors, neon pink and cyan, "
                        "retro-futurism, chromatic aberration",
        negative_prompt=_NEG_COMMON + ", modern, clean, digital crisp",
        recommended_settings={"guidance_scale": 7.5, "num_steps": 28},
        tags=["80s", "retro", "neon"],
        builtin=True,
    ),
    Preset(
        id="style_cyberpunk",
        name="Cyberpunk",
        category="style",
        description="Neon lighting, rain-slicked streets, holographic UI",
        positive_prompt="cyberpunk, neon signs, rain-slicked streets, holographic UI, "
                        "wet reflective pavement, moody atmosphere, Blade Runner inspired",
        negative_prompt=_NEG_COMMON + ", pastoral, daylight, rural",
        recommended_settings={"guidance_scale": 8.0, "num_steps": 32},
        tags=["cyberpunk", "neon"],
        builtin=True,
    ),
    Preset(
        id="style_watercolor",
        name="Watercolor",
        category="style",
        description="Soft wash textures, flowing paint effects",
        positive_prompt="watercolor painting, soft wash textures, flowing paint, "
                        "visible brush strokes, paper texture, muted color palette",
        negative_prompt=_NEG_COMMON + ", photograph, crisp, digital",
        recommended_settings={"guidance_scale": 7.0, "num_steps": 30},
        tags=["watercolor", "painting"],
        builtin=True,
    ),

    # --- Genre ---
    Preset(
        id="genre_scifi",
        name="Sci-Fi",
        category="genre",
        description="Starships, holograms, zero-gravity choreography",
        positive_prompt="science fiction, futuristic technology, starship interior, "
                        "holographic displays, zero gravity, advanced architecture",
        negative_prompt=_NEG_COMMON,
        tags=["scifi", "futuristic"],
        builtin=True,
    ),
    Preset(
        id="genre_horror",
        name="Horror",
        category="genre",
        description="Low key lighting, unsettling composition, muted palette",
        positive_prompt="horror atmosphere, low key lighting, dim corridors, "
                        "unsettling composition, muted palette, ominous shadows",
        negative_prompt=_NEG_COMMON + ", bright, cheerful, saturated",
        tags=["horror", "dark"],
        builtin=True,
    ),
    Preset(
        id="genre_documentary",
        name="Documentary",
        category="genre",
        description="Handheld feel, natural light, observational framing",
        positive_prompt="documentary style, handheld camera, natural light, "
                        "observational framing, candid moment",
        negative_prompt=_NEG_COMMON + ", staged, over-lit, posed",
        tags=["doc", "handheld"],
        builtin=True,
    ),
    Preset(
        id="genre_romance",
        name="Romance",
        category="genre",
        description="Soft golden hour light, intimate close-ups",
        positive_prompt="romantic, golden hour lighting, soft bokeh, "
                        "intimate close-up, warm skin tones, tender gaze",
        negative_prompt=_NEG_COMMON,
        tags=["romance", "golden"],
        builtin=True,
    ),
    Preset(
        id="genre_fantasy",
        name="Fantasy",
        category="genre",
        description="Mythic landscapes, magical glow, grand scale",
        positive_prompt="high fantasy, mythic landscape, magical glow, "
                        "ancient ruins, towering mountains, misty valleys",
        negative_prompt=_NEG_COMMON + ", modern, urban",
        tags=["fantasy"],
        builtin=True,
    ),
    Preset(
        id="genre_western",
        name="Western",
        category="genre",
        description="Sun-bleached desert, lone silhouette, saloon interior",
        positive_prompt="western, sun-bleached desert, dust and tumbleweed, "
                        "lone silhouette at horizon, saloon interior, 1870s",
        negative_prompt=_NEG_COMMON + ", urban, modern technology",
        tags=["western"],
        builtin=True,
    ),

    # --- Shot ---
    Preset(
        id="shot_establishing",
        name="Establishing shot",
        category="shot",
        description="Wide composition revealing the location",
        positive_prompt="establishing shot, wide composition, revealing the location, "
                        "high vantage point, scene-setting",
        negative_prompt=_NEG_COMMON,
        tags=["shot"],
        builtin=True,
    ),
    Preset(
        id="shot_closeup",
        name="Close-up",
        category="shot",
        description="Face fills frame, shallow depth of field",
        positive_prompt="extreme close-up, face fills frame, "
                        "shallow depth of field, intense emotion",
        negative_prompt=_NEG_COMMON,
        tags=["shot"],
        builtin=True,
    ),
    Preset(
        id="shot_tracking",
        name="Tracking shot",
        category="shot",
        description="Smooth dolly or gimbal following the subject",
        positive_prompt="tracking shot, smooth gimbal following the subject, "
                        "continuous motion, immersive flow",
        negative_prompt=_NEG_COMMON,
        tags=["shot"],
        builtin=True,
    ),
    Preset(
        id="shot_aerial",
        name="Aerial",
        category="shot",
        description="Drone overhead, revealing scale",
        positive_prompt="aerial drone shot, high altitude, revealing scale, "
                        "sweeping landscape, top-down perspective",
        negative_prompt=_NEG_COMMON,
        tags=["shot"],
        builtin=True,
    ),

    # --- Lighting ---
    Preset(
        id="lighting_golden",
        name="Golden hour",
        category="lighting",
        description="Low-angle warm sunlight, long shadows",
        positive_prompt="golden hour, low-angle warm sunlight, long shadows, "
                        "amber tones, rim lighting",
        negative_prompt=_NEG_COMMON + ", overhead sun, flat lighting",
        tags=["lighting", "warm"],
        builtin=True,
    ),
    Preset(
        id="lighting_blue",
        name="Blue hour",
        category="lighting",
        description="Twilight blues, soft diffuse light",
        positive_prompt="blue hour, twilight, soft diffuse light, "
                        "cool blue tones, illuminated windows",
        negative_prompt=_NEG_COMMON + ", harsh sunlight",
        tags=["lighting", "blue"],
        builtin=True,
    ),
    Preset(
        id="lighting_neon",
        name="Neon noir",
        category="lighting",
        description="Saturated signage, reflections, rainy streets",
        positive_prompt="neon noir lighting, saturated signage, "
                        "reflections on wet pavement, high contrast",
        negative_prompt=_NEG_COMMON + ", daylight, pastoral",
        tags=["lighting", "neon"],
        builtin=True,
    ),
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def list_builtin_presets() -> List[Dict[str, Any]]:
    return [p.to_dict() for p in _BUILTINS]


def list_builtin_categories() -> List[str]:
    seen = []
    for p in _BUILTINS:
        if p.category not in seen:
            seen.append(p.category)
    return seen


def _ensure_dir() -> None:
    os.makedirs(PRESETS_DIR, exist_ok=True)


def _preset_path(preset_id: str) -> Optional[str]:
    """Resolve a user preset id to its JSON path. Returns None if the id
    doesn't pass the safe-id regex or escapes the presets dir - either
    case stops a caller from reading an arbitrary file."""
    if not _SAFE_ID_RE.match(preset_id):
        return None
    target = os.path.abspath(os.path.join(PRESETS_DIR, f"{preset_id}.json"))
    root = os.path.abspath(PRESETS_DIR)
    if not target.startswith(root + os.sep) and target != root:
        return None
    return target


def list_user_presets() -> List[Dict[str, Any]]:
    _ensure_dir()
    out: List[Dict[str, Any]] = []
    for name in sorted(os.listdir(PRESETS_DIR)):
        if not name.endswith(".json"):
            continue
        p = os.path.join(PRESETS_DIR, name)
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and "id" in data and "name" in data:
                data.setdefault("builtin", False)
                out.append(data)
        except (OSError, json.JSONDecodeError):
            # Skip malformed entries; don't fail the whole listing because
            # of one bad file. Admin can clean it up out-of-band.
            continue
    return out


def _normalize_preset_input(data: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce user input into a Preset-compatible dict. Rejects anything
    whose essentials are missing by raising ValueError."""
    if not isinstance(data, dict):
        raise ValueError("Preset must be a JSON object")
    name = (data.get("name") or "").strip()
    if not name:
        raise ValueError("Preset.name is required")
    category = (data.get("category") or "custom").strip() or "custom"
    pid = (data.get("id") or "").strip()
    if not pid:
        slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
        pid = f"custom_{slug[:40]}_{uuid.uuid4().hex[:6]}"
    if not _SAFE_ID_RE.match(pid):
        raise ValueError("Preset.id must match [a-zA-Z0-9_-]{1,64}")

    out = {
        "id": pid,
        "name": name[:120],
        "category": category[:40],
        "description": str(data.get("description") or "")[:500],
        "positive_prompt": str(data.get("positive_prompt") or "")[:4000],
        "negative_prompt": str(data.get("negative_prompt") or "")[:4000],
        "recommended_settings": data.get("recommended_settings") or {},
        "tags": [str(t)[:40] for t in (data.get("tags") or [])][:16],
        "builtin": False,
    }
    if not isinstance(out["recommended_settings"], dict):
        out["recommended_settings"] = {}
    return out


def save_user_preset(data: Dict[str, Any]) -> Dict[str, Any]:
    _ensure_dir()
    norm = _normalize_preset_input(data)
    path = _preset_path(norm["id"])
    if path is None:
        raise ValueError("Preset.id resolves outside the presets directory")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(norm, f, indent=2, ensure_ascii=False)
    return norm


def delete_user_preset(preset_id: str) -> bool:
    path = _preset_path(preset_id)
    if path is None or not os.path.isfile(path):
        return False
    os.remove(path)
    return True


def get_preset(preset_id: str) -> Optional[Dict[str, Any]]:
    for p in _BUILTINS:
        if p.id == preset_id:
            return p.to_dict()
    path = _preset_path(preset_id)
    if path is None or not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data.setdefault("builtin", False)
            return data
    except (OSError, json.JSONDecodeError):
        return None
    return None
