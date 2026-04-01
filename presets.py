"""
Prompt Templates & Style Presets for StudioLite.

Reusable prompt library for consistent video generation with
style presets, shot templates, lighting templates, genre recipes,
and camera movement prompts.
"""

import os
import json
from pathlib import Path

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PRESETS_DIR = os.path.join(ROOT_DIR, "presets")


# =============================================
# Style Presets
# =============================================

STYLE_PRESETS = {
    "cinematic_film": {
        "name": "Cinematic Film",
        "icon": "🎬",
        "description": "35mm film look with dramatic lighting and color grading",
        "positive": "cinematic, 35mm film grain, anamorphic lens, shallow depth of field, dramatic lighting, professional color grading, movie scene, high production value",
        "negative": "amateur, low quality, phone camera, flat lighting, oversaturated",
        "guidance_scale": 6.0,
    },
    "anime": {
        "name": "Anime",
        "icon": "🎌",
        "description": "Japanese animation style with vibrant colors",
        "positive": "anime style, Japanese animation, vibrant colors, clean lines, cel shading, detailed background, Studio Ghibli inspired",
        "negative": "realistic, photographic, 3D render, low quality, blurry",
        "guidance_scale": 7.0,
    },
    "photorealistic": {
        "name": "Photorealistic",
        "icon": "📷",
        "description": "Ultra-realistic photography quality",
        "positive": "photorealistic, ultra detailed, RAW photo, 8K UHD, DSLR quality, natural lighting, sharp focus, professional photography",
        "negative": "illustration, painting, cartoon, anime, CGI, artificial",
        "guidance_scale": 5.0,
    },
    "film_noir": {
        "name": "Film Noir",
        "icon": "🕵️",
        "description": "High contrast black and white with dramatic shadows",
        "positive": "film noir, black and white, high contrast, dramatic shadows, venetian blind lighting, detective aesthetic, 1940s style, moody",
        "negative": "colorful, bright, cheerful, modern, low contrast",
        "guidance_scale": 6.5,
    },
    "dreamlike": {
        "name": "Dreamlike",
        "icon": "💭",
        "description": "Ethereal, soft focus with pastel palette",
        "positive": "dreamlike, ethereal, soft focus, pastel colors, glowing light, surreal, magical atmosphere, lens flare, bokeh",
        "negative": "harsh, sharp, realistic, dark, gritty, high contrast",
        "guidance_scale": 7.0,
    },
    "vintage_80s": {
        "name": "Vintage 80s",
        "icon": "📼",
        "description": "VHS aesthetic with retro color palette",
        "positive": "1980s retro, VHS grain, scan lines, neon colors, synthwave aesthetic, retro futurism, vintage television, warm tones",
        "negative": "modern, clean, 4K, sharp, minimalist",
        "guidance_scale": 5.5,
    },
    "cyberpunk": {
        "name": "Cyberpunk",
        "icon": "🌆",
        "description": "Neon-lit futuristic urban aesthetic",
        "positive": "cyberpunk, neon lights, rain-slicked streets, holographic displays, futuristic city, dark atmosphere, pink and blue neon, blade runner style",
        "negative": "natural, pastoral, bright daylight, historical, rustic",
        "guidance_scale": 6.0,
    },
    "watercolor": {
        "name": "Watercolor",
        "icon": "🎨",
        "description": "Soft watercolor painting style",
        "positive": "watercolor painting, soft washes, flowing paint, artistic, paper texture, loose brushstrokes, pastel tones, hand-painted",
        "negative": "photorealistic, digital, sharp, 3D, high contrast, CGI",
        "guidance_scale": 7.0,
    },
    "oil_painting": {
        "name": "Oil Painting",
        "icon": "🖼️",
        "description": "Classical oil painting with rich textures",
        "positive": "oil painting, thick brushstrokes, canvas texture, rich colors, classical art, impasto technique, museum quality, Renaissance lighting",
        "negative": "digital, clean, modern, photographic, minimalist",
        "guidance_scale": 6.5,
    },
    "3d_render": {
        "name": "3D Render",
        "icon": "🔮",
        "description": "Polished 3D CGI render",
        "positive": "3D render, CGI, Pixar style, clean geometry, subsurface scattering, global illumination, octane render, unreal engine 5",
        "negative": "2D, flat, hand-drawn, painting, photographic, grainy",
        "guidance_scale": 5.5,
    },
    "documentary": {
        "name": "Documentary",
        "icon": "📹",
        "description": "Natural documentary filmmaking style",
        "positive": "documentary style, natural lighting, handheld camera, authentic, observational, National Geographic quality, real-world setting",
        "negative": "stylized, fantasy, CGI, artificial lighting, staged",
        "guidance_scale": 4.5,
    },
    "horror": {
        "name": "Horror",
        "icon": "👻",
        "description": "Dark, unsettling atmospheric horror",
        "positive": "horror atmosphere, dark shadows, unsettling, eerie lighting, fog, desaturated colors, tense mood, creepy, dimly lit",
        "negative": "bright, cheerful, colorful, sunny, happy, comedic",
        "guidance_scale": 6.0,
    },
    "minimalist": {
        "name": "Minimalist",
        "icon": "◻️",
        "description": "Clean, simple design with negative space",
        "positive": "minimalist, clean design, white space, simple composition, modern, elegant, uncluttered, geometric, subtle",
        "negative": "busy, cluttered, detailed, ornate, complex, noisy",
        "guidance_scale": 5.0,
    },
    "fantasy_epic": {
        "name": "Fantasy Epic",
        "icon": "⚔️",
        "description": "Epic high fantasy with magical elements",
        "positive": "epic fantasy, Lord of the Rings style, magical, grand scale, dramatic landscape, mystical lighting, medieval, enchanted",
        "negative": "modern, urban, realistic, mundane, small scale, sci-fi",
        "guidance_scale": 7.0,
    },
    "stop_motion": {
        "name": "Stop Motion",
        "icon": "🧸",
        "description": "Claymation/stop motion animation look",
        "positive": "stop motion animation, claymation, handcrafted, miniature sets, slightly jerky movement, Wes Anderson style, tactile textures",
        "negative": "smooth, digital, photorealistic, CGI, clean, sharp",
        "guidance_scale": 6.0,
    },
}


# =============================================
# Shot Templates
# =============================================

SHOT_TEMPLATES = {
    "establishing": {
        "name": "Establishing Shot",
        "description": "Wide shot showing the full setting/location",
        "prompt": "wide establishing shot, showing the full scene, distant view, environmental context, cinematic wide angle",
    },
    "close_up": {
        "name": "Close-Up",
        "description": "Tight frame on subject's face or detail",
        "prompt": "extreme close-up shot, detailed, tight framing, shallow depth of field, intimate perspective",
    },
    "medium": {
        "name": "Medium Shot",
        "description": "Subject from waist up",
        "prompt": "medium shot, waist-up framing, conversational distance, balanced composition",
    },
    "over_shoulder": {
        "name": "Over-the-Shoulder",
        "description": "Camera behind one subject looking at another",
        "prompt": "over-the-shoulder shot, perspective view, foreground subject blurred, depth",
    },
    "tracking": {
        "name": "Tracking Shot",
        "description": "Camera follows subject movement",
        "prompt": "tracking shot, camera following the subject, smooth movement, dynamic framing",
    },
    "aerial": {
        "name": "Aerial / Drone",
        "description": "Bird's eye view from above",
        "prompt": "aerial drone shot, bird's eye view, looking down, sweeping landscape, high altitude perspective",
    },
    "low_angle": {
        "name": "Low Angle Hero",
        "description": "Looking up at subject for power/drama",
        "prompt": "low angle shot, looking upward, heroic perspective, dramatic, powerful, imposing",
    },
    "dutch_angle": {
        "name": "Dutch Angle",
        "description": "Tilted camera for unease/tension",
        "prompt": "dutch angle, tilted camera, off-kilter framing, tension, disorientation, dramatic",
    },
    "pov": {
        "name": "POV / First Person",
        "description": "From subject's point of view",
        "prompt": "point of view shot, first person perspective, subjective camera, immersive",
    },
    "slow_motion": {
        "name": "Slow Motion",
        "description": "Time-stretched dramatic moment",
        "prompt": "slow motion, time slowed down, dramatic impact, every detail visible, high frame rate cinematic",
    },
}


# =============================================
# Lighting Templates
# =============================================

LIGHTING_TEMPLATES = {
    "golden_hour": {
        "name": "Golden Hour",
        "description": "Warm sunset/sunrise lighting",
        "prompt": "golden hour lighting, warm orange tones, long shadows, soft directional light, magic hour",
    },
    "blue_hour": {
        "name": "Blue Hour",
        "description": "Cool twilight ambient light",
        "prompt": "blue hour lighting, cool blue tones, twilight, soft ambient light, serene atmosphere",
    },
    "neon_noir": {
        "name": "Neon Noir",
        "description": "Colorful neon with dark shadows",
        "prompt": "neon lighting, colorful neon signs reflected on wet surfaces, noir shadows, urban night, pink and blue neon glow",
    },
    "studio_3point": {
        "name": "Studio Three-Point",
        "description": "Professional studio lighting setup",
        "prompt": "professional three-point lighting, key light, fill light, rim light, studio quality, even illumination",
    },
    "natural_daylight": {
        "name": "Natural Daylight",
        "description": "Bright, natural outdoor lighting",
        "prompt": "natural daylight, bright sun, clear sky, natural colors, outdoor lighting, no artificial light",
    },
    "moonlit": {
        "name": "Moonlit",
        "description": "Silvery moonlight night scene",
        "prompt": "moonlit scene, silver moonlight, cool blue tones, night atmosphere, subtle illumination, starry sky",
    },
    "candlelight": {
        "name": "Candlelight",
        "description": "Warm, flickering candlelight",
        "prompt": "candlelight, warm flickering glow, intimate atmosphere, orange tones, soft shadows, romantic",
    },
    "dramatic_backlight": {
        "name": "Dramatic Backlight",
        "description": "Strong light behind subject creating silhouette",
        "prompt": "dramatic backlighting, silhouette, rim lighting, lens flare, halo effect, high contrast",
    },
    "foggy": {
        "name": "Foggy / Misty",
        "description": "Atmospheric fog diffusing light",
        "prompt": "foggy atmosphere, mist, diffused light, low visibility, mysterious, volumetric lighting, god rays",
    },
    "underwater": {
        "name": "Underwater Caustics",
        "description": "Dappled underwater light patterns",
        "prompt": "underwater lighting, caustic light patterns, blue-green tones, rippling light, submerged, aquatic atmosphere",
    },
}


# =============================================
# Genre Templates
# =============================================

GENRE_TEMPLATES = {
    "sci_fi": {
        "name": "Sci-Fi",
        "icon": "🚀",
        "positive": "science fiction, futuristic technology, space, advanced civilization, sleek design, holographic interfaces",
        "negative": "medieval, historical, rustic, natural, low-tech",
        "mood": "mysterious and awe-inspiring",
    },
    "horror": {
        "name": "Horror",
        "icon": "🎃",
        "positive": "horror, dark atmosphere, unsettling, suspenseful, shadows, eerie, tension, foreboding",
        "negative": "bright, cheerful, comedic, sunny, happy, relaxing",
        "mood": "tense and frightening",
    },
    "romance": {
        "name": "Romance",
        "icon": "❤️",
        "positive": "romantic, warm tones, soft lighting, intimate, beautiful scenery, emotional, tender moments",
        "negative": "cold, harsh, violent, dark, industrial, scary",
        "mood": "warm and emotional",
    },
    "documentary": {
        "name": "Documentary",
        "icon": "🎥",
        "positive": "documentary, authentic, real-world, observational, natural, informative, factual",
        "negative": "fictional, fantasy, stylized, unrealistic, CGI",
        "mood": "informative and authentic",
    },
    "action": {
        "name": "Action",
        "icon": "💥",
        "positive": "action, dynamic, fast-paced, explosive, intense, adrenaline, dramatic angles, motion blur",
        "negative": "calm, slow, peaceful, static, boring, still",
        "mood": "intense and thrilling",
    },
    "comedy": {
        "name": "Comedy",
        "icon": "😂",
        "positive": "comedic, bright colors, playful, whimsical, exaggerated expressions, lighthearted, fun",
        "negative": "dark, serious, grim, depressing, horror",
        "mood": "lighthearted and fun",
    },
    "mystery": {
        "name": "Mystery",
        "icon": "🔍",
        "positive": "mysterious, shadowy, clues, suspenseful, detective, noir lighting, hidden details, intrigue",
        "negative": "obvious, bright, simple, straightforward, cheerful",
        "mood": "suspenseful and intriguing",
    },
    "western": {
        "name": "Western",
        "icon": "🤠",
        "positive": "western, desert landscape, dusty, warm tones, frontier town, sunset, rugged, Sergio Leone style",
        "negative": "modern, urban, futuristic, tropical, cold, snowy",
        "mood": "rugged and nostalgic",
    },
    "fantasy": {
        "name": "Fantasy",
        "icon": "🧙",
        "positive": "high fantasy, magical, enchanted, mythical creatures, epic landscape, ethereal, ancient, mystical",
        "negative": "modern, realistic, urban, technological, mundane",
        "mood": "magical and epic",
    },
}


# =============================================
# Camera Movements (Director Mode)
# =============================================

CAMERA_MOVEMENTS = {
    "static": {"name": "Static", "prompt": "", "description": "No camera movement"},
    "slow_pan_right": {"name": "Slow Pan Right", "prompt": "slow camera pan from left to right, smooth horizontal movement", "description": "Camera pans right"},
    "slow_pan_left": {"name": "Slow Pan Left", "prompt": "slow camera pan from right to left, smooth horizontal movement", "description": "Camera pans left"},
    "zoom_in": {"name": "Zoom In", "prompt": "camera slowly zooms in, gradually closer, tightening frame", "description": "Gradual zoom in"},
    "zoom_out": {"name": "Zoom Out", "prompt": "camera slowly zooms out, revealing more of the scene, widening frame", "description": "Gradual zoom out"},
    "tilt_up": {"name": "Tilt Up", "prompt": "camera tilts upward, revealing sky and height, vertical pan up", "description": "Camera tilts up"},
    "tilt_down": {"name": "Tilt Down", "prompt": "camera tilts downward, revealing ground, vertical pan down", "description": "Camera tilts down"},
    "orbit_cw": {"name": "Orbit Clockwise", "prompt": "camera orbits clockwise around the subject, rotating view, 360 degree movement", "description": "Orbit around subject"},
    "orbit_ccw": {"name": "Orbit Counter-CW", "prompt": "camera orbits counter-clockwise around the subject, rotating perspective", "description": "Reverse orbit"},
    "dolly_forward": {"name": "Dolly Forward", "prompt": "camera moves forward into the scene, tracking forward, approaching", "description": "Camera moves forward"},
    "dolly_backward": {"name": "Dolly Backward", "prompt": "camera pulls back from the scene, retreating, revealing", "description": "Camera pulls back"},
    "crane_up": {"name": "Crane Up", "prompt": "camera cranes upward, aerial reveal, rising overhead shot", "description": "Rising crane shot"},
    "crane_down": {"name": "Crane Down", "prompt": "camera cranes downward from above, descending, lowering to ground level", "description": "Descending crane"},
    "handheld": {"name": "Handheld", "prompt": "slight handheld camera movement, natural shake, documentary style, raw feel", "description": "Natural handheld"},
    "steadicam": {"name": "Steadicam Follow", "prompt": "smooth steadicam following the subject, tracking shot, fluid movement", "description": "Smooth tracking"},
    "dutch_tilt": {"name": "Dutch Tilt", "prompt": "tilted dutch angle camera, dramatic diagonal perspective, unsettling angle", "description": "Tilted angle"},
    "whip_pan": {"name": "Whip Pan", "prompt": "fast whip pan, motion blur, rapid camera movement, energetic transition", "description": "Fast snap pan"},
    "push_in": {"name": "Push In (Dramatic)", "prompt": "dramatic push in toward subject, intense, building tension, Hitchcock zoom", "description": "Dramatic push in"},
}


# =============================================
# Functions
# =============================================

def get_style_presets() -> dict:
    return dict(STYLE_PRESETS)

def get_shot_templates() -> dict:
    return dict(SHOT_TEMPLATES)

def get_lighting_templates() -> dict:
    return dict(LIGHTING_TEMPLATES)

def get_genre_templates() -> dict:
    return dict(GENRE_TEMPLATES)

def get_camera_movements() -> dict:
    return dict(CAMERA_MOVEMENTS)


def apply_preset(prompt: str, preset_name: str) -> tuple:
    """
    Apply a style preset to a prompt.
    Returns (enhanced_prompt, negative_prompt, guidance_override).
    """
    preset = STYLE_PRESETS.get(preset_name)
    if not preset:
        return prompt, "", None

    enhanced = f"{prompt}, {preset['positive']}"
    return enhanced, preset.get("negative", ""), preset.get("guidance_scale")


def apply_shot(prompt: str, shot_name: str) -> str:
    """Apply a shot template to a prompt."""
    shot = SHOT_TEMPLATES.get(shot_name)
    if not shot:
        return prompt
    return f"{shot['prompt']}, {prompt}"


def apply_lighting(prompt: str, lighting_name: str) -> str:
    """Apply a lighting template to a prompt."""
    lighting = LIGHTING_TEMPLATES.get(lighting_name)
    if not lighting:
        return prompt
    return f"{prompt}, {lighting['prompt']}"


def apply_camera_movement(prompt: str, movement_name: str) -> str:
    """Apply a camera movement to a prompt."""
    movement = CAMERA_MOVEMENTS.get(movement_name)
    if not movement or not movement.get("prompt"):
        return prompt
    return f"{prompt}, {movement['prompt']}"


def apply_genre(prompt: str, genre_name: str) -> tuple:
    """Apply genre template. Returns (enhanced, negative)."""
    genre = GENRE_TEMPLATES.get(genre_name)
    if not genre:
        return prompt, ""
    return f"{prompt}, {genre['positive']}", genre.get("negative", "")


def build_full_prompt(
    base_prompt: str,
    style: str = None,
    shot: str = None,
    lighting: str = None,
    camera: str = None,
    genre: str = None,
) -> tuple:
    """
    Build a complete prompt with all enhancements.
    Returns (final_prompt, negative_prompt, guidance_override).
    """
    prompt = base_prompt
    negative = ""
    guidance = None

    if genre:
        prompt, neg = apply_genre(prompt, genre)
        negative = neg

    if style:
        prompt, neg, g = apply_preset(prompt, style)
        if neg:
            negative = f"{negative}, {neg}" if negative else neg
        if g:
            guidance = g

    if shot:
        prompt = apply_shot(prompt, shot)

    if lighting:
        prompt = apply_lighting(prompt, lighting)

    if camera:
        prompt = apply_camera_movement(prompt, camera)

    return prompt, negative, guidance


# =============================================
# Custom Presets (user-saved)
# =============================================

def save_custom_preset(name: str, config: dict):
    """Save a custom preset to the presets directory."""
    os.makedirs(PRESETS_DIR, exist_ok=True)
    safe_name = name.lower().replace(" ", "_").replace("/", "_")
    path = os.path.join(PRESETS_DIR, f"{safe_name}.json")
    config["name"] = name
    config["custom"] = True
    with open(path, "w") as f:
        json.dump(config, f, indent=2)


def load_custom_presets() -> dict:
    """Load all custom presets from the presets directory."""
    if not os.path.exists(PRESETS_DIR):
        return {}
    presets = {}
    for f in os.listdir(PRESETS_DIR):
        if f.endswith(".json"):
            try:
                with open(os.path.join(PRESETS_DIR, f)) as fp:
                    data = json.load(fp)
                    key = f.replace(".json", "")
                    presets[key] = data
            except Exception:
                continue
    return presets


def delete_custom_preset(name: str) -> bool:
    """Delete a custom preset."""
    safe_name = name.lower().replace(" ", "_")
    path = os.path.join(PRESETS_DIR, f"{safe_name}.json")
    if os.path.exists(path):
        os.remove(path)
        return True
    return False


def list_all_presets() -> dict:
    """Get all presets (built-in + custom)."""
    all_presets = dict(STYLE_PRESETS)
    all_presets.update(load_custom_presets())
    return all_presets
