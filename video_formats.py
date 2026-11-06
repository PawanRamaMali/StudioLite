"""Video format presets for social-media-shaped output.

Pure-function module: platform specs in one place, dimension resolver that
respects per-engine divisibility constraints, and safe-zone margins so burned
captions avoid platform UI chrome.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class FormatPreset:
    id: str
    label: str
    aspect: str           # "9:16", "16:9", "1:1", "4:5"
    width: int            # export dimensions (what viewers see)
    height: int
    gen_width: int        # dims to feed the diffusion model (may differ)
    gen_height: int
    max_duration_s: float
    target_fps: int
    max_bitrate_mbps: float
    audio_lufs: float     # target loudness for this platform
    safe_top_pct: float   # fraction of height reserved for platform UI (top gutter)
    safe_bottom_pct: float
    notes: str = ""


# Platform specs cross-referenced with each platform's publishing guides.
# gen_width/gen_height are what we feed SDXL and I2V: smaller than export dims
# so VRAM stays manageable; a post-pass upscale brings them to the export size.
FORMAT_PRESETS: dict[str, FormatPreset] = {
    "reels": FormatPreset(
        id="reels",
        label="Instagram Reels (9:16)",
        aspect="9:16",
        width=1080, height=1920,
        gen_width=544, gen_height=960,
        max_duration_s=90.0,
        target_fps=30,
        max_bitrate_mbps=25.0,
        audio_lufs=-14.0,
        safe_top_pct=0.07,
        safe_bottom_pct=0.18,
        notes="Captions avoid top-right close button and bottom action rail.",
    ),
    "shorts": FormatPreset(
        id="shorts",
        label="YouTube Shorts (9:16)",
        aspect="9:16",
        width=1080, height=1920,
        gen_width=544, gen_height=960,
        max_duration_s=60.0,
        target_fps=30,
        max_bitrate_mbps=100.0,  # youtube reencodes anyway; headroom is fine
        audio_lufs=-14.0,
        safe_top_pct=0.08,
        safe_bottom_pct=0.22,
        notes="Keep lower third clear - large title overlay and progress bar.",
    ),
    "tiktok": FormatPreset(
        id="tiktok",
        label="TikTok (9:16)",
        aspect="9:16",
        width=1080, height=1920,
        gen_width=544, gen_height=960,
        max_duration_s=180.0,
        target_fps=30,
        max_bitrate_mbps=10.0,
        audio_lufs=-14.0,
        safe_top_pct=0.08,
        safe_bottom_pct=0.24,
        notes="Right rail covers ~10% width; keep subjects left-of-center.",
    ),
    "square": FormatPreset(
        id="square",
        label="Instagram Feed (1:1)",
        aspect="1:1",
        width=1080, height=1080,
        gen_width=768, gen_height=768,
        max_duration_s=60.0,
        target_fps=30,
        max_bitrate_mbps=25.0,
        audio_lufs=-14.0,
        safe_top_pct=0.04,
        safe_bottom_pct=0.04,
        notes="Light safe zones - UI chrome lives outside the frame.",
    ),
    "feed_portrait": FormatPreset(
        id="feed_portrait",
        label="Instagram Feed Portrait (4:5)",
        aspect="4:5",
        width=1080, height=1350,
        gen_width=640, gen_height=800,
        max_duration_s=60.0,
        target_fps=30,
        max_bitrate_mbps=25.0,
        audio_lufs=-14.0,
        safe_top_pct=0.04,
        safe_bottom_pct=0.08,
        notes="Tallest feed-safe aspect; caption bar overlays bottom ~8%.",
    ),
    "landscape": FormatPreset(
        id="landscape",
        label="YouTube Landscape (16:9)",
        aspect="16:9",
        width=1920, height=1080,
        gen_width=960, gen_height=544,
        max_duration_s=600.0,
        target_fps=30,
        max_bitrate_mbps=50.0,
        audio_lufs=-14.0,
        safe_top_pct=0.0,
        safe_bottom_pct=0.0,
        notes="Full-frame, no platform safe zones.",
    ),
}


# Engines have per-dimension divisibility constraints (temporal/spatial VAE
# downsample factor). Wan/Hunyuan/VACE need multiples of 16; LTX uses 32.
ENGINE_DIM_STRIDE: dict[str, int] = {
    "wan22_14b_gguf": 16,
    "hunyuan15_i2v_gguf": 16,
    "ltx23_distilled_gguf": 32,
    "vace_1.3b": 16,
    "wan": 16,
}


def resolve_format(format_id: Optional[str]) -> FormatPreset:
    """Return the preset for ``format_id`` or fall back to Reels.

    Unknown ids (including None, "", or garbage) resolve to Reels - the most
    common social aspect - rather than raising, so a stale frontend value
    still produces a reasonable output.
    """
    if not format_id:
        return FORMAT_PRESETS["reels"]
    return FORMAT_PRESETS.get(str(format_id).lower(), FORMAT_PRESETS["reels"])


def dims_for_engine(engine: str, preset: FormatPreset) -> tuple[int, int]:
    """Round the preset's gen dims down to the engine's stride.

    Returns (width, height) ready to pass to the pipeline. Rounding DOWN keeps
    VRAM predictable; rounding up could OOM on 12GB cards.
    """
    stride = ENGINE_DIM_STRIDE.get(engine, 16)
    w = max(stride, (preset.gen_width // stride) * stride)
    h = max(stride, (preset.gen_height // stride) * stride)
    return w, h


def safe_zone_px(preset: FormatPreset) -> tuple[int, int, int, int]:
    """Return (top, right, bottom, left) pixel margins to stay inside platform UI."""
    top = int(round(preset.height * preset.safe_top_pct))
    bottom = int(round(preset.height * preset.safe_bottom_pct))
    # Platform right rail typically covers ~10% on vertical formats
    right = int(round(preset.width * 0.10)) if preset.aspect == "9:16" else 0
    left = 0
    return top, right, bottom, left


def platform_targets(format_id: str) -> dict:
    """UI-facing summary for the format panel. Serializable."""
    p = resolve_format(format_id)
    top, right, bottom, left = safe_zone_px(p)
    return {
        "id": p.id,
        "label": p.label,
        "aspect": p.aspect,
        "width": p.width,
        "height": p.height,
        "gen_width": p.gen_width,
        "gen_height": p.gen_height,
        "max_duration_s": p.max_duration_s,
        "target_fps": p.target_fps,
        "max_bitrate_mbps": p.max_bitrate_mbps,
        "audio_lufs": p.audio_lufs,
        "safe_zone_px": {"top": top, "right": right, "bottom": bottom, "left": left},
        "notes": p.notes,
    }


def all_presets() -> list[dict]:
    """List every preset as plain dicts (used by GET /api/v1/formats)."""
    return [platform_targets(pid) for pid in FORMAT_PRESETS.keys()]


def clamp_duration(format_id: str, duration_s: float) -> float:
    """Cap ``duration_s`` to what the platform accepts. Never below 1 second."""
    p = resolve_format(format_id)
    return max(1.0, min(float(duration_s), p.max_duration_s))
