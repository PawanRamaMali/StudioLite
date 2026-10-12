"""Per-stage backend registry: capability contracts for every pluggable
model the pipeline can call.

Each stage of the pipeline (motion, voice, music, lipsync, upscale, shots)
can route to one of several backends. Instead of each stage hardcoding a
VRAM number or license string in its own file, we keep the facts in one
place keyed by (stage, backend_name):

  - license:  short SPDX-style tag ("Apache-2.0", "CC-BY-NC", "MIT")
  - min_vram_gb_fp16:  floor in GB at full precision (None = CPU-fine)
  - cloud_burst:  may this backend run via the cloud-burst dispatcher?
  - base_model:  points at the underlying weights family when relevant
  - quant_variants:  list of QuantVariant describing shippable quant paths
  - notes:  one-line human-readable context, surfaced in the UI

The research (deep-research/wj2ricjv6 + wxtc85vzl) gave us concrete numbers
for the committed set. Numbers here are the floor the registry can claim;
callers should treat anything missing as "unknown" rather than infer a
value. Entries that only exist as roadmap aspirations are flagged with
``available="roadmap"`` so the UI can show them without pretending they
can be selected today.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class QuantVariant:
    """A quant level for a backend. ``vram_gb`` is the inference-time
    footprint. ``format`` is one of 'fp16' | 'bf16' | 'fp8' | 'w4a8' |
    'gguf_q4' | 'gguf_q5' | 'gguf_q8' | 'nf4' | 'int8' | 'onnx' | 'coreml'
    - these are the names that round-trip into the UI."""
    format: str
    vram_gb: float
    source_url: str = ""
    notes: str = ""


@dataclass(frozen=True)
class BackendSpec:
    stage: str
    name: str
    license: str
    available: str                          # "shipping" | "roadmap"
    min_vram_gb_fp16: Optional[float] = None
    cloud_burst: bool = False
    base_model: str = ""
    quant_variants: Tuple[QuantVariant, ...] = ()
    source_url: str = ""
    notes: str = ""


# Short, documented spec list. One entry per backend per stage. Numbers are
# the registry's statement of record; the harness tests pin a few of them
# so a drift here forces a test update, which is the intended review path.
_REGISTRY: Dict[Tuple[str, str], BackendSpec] = {}


def register(spec: BackendSpec) -> None:
    key = (spec.stage, spec.name)
    _REGISTRY[key] = spec


def get(stage: str, name: str) -> Optional[BackendSpec]:
    return _REGISTRY.get((stage, name))


def for_stage(stage: str) -> List[BackendSpec]:
    return [s for (st, _), s in _REGISTRY.items() if st == stage]


def all_specs() -> List[BackendSpec]:
    return list(_REGISTRY.values())


def accepted_names(stage: str) -> List[str]:
    """Names a config patch for this stage may legally hold. Used by the
    projects.py patch validator so adding a backend here lights it up for
    the API without a second edit."""
    return sorted(name for (st, name) in _REGISTRY if st == stage)


# ---------------------------------------------------------------------------
# Shipping entries - these backends have code in the repo today.

register(BackendSpec(
    stage="upscale", name="none",
    license="N/A", available="shipping",
    min_vram_gb_fp16=0.0, cloud_burst=False,
    notes="Explicit opt-out; leaves the final cut untouched.",
))
register(BackendSpec(
    stage="upscale", name="realesrgan_x2",
    license="BSD-3-Clause", available="shipping",
    min_vram_gb_fp16=2.0, cloud_burst=True,
    base_model="ai-forever/Real-ESRGAN",
    source_url="https://huggingface.co/ai-forever/Real-ESRGAN",
    notes="2x neural upscale, 720p -> 1440p. Light VRAM, CPU fallback is slow but works.",
))
register(BackendSpec(
    stage="upscale", name="realesrgan_x4",
    license="BSD-3-Clause", available="shipping",
    min_vram_gb_fp16=4.0, cloud_burst=True,
    base_model="ai-forever/Real-ESRGAN",
    source_url="https://huggingface.co/ai-forever/Real-ESRGAN",
    notes="4x neural upscale; usually overkill on AI video.",
))

register(BackendSpec(
    stage="lipsync", name="none",
    license="N/A", available="shipping",
    min_vram_gb_fp16=0.0, cloud_burst=False,
    notes="Explicit opt-out.",
))
register(BackendSpec(
    stage="lipsync", name="musetalk",
    license="MIT", available="shipping",
    min_vram_gb_fp16=8.0, cloud_burst=True,
    base_model="TMElyralab/MuseTalk-1.5",
    source_url="https://github.com/TMElyralab/MuseTalk",
    notes="Mouth-only repaint; isolated venv required (torch 2.0.1 + mmcv 2.0.1).",
))
register(BackendSpec(
    stage="lipsync", name="latentsync",
    license="Apache-2.0", available="shipping",
    min_vram_gb_fp16=8.0, cloud_burst=True,
    base_model="bytedance/LatentSync-1.5",
    source_url="https://github.com/bytedance/LatentSync",
    notes="Diffusion-based mouth repaint; isolated venv required.",
))
register(BackendSpec(
    stage="lipsync", name="echomimicv3",
    license="Apache-2.0", available="shipping",
    min_vram_gb_fp16=12.0, cloud_burst=True,
    base_model="BadToBest/EchoMimicV3 (on Wan2.1-Fun-V1.1-1.3B-InP)",
    source_url="https://github.com/antgroup/echomimic_v3",
    notes=("Holistic audio-driven portrait animator; 1.3B params on Wan2.1-Fun base. "
           "AAAI 2026. 12GB VRAM floor; no documented CPU path."),
))

# Motion backend names (strings projects.py already accepts) - repeated
# here so the UI can source VRAM floors and license strings from one place.
register(BackendSpec(
    stage="motion_shots", name="auto",
    license="N/A", available="shipping",
    min_vram_gb_fp16=0.0, cloud_burst=False,
    notes="Falls through SVD -> AnimateDiff -> Wan -> Ken Burns.",
))
register(BackendSpec(
    stage="motion_shots", name="wan22",
    license="Apache-2.0", available="shipping",
    min_vram_gb_fp16=16.0, cloud_burst=True,
    base_model="Wan-AI/Wan2.2-TI2V-5B",
    source_url="https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B",
    quant_variants=(
        QuantVariant(format="fp8", vram_gb=12.0,
                     source_url="https://huggingface.co/city96/Wan2.2-TI2V-5B-fp8",
                     notes="Community fp8; ComfyUI-GGUF unet dir."),
        QuantVariant(format="w4a8", vram_gb=2.75,
                     source_url="https://github.com/ameriles/comfy-quant-bench",
                     notes="comfy-quant-bench verified: 9.31 -> 2.75 GiB on W4A8, 1.59x speedup."),
    ),
    notes="Wan 2.2 TI2V-5B, 704p native, 5s per clip.",
))

register(BackendSpec(
    stage="composer", name="musicgen",
    license="CC-BY-NC-4.0", available="shipping",
    min_vram_gb_fp16=4.0, cloud_burst=True,
    base_model="facebook/musicgen-small",
    source_url="https://huggingface.co/facebook/musicgen-small",
    notes="Default. Non-commercial license - swap for anything commercial.",
))
register(BackendSpec(
    stage="composer", name="acestep",
    license="MIT", available="shipping",
    min_vram_gb_fp16=6.0, cloud_burst=True,
    base_model="ACE-Step/ACE-Step-v1.5",
    source_url="https://github.com/ace-step/ACE-Step",
    notes="2-4B DiT; benchmarks between Suno v4.5 and v5 (author-reported, unverified).",
))

register(BackendSpec(
    stage="voice_actor", name="piper",
    license="MIT", available="shipping",
    min_vram_gb_fp16=0.0, cloud_burst=False,
    notes="Default. CPU-fine, flat delivery.",
))
register(BackendSpec(
    stage="voice_actor", name="xtts",
    license="CPML (non-commercial)", available="shipping",
    min_vram_gb_fp16=4.0, cloud_burst=True,
    base_model="coqui/XTTS-v2",
    source_url="https://huggingface.co/coqui/XTTS-v2",
    notes="24 kHz, 6s-clip cloning. License is CPML; not drop-in commercial.",
))

# ---------------------------------------------------------------------------
# Roadmap entries - not yet wired in a stage runner, but surfaced in the
# registry so the UI, docs, and tests can refer to them from day one.

register(BackendSpec(
    stage="upscale", name="seedvr2",
    license="Apache-2.0", available="roadmap",
    min_vram_gb_fp16=16.0, cloud_burst=True,
    base_model="ByteDance-Seed/SeedVR2-3B",
    source_url="https://huggingface.co/ByteDance-Seed/SeedVR2-3B",
    quant_variants=(
        QuantVariant(format="gguf_q3_k_m", vram_gb=1.55,
                     source_url="https://huggingface.co/cmeka/SeedVR2-GGUF"),
        QuantVariant(format="gguf_q4_k_m", vram_gb=2.0,
                     source_url="https://huggingface.co/cmeka/SeedVR2-GGUF"),
        QuantVariant(format="gguf_q5_k_m", vram_gb=2.41,
                     source_url="https://huggingface.co/cmeka/SeedVR2-GGUF"),
        QuantVariant(format="gguf_q6_k", vram_gb=2.85,
                     source_url="https://huggingface.co/cmeka/SeedVR2-GGUF"),
        QuantVariant(format="gguf_q8_0", vram_gb=3.66,
                     source_url="https://huggingface.co/cmeka/SeedVR2-GGUF"),
    ),
    notes=("Diffusion-transformer video restoration (CVPR 2025 / ICLR 2026). "
           "GGUF path via ComfyUI-GGUF, needs numz/SeedVR2_comfyUI finetune."),
))

register(BackendSpec(
    stage="motion_shots", name="wan21_flf2v",
    license="Apache-2.0", available="roadmap",
    min_vram_gb_fp16=16.0, cloud_burst=True,
    base_model="Wan-AI/Wan2.1-Fun-14B-InP",
    source_url="https://huggingface.co/city96/wan2.1-fun-14b-inp-gguf",
    quant_variants=(
        QuantVariant(format="gguf_q2_k", vram_gb=6.02,
                     source_url="https://huggingface.co/city96/wan2.1-fun-14b-inp-gguf"),
        QuantVariant(format="gguf_q8_0", vram_gb=17.7,
                     source_url="https://huggingface.co/city96/wan2.1-fun-14b-inp-gguf"),
        QuantVariant(format="bf16", vram_gb=32.8,
                     source_url="https://huggingface.co/city96/wan2.1-fun-14b-inp-gguf"),
    ),
    notes=("First-last-frame-to-video for shot transitions (InfinityStory recipe). "
           "Full GGUF ladder via city96's ComfyUI-GGUF."),
))

register(BackendSpec(
    stage="composer", name="diffrhythm2",
    license="Apache-2.0", available="roadmap",
    min_vram_gb_fp16=8.0, cloud_burst=True,
    base_model="ASLP-lab/DiffRhythm2",
    source_url="https://huggingface.co/ASLP-lab/DiffRhythm2",
    notes=("Semi-autoregressive block flow matching with 5 Hz Music VAE. "
           "25.6 s for 2 min music on RTX 4090 (RTF 0.213). "
           "Lyric-to-vocal alignment without external labels."),
))

register(BackendSpec(
    stage="shots", name="omnigen2",
    license="Apache-2.0", available="roadmap",
    min_vram_gb_fp16=20.0, cloud_burst=True,
    base_model="VectorSpaceLab/OmniGen2",
    source_url="https://arxiv.org/html/2505.24862v5",
    notes=("Multi-character identity preservation (ViStoryBench CIDS leader). "
           "Composed with StoryBooth bounded cross-frame attention."),
))
