"""Central registry of every model weight StudioLite can use.

Each entry is a plain `ModelSpec` dataclass with:

- `id` - short machine key (`"clip_vit_b32"`, `"whisper_small"`, …).
- `name` + `description` - what to show the user.
- `kind` - the broad category used to group rows in the UI.
- `expected_path` - absolute path (or a zero-arg callable returning one)
  where the loader will look. Anchored at the same HF cache or
  `~/models/*` directory the backends already use, so what the UI
  reports lines up with what the renderers see.
- `min_size_bytes` - the smallest a healthy install can plausibly be.
  Used to flag half-finished downloads: a file smaller than this counts
  as missing rather than present.
- `source_url` - the official page the user clicks "Where to get it".
- `backend_tag` - which StudioLite backend this weight belongs to
  (`"library"`, `"filmmaker"`, …) - useful for grouping in future.

`probe_model` is a pure filesystem check - it never hits the network,
never imports torch, and never kicks off a download. The Settings panel
calls it on every row and renders green-check / grey-missing.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional, Union

ModelKind = Literal["text", "image", "video", "audio", "face", "other"]


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def hf_cache_dir() -> str:
    """Return the active HuggingFace hub cache directory.

    Mirrors how the loader (`transformers`, `diffusers`, `huggingface_hub`)
    resolves it: respect `HF_HOME` when set, otherwise fall back to the
    OS-default `~/.cache/huggingface/hub`.
    """
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return os.path.join(hf_home, "hub")
    return os.path.expanduser("~/.cache/huggingface/hub")


def hf_hub_model_path(hf_id: str) -> str:
    """Return the on-disk directory HF hub creates for a given repo id.

    HF rewrites "org/name" to "models--org--name" inside the hub cache.
    This is where `snapshot_download` and `from_pretrained(... local_files_only=True)`
    read from.
    """
    return os.path.join(hf_cache_dir(), f"models--{hf_id.replace('/', '--')}")


def _library_face_models_dir() -> str:
    """Where `library.faces.ensure_models` drops YuNet / SFace ONNX files.

    The library module stores them under `<library_dir>/face_models/`, and
    the library dir itself defaults to `.mp/library` next to the running
    process (see `library.store`). We resolve the same default here so the
    UI's "expected path" matches where the loader actually looks.
    """
    base = os.environ.get("STUDIOLITE_LIBRARY_DIR") or os.path.join(
        os.path.abspath(os.path.curdir), ".mp", "library"
    )
    return os.path.join(base, "face_models")


def _piper_models_dir() -> str:
    """Where `mpv2.classes.PiperTts` writes voice ONNX files.

    `get_piper_models_dir()` resolves this to `<ROOT_DIR>/piper_models/`
    where ROOT_DIR is the StudioLite repo root. We compute the same path
    without importing the heavy TTS module (which pulls numpy/soundfile).
    """
    override = os.environ.get("STUDIOLITE_PIPER_DIR")
    if override:
        return os.path.expanduser(override)
    # repo root = parent of this file's parent (models/registry.py → repo/)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(repo_root, "piper_models")


# ---------------------------------------------------------------------------
# Spec
# ---------------------------------------------------------------------------

@dataclass
class ModelSpec:
    """Everything the UI needs to describe one model."""

    id: str
    name: str
    kind: ModelKind
    description: str
    # Either an absolute path string, or a zero-arg callable that returns
    # one. Callables let us evaluate `HF_HOME` / `~/models/*` at probe
    # time, picking up env changes the user makes in Settings.
    expected_path: Union[str, Callable[[], str]]
    min_size_bytes: int
    source_url: str
    backend_tag: str
    # Optional notes shown as a hover title in the UI.
    notes: str = ""
    # Optional list of alternative paths - helpful when the loader
    # accepts the file in several places (e.g. HF cache OR ~/models/*).
    alt_paths: List[Callable[[], str]] = field(default_factory=list)


def expected_path_for(spec: ModelSpec) -> str:
    """Resolve `spec.expected_path` to an absolute string.

    Deferred callables are invoked here so changes to `HF_HOME` or the
    custom model dirs take effect without restarting the process.
    """
    p = spec.expected_path
    if callable(p):
        p = p()
    return os.path.abspath(os.path.expanduser(str(p)))


def _candidate_paths(spec: ModelSpec) -> List[str]:
    paths = [expected_path_for(spec)]
    for fn in spec.alt_paths:
        try:
            paths.append(os.path.abspath(os.path.expanduser(fn())))
        except Exception:  # noqa: BLE001 — alt_paths are best-effort
            pass
    return paths


# ---------------------------------------------------------------------------
# Probe
# ---------------------------------------------------------------------------

def _dir_size_bytes(path: str) -> int:
    """Walk `path` and sum file sizes. Returns 0 if the path is missing."""
    if not os.path.isdir(path):
        return 0
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                # Vanished mid-walk or permission-denied; keep going.
                pass
    return total


def _path_size_bytes(path: str) -> int:
    """Return size for a file, or recursive sum for a directory."""
    if os.path.isfile(path):
        try:
            return os.path.getsize(path)
        except OSError:
            return 0
    if os.path.isdir(path):
        return _dir_size_bytes(path)
    return 0


def probe_model(spec: ModelSpec) -> Dict[str, Union[bool, int, str]]:
    """Return the on-disk status of a single `ModelSpec`.

    Pure filesystem check - never downloads, never imports heavy ML
    libraries. The response is dict-shaped so routers can serialize it
    straight to JSON without an intermediate schema class.

    Returns::

        {
            "id": spec.id,
            "present": bool,
            "size_bytes": int,          # 0 if missing
            "path": str,                # resolved expected path
            "matched_path": str | None, # which candidate actually hit (if any)
        }

    A spec is considered "present" when any candidate path exists AND the
    on-disk size is at or above `spec.min_size_bytes` (so a 0-byte stub
    from an aborted download doesn't count as installed).
    """
    primary = expected_path_for(spec)
    best_path: Optional[str] = None
    best_size = 0
    for p in _candidate_paths(spec):
        size = _path_size_bytes(p)
        if size > best_size:
            best_size = size
            best_path = p

    present = (
        best_path is not None
        and os.path.exists(best_path)
        and best_size >= max(1, spec.min_size_bytes)
    )
    return {
        "id": spec.id,
        "present": present,
        "size_bytes": best_size,
        "path": primary,
        "matched_path": best_path if present else None,
    }


def probe_registry(
    registry: Optional[Dict[str, ModelSpec]] = None,
) -> List[Dict[str, object]]:
    """Probe every entry in the registry (or the default REGISTRY).

    Returns a list of dicts the API router can send straight to the UI,
    merging each spec's metadata with its probe result.
    """
    r = registry if registry is not None else REGISTRY
    out: List[Dict[str, object]] = []
    for spec in r.values():
        probe = probe_model(spec)
        out.append(
            {
                "id": spec.id,
                "name": spec.name,
                "kind": spec.kind,
                "description": spec.description,
                "backend_tag": spec.backend_tag,
                "min_size_bytes": spec.min_size_bytes,
                "source_url": spec.source_url,
                "notes": spec.notes,
                "expected_path": probe["path"],
                "present": probe["present"],
                "size_bytes": probe["size_bytes"],
                "matched_path": probe["matched_path"],
            }
        )
    return out


def registry_summary(
    registry: Optional[Dict[str, ModelSpec]] = None,
) -> Dict[str, int]:
    """Totals for the UI banner: present / total / bytes on disk."""
    rows = probe_registry(registry)
    present = sum(1 for r in rows if r["present"])
    bytes_total = sum(int(r["size_bytes"]) for r in rows)
    return {
        "total": len(rows),
        "present": present,
        "missing": len(rows) - present,
        "bytes_on_disk": bytes_total,
    }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def _hf(hf_id: str) -> Callable[[], str]:
    """Shortcut for an HF-hub-cached model's expected directory."""
    return lambda: hf_hub_model_path(hf_id)


def _expanded(path_template: str) -> Callable[[], str]:
    """Lambda-wrap `os.path.expanduser(env_or_default)` so the resolution
    is deferred to probe time.
    """
    return lambda: os.path.expanduser(path_template)


# Minimum sensible sizes per model (bytes). Picked just below the real
# weight size so a healthy download reads as present and a 0-byte or
# partial file reads as missing. These are deliberately generous.
_MB = 1024 * 1024
_GB = 1024 * _MB


REGISTRY: Dict[str, ModelSpec] = {
    # --- Text / vision-language -------------------------------------------
    "clip_vit_b32": ModelSpec(
        id="clip_vit_b32",
        name="CLIP ViT-B/32",
        kind="text",
        description=(
            "OpenAI CLIP base — image + text embeddings. Powers library "
            "semantic search, auto-tagging, and video clustering."
        ),
        expected_path=_hf("openai/clip-vit-base-patch32"),
        min_size_bytes=300 * _MB,
        source_url="https://huggingface.co/openai/clip-vit-base-patch32",
        backend_tag="library",
        notes="~605 MB. Loaded with local_files_only when cached.",
    ),

    # --- Audio: speech-to-text (faster-whisper via Systran repos) ---------
    "whisper_tiny": ModelSpec(
        id="whisper_tiny",
        name="Whisper tiny (faster-whisper)",
        kind="audio",
        description=(
            "Smallest Whisper. Live Transcribe default and library "
            "batch transcript baseline."
        ),
        expected_path=_hf("Systran/faster-whisper-tiny"),
        min_size_bytes=60 * _MB,
        source_url="https://huggingface.co/Systran/faster-whisper-tiny",
        backend_tag="library",
        notes="~75 MB CTranslate2 build.",
    ),
    "whisper_base": ModelSpec(
        id="whisper_base",
        name="Whisper base (faster-whisper)",
        kind="audio",
        description="Base Whisper for mixed-use library transcripts.",
        expected_path=_hf("Systran/faster-whisper-base"),
        min_size_bytes=120 * _MB,
        source_url="https://huggingface.co/Systran/faster-whisper-base",
        backend_tag="library",
        notes="~145 MB.",
    ),
    "whisper_small": ModelSpec(
        id="whisper_small",
        name="Whisper small (faster-whisper)",
        kind="audio",
        description=(
            "Small Whisper — the production-quality default for library "
            "indexing and video-transcribe tooling."
        ),
        expected_path=_hf("Systran/faster-whisper-small"),
        min_size_bytes=400 * _MB,
        source_url="https://huggingface.co/Systran/faster-whisper-small",
        backend_tag="library",
        notes="~465 MB.",
    ),
    "whisper_medium": ModelSpec(
        id="whisper_medium",
        name="Whisper medium (faster-whisper)",
        kind="audio",
        description="Medium Whisper — accuracy upgrade for long content.",
        expected_path=_hf("Systran/faster-whisper-medium"),
        min_size_bytes=1200 * _MB,
        source_url="https://huggingface.co/Systran/faster-whisper-medium",
        backend_tag="library",
        notes="~1.5 GB.",
    ),

    # --- Face: library people index --------------------------------------
    "yunet": ModelSpec(
        id="yunet",
        name="YuNet face detector (OpenCV Zoo)",
        kind="face",
        description=(
            "Fast CPU-friendly face detector used by the library people "
            "index. Pairs with SFace for recognition."
        ),
        expected_path=lambda: os.path.join(
            _library_face_models_dir(), "face_detection_yunet_2023mar.onnx"
        ),
        min_size_bytes=200 * 1024,  # ~230 KB
        source_url=(
            "https://github.com/opencv/opencv_zoo/tree/main/"
            "models/face_detection_yunet"
        ),
        backend_tag="library",
        notes="~230 KB ONNX. Auto-downloads on first use.",
    ),
    "sface": ModelSpec(
        id="sface",
        name="SFace face recognizer (OpenCV Zoo)",
        kind="face",
        description=(
            "128-d L2-normalized face embeddings for the library's person "
            "clustering + search."
        ),
        expected_path=lambda: os.path.join(
            _library_face_models_dir(), "face_recognition_sface_2021dec.onnx"
        ),
        min_size_bytes=30 * _MB,
        source_url=(
            "https://github.com/opencv/opencv_zoo/tree/main/"
            "models/face_recognition_sface"
        ),
        backend_tag="library",
        notes="~37 MB ONNX. Auto-downloads on first use.",
    ),

    # --- Image: SDXL variants used by shots + portraits ------------------
    "sdxl_turbo": ModelSpec(
        id="sdxl_turbo",
        name="SDXL Turbo",
        kind="image",
        description=(
            "Fast distilled SDXL — default for Film Studio shot keyframes "
            "and character portraits."
        ),
        expected_path=_hf("stabilityai/sdxl-turbo"),
        min_size_bytes=5 * _GB,
        source_url="https://huggingface.co/stabilityai/sdxl-turbo",
        backend_tag="filmmaker",
        notes="~6.5 GB. Shared with Images Studio.",
    ),
    "sdxl_base_1_0": ModelSpec(
        id="sdxl_base_1_0",
        name="SDXL base 1.0",
        kind="image",
        description=(
            "Full-strength SDXL base. Used when the shot generator asks "
            "for higher-fidelity faces than Turbo delivers."
        ),
        expected_path=_hf("stabilityai/stable-diffusion-xl-base-1.0"),
        min_size_bytes=5 * _GB,
        source_url="https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0",
        backend_tag="filmmaker",
        notes="~6.9 GB. Loaded on demand when SDXL variant is 'base'.",
    ),

    # --- Video: motion generation ----------------------------------------
    "wan22_ti2v_5b": ModelSpec(
        id="wan22_ti2v_5b",
        name="Wan 2.2 TI2V-5B",
        kind="video",
        description=(
            "Image-to-video renderer used by the Film Studio motion_shots "
            "stage. Opt-in via WAN22_MODEL_DIR."
        ),
        expected_path=_expanded(
            os.environ.get("WAN22_MODEL_DIR", "~/models/wan22-ti2v-5b")
        ),
        min_size_bytes=5 * _GB,
        source_url="https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        backend_tag="filmmaker",
        notes="~10 GB. Resolved from $WAN22_MODEL_DIR or ~/models/wan22-ti2v-5b.",
    ),
    "wan21_t2v_1_3b": ModelSpec(
        id="wan21_t2v_1_3b",
        name="Wan 2.1 T2V-1.3B",
        kind="video",
        description="Compact Wan 2.1 — the 12 GB-card motion fallback.",
        expected_path=_hf("Wan-AI/Wan2.1-T2V-1.3B-Diffusers"),
        min_size_bytes=5 * _GB,
        source_url="https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        backend_tag="filmmaker",
        notes="~7 GB at fp16.",
    ),

    # --- Audio: music + ambient ------------------------------------------
    "musicgen_small": ModelSpec(
        id="musicgen_small",
        name="MusicGen Small",
        kind="audio",
        description="Composer stage score renderer. ~1.5 GB.",
        expected_path=_hf("facebook/musicgen-small"),
        min_size_bytes=1200 * _MB,
        source_url="https://huggingface.co/facebook/musicgen-small",
        backend_tag="filmmaker",
    ),
    "musicgen_medium": ModelSpec(
        id="musicgen_medium",
        name="MusicGen Medium",
        kind="audio",
        description="Richer MusicGen variant preferred when cached.",
        expected_path=_hf("facebook/musicgen-medium"),
        min_size_bytes=3 * _GB,
        source_url="https://huggingface.co/facebook/musicgen-medium",
        backend_tag="filmmaker",
        notes="~3.5 GB. Optional - small is the fallback.",
    ),
    "audioldm2": ModelSpec(
        id="audioldm2",
        name="AudioLDM 2",
        kind="audio",
        description="Ambient-bed renderer used by the Film Studio ambient stage.",
        expected_path=_hf("cvssp/audioldm2"),
        min_size_bytes=1 * _GB,
        source_url="https://huggingface.co/cvssp/audioldm2",
        backend_tag="filmmaker",
        notes="~2.5 GB.",
    ),

    # --- Audio: TTS -------------------------------------------------------
    "piper_amy": ModelSpec(
        id="piper_amy",
        name="Piper voice: Amy (en_US)",
        kind="audio",
        description=(
            "Piper neural TTS voice used by the Film Studio voice_actor "
            "stage when the voice_backend is 'piper'."
        ),
        expected_path=lambda: os.path.join(
            _piper_models_dir(), "en_US-amy-medium.onnx"
        ),
        min_size_bytes=50 * _MB,
        source_url="https://huggingface.co/rhasspy/piper-voices",
        backend_tag="filmmaker",
        notes="Pair of .onnx + .onnx.json files under piper_models/.",
    ),
    "piper_ryan": ModelSpec(
        id="piper_ryan",
        name="Piper voice: Ryan (en_US)",
        kind="audio",
        description="High-quality Piper male voice.",
        expected_path=lambda: os.path.join(
            _piper_models_dir(), "en_US-ryan-high.onnx"
        ),
        min_size_bytes=60 * _MB,
        source_url="https://huggingface.co/rhasspy/piper-voices",
        backend_tag="filmmaker",
    ),
    "indextts2": ModelSpec(
        id="indextts2",
        name="IndexTTS-2",
        kind="audio",
        description=(
            "Expressive TTS used when voice_backend='indextts2'. "
            "Opt-in via INDEXTTS2_MODEL_DIR."
        ),
        expected_path=_expanded(
            os.environ.get("INDEXTTS2_MODEL_DIR", "~/models/indextts2")
        ),
        min_size_bytes=1 * _GB,
        source_url="https://huggingface.co/IndexTeam/IndexTTS-2",
        backend_tag="filmmaker",
        notes="Local-dir install: `hf download IndexTeam/IndexTTS-2 --local-dir ~/models/indextts2`.",
    ),

    # --- Image / Video: upscalers ----------------------------------------
    "realesrgan_x2": ModelSpec(
        id="realesrgan_x2",
        name="Real-ESRGAN x2",
        kind="image",
        description="2x upscaler used by the Film Studio upscale stage and library enhance.",
        expected_path=_expanded(
            os.path.join(
                os.environ.get("REALESRGAN_MODEL_DIR", "~/models/realesrgan"),
                "RealESRGAN_x2.pth",
            )
        ),
        min_size_bytes=30 * _MB,
        source_url="https://huggingface.co/ai-forever/Real-ESRGAN",
        backend_tag="filmmaker",
        notes="~65 MB .pth. Downloads lazily on first use.",
    ),
    "realesrgan_x4": ModelSpec(
        id="realesrgan_x4",
        name="Real-ESRGAN x4",
        kind="image",
        description="4x upscaler — slower but higher fidelity than x2.",
        expected_path=_expanded(
            os.path.join(
                os.environ.get("REALESRGAN_MODEL_DIR", "~/models/realesrgan"),
                "RealESRGAN_x4.pth",
            )
        ),
        min_size_bytes=30 * _MB,
        source_url="https://huggingface.co/ai-forever/Real-ESRGAN",
        backend_tag="filmmaker",
        notes="~65 MB .pth.",
    ),
    "gfpgan_v14": ModelSpec(
        id="gfpgan_v14",
        name="GFPGAN v1.4 (face restore)",
        kind="face",
        description=(
            "Optional face restorer used by library enhance to clean up "
            "upscaled faces. Skipped silently when missing."
        ),
        expected_path=_expanded("~/models/gfpgan/GFPGANv1.4.pth"),
        min_size_bytes=100 * _MB,
        source_url=(
            "https://github.com/TencentARC/GFPGAN/releases/tag/v1.3.0"
        ),
        backend_tag="library",
        notes="~333 MB .pth. Drop under ~/models/gfpgan/.",
    ),
}
