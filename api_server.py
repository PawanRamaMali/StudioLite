"""
StudioLite REST API Server

Run with: uvicorn api_server:app --host 0.0.0.0 --port 8000
Or: python api_server.py
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict
import logging
import threading
import uuid
import time
import os
import traceback

logger = logging.getLogger("studiolite.api")

app = FastAPI(
    title="StudioLite API",
    version="1.0.0",
    description="AI Video Generation & Editing API",
)

# CORS middleware - allow all origins by default for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root directory (same as the rest of StudioLite)
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(ROOT_DIR, ".mp")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Character portraits directory
CHAR_PORTRAITS_DIR = os.path.join(OUTPUT_DIR, "character_portraits")
os.makedirs(CHAR_PORTRAITS_DIR, exist_ok=True)

# Serve character portraits as static files
app.mount("/static/portraits", StaticFiles(directory=CHAR_PORTRAITS_DIR), name="portraits")

# Image studio output (text-to-image, edits, upscales, etc.)
IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")
os.makedirs(IMAGES_DIR, exist_ok=True)
app.mount("/static/images", StaticFiles(directory=IMAGES_DIR), name="images")

# Live transcripts directory
TRANSCRIPTS_DIR = os.path.join(OUTPUT_DIR, "transcripts")
os.makedirs(TRANSCRIPTS_DIR, exist_ok=True)
app.mount("/static/transcripts", StaticFiles(directory=TRANSCRIPTS_DIR), name="transcripts")

# Live screen-OCR transcripts directory
SCREEN_TRANSCRIPTS_DIR = os.path.join(OUTPUT_DIR, "screen_transcripts")
os.makedirs(SCREEN_TRANSCRIPTS_DIR, exist_ok=True)
app.mount("/static/screen_transcripts", StaticFiles(directory=SCREEN_TRANSCRIPTS_DIR), name="screen_transcripts")

# ---------------------------------------------------------------------------
# Job tracking
# ---------------------------------------------------------------------------
jobs = {}  # job_id -> dict
_jobs_lock = threading.Lock()


def _create_job(kind: str, params: dict | None = None) -> str:
    job_id = str(uuid.uuid4())
    with _jobs_lock:
        jobs[job_id] = {
            "status": "queued",
            "progress": 0.0,
            "message": "",
            "result": None,
            "error": None,
            "created_at": time.time(),
            "kind": kind,
            "params": params or {},
        }
    return job_id


def _update_job(job_id: str, **kwargs):
    with _jobs_lock:
        if job_id in jobs:
            jobs[job_id].update(kwargs)


def _job_response(job_id: str) -> dict:
    with _jobs_lock:
        job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return {
        "job_id": job_id,
        "status": job["status"],
        "progress": job["progress"],
        "message": job.get("message", ""),
        "result": job.get("result"),
        "error": job.get("error"),
        "created_at": job["created_at"],
        "elapsed": round(time.time() - job["created_at"], 2),
    }


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class TextToVideoRequest(BaseModel):
    prompt: str
    negative_prompt: str = "low quality, blurry, distorted"
    engine: str = "wan"  # wan, ltx, cogvideox, hunyuan
    model: str = "1.3b"
    resolution: str = "480p"
    num_frames: int = 49
    num_inference_steps: int = 30
    guidance_scale: float = 5.0
    fps: int = 16
    seed: Optional[int] = None


class ImageToVideoRequest(BaseModel):
    prompt: str
    image_path: str
    engine: str = "wan"
    model: str = "1.3b"
    resolution: str = "480p"
    num_frames: int = 49
    num_inference_steps: int = 30
    guidance_scale: float = 5.0
    fps: int = 16
    seed: Optional[int] = None


class StoryScene(BaseModel):
    title: str = ""
    visual: str = ""
    narration: str = ""
    duration: int = 10
    characters: list[str] = []  # Character IDs/names in this scene (for per-scene IP-Adapter)

class StoryRequest(BaseModel):
    concept: str
    num_scenes: int = Field(default=4, ge=1, le=8)
    genre: str = "Cinematic"
    mood: str = "Epic"
    engine: str = "wan"
    model: str = "1.3b"
    enable_narration: bool = True
    continuity_mode: str = "Prompt Anchoring"
    visual_anchor: str = ""
    transition_type: str = "cut"
    num_frames: int = 49
    fps: int = 16
    resolution: str = "480p"
    scenes: list[StoryScene] = []  # If provided, use these instead of LLM generation
    # IP-Adapter character consistency
    ip_adapter_enabled: bool = True  # Enable IP-Adapter auto-capture from first scene
    ip_adapter_strength: float = 0.6  # 0.0-1.0, how strongly to enforce character appearance
    character_ref_images: list[str] = []  # Paths to character reference images (uploaded)
    # Quality settings
    quality_preset: str = "standard"  # draft, standard, high
    enable_upscale: bool = False  # Post-process upscale with Real-ESRGAN
    enable_interpolation: bool = False  # RIFE frame interpolation for smoother motion
    # Engine override — "auto" picks best available (Wan 2.2 > Hunyuan 1.5 > LTX-2.3 > VACE > T2V).
    # Map quality tiers: "draft" -> LTX-2.3, "balanced" -> Hunyuan 1.5, "quality" -> Wan 2.2.
    video_engine: str = "auto"
    # LatentSync 1.6 post-processing for talking-head scenes.
    enable_lip_sync: bool = False


class TTSRequest(BaseModel):
    text: str
    voice: str = "Amy"
    engine: str = "piper"  # piper or kitten
    persona: str = ""               # optional persona id from list_personas()
    speed: float = Field(default=1.0, ge=0.25, le=4.0)
    pitch: float = Field(default=0.0, ge=-12.0, le=12.0)
    volume: float = Field(default=1.0, ge=0.0, le=3.0)
    output_format: str = "wav"      # 'wav' or 'mp3'


class SFXRequest(BaseModel):
    sfx_type: str = ""  # one of generate_sfx_procedural's preset keys, or empty when using prompt
    prompt: str = ""    # free-text description (matched to a preset by keyword)
    duration: float = Field(default=2.0, ge=0.2, le=30.0)


class RestructureRequest(BaseModel):
    session_id: str
    model: str
    style: str = "cleanup"
    custom_prompt: Optional[str] = None
    title: Optional[str] = None
    temperature: float = Field(default=0.2, ge=0.0, le=1.5)
    host: Optional[str] = None  # override OLLAMA_HOST per request


class TrimRequest(BaseModel):
    video_path: str
    start_time: float = Field(..., ge=0)
    end_time: float = Field(..., gt=0)


class MergeRequest(BaseModel):
    video_paths: list[str]
    transition: str = "cut"  # cut, crossfade
    transition_duration: float = Field(default=1.0, ge=0)


class UpscaleRequest(BaseModel):
    video_path: str
    scale: int = Field(default=2, ge=1, le=4)
    preset: str = "quality_2x"


class ExtractAudioRequest(BaseModel):
    video_path: str
    format: str = "wav"  # wav | mp3


class VideoPathRequest(BaseModel):
    video_path: str


class CompressRequest(BaseModel):
    video_path: str
    preset: str = "medium"  # high | medium | low


class RotateFlipRequest(BaseModel):
    video_path: str
    rotate: int = 0  # 0 | 90 | 180 | 270
    flip: str = "none"  # none | horizontal | vertical


class ReverseRequest(BaseModel):
    video_path: str
    include_audio: bool = False


class LoopRequest(BaseModel):
    video_path: str
    count: int = Field(default=2, ge=2, le=20)


class StabilizeRequest(BaseModel):
    video_path: str
    shakiness: int = Field(default=5, ge=1, le=10)
    accuracy: int = Field(default=9, ge=1, le=15)


class ColorCorrectionRequest(BaseModel):
    video_path: str
    brightness: float = Field(default=0.0, ge=-1.0, le=1.0)
    contrast: float = Field(default=1.0, ge=0.0, le=3.0)
    saturation: float = Field(default=1.0, ge=0.0, le=3.0)
    hue: float = Field(default=0.0, ge=-180.0, le=180.0)


class RegionEffectRequest(BaseModel):
    video_path: str
    x: int = Field(default=0, ge=0)
    y: int = Field(default=0, ge=0)
    width: int = Field(default=200, ge=1)
    height: int = Field(default=120, ge=1)
    mode: str = "blur"  # blur | pixelate
    strength: int = Field(default=12, ge=2, le=80)


class PictureInPictureRequest(BaseModel):
    video_path: str
    overlay_path: str
    position: str = "bottom_right"  # top_left | top_right | bottom_left | bottom_right | center
    size_percent: int = Field(default=25, ge=10, le=80)
    margin: int = Field(default=24, ge=0, le=200)


class BackgroundMusicRequest(BaseModel):
    video_path: str
    audio_path: str
    music_volume: float = Field(default=0.35, ge=0.0, le=2.0)
    video_volume: float = Field(default=1.0, ge=0.0, le=2.0)


class SpeedRequest(BaseModel):
    video_path: str
    speed: float = Field(default=1.0, ge=0.25, le=4.0)
    keep_audio: bool = True


class GifRequest(BaseModel):
    video_path: str
    start_time: float = Field(default=0.0, ge=0.0)
    duration: float = Field(default=3.0, gt=0.0, le=30.0)
    fps: int = Field(default=12, ge=4, le=30)
    width: int = Field(default=480, ge=120, le=1280)


class ThumbnailRequest(BaseModel):
    video_path: str
    timestamp: float = Field(default=0.0, ge=0.0)
    width: int = Field(default=1280, ge=120, le=3840)


# --- Image Studio request models ---

class ImageGenerateRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    size: str = "portrait_9x16"
    width: Optional[int] = None
    height: Optional[int] = None
    style: Optional[str] = "photorealistic"
    seed: Optional[int] = None
    steps: int = Field(default=30, ge=1, le=80)
    guidance: float = Field(default=7.5, ge=0.0, le=20.0)
    provider: str = "sdxl"   # sdxl | fooocus | nanobanana2
    model: Optional[str] = None
    enhance_prompt: bool = False  # If true, run prompt through LLM first


class ImageEditRequest(BaseModel):
    image_path: str
    prompt: str
    negative_prompt: str = ""
    strength: float = Field(default=0.75, ge=0.0, le=1.0)
    style: Optional[str] = None
    seed: Optional[int] = None
    steps: int = Field(default=30, ge=1, le=80)
    guidance: float = Field(default=7.5, ge=0.0, le=20.0)
    provider: str = "sdxl"
    # Edit dispatch: "auto" (heuristic), "instruct" (InstructPix2Pix — best for
    # 'raise her hand' / 'make it night'), or "redraw" (SDXL Img2Img).
    technique: str = "auto"
    # InstructPix2Pix only — text vs image preservation balance (1.0-2.0).
    image_guidance: float = Field(default=1.5, ge=1.0, le=3.0)


class ImageInpaintRequest(BaseModel):
    image_path: str
    mask_path: str
    prompt: str
    negative_prompt: str = ""
    style: Optional[str] = None
    seed: Optional[int] = None
    steps: int = Field(default=30, ge=1, le=80)
    guidance: float = Field(default=7.5, ge=0.0, le=20.0)
    invert_mask: bool = False


class ImageVariationRequest(BaseModel):
    image_path: str
    prompt: Optional[str] = None
    strength: float = Field(default=0.45, ge=0.05, le=0.95)
    style: Optional[str] = None
    seed: Optional[int] = None


class ImageUpscaleRequest(BaseModel):
    image_path: str
    scale: int = Field(default=2, ge=2, le=4)
    method: str = "lanczos"  # lanczos | realesrgan


class ImageBgRemoveRequest(BaseModel):
    image_path: str


class PromptEnhanceRequest(BaseModel):
    prompt: str
    style: Optional[str] = None
    mode: str = "expand"  # expand | shorten
    image_path: Optional[str] = None  # When set, BLIP captions it for context
    negative_prompt: Optional[str] = None  # User's existing negative to refine


class JobResponse(BaseModel):
    job_id: str
    status: str  # queued, running, completed, failed
    progress: float = 0
    message: str = ""
    result: Optional[dict] = None
    error: Optional[str] = None
    created_at: float = 0
    elapsed: float = 0


# ---------------------------------------------------------------------------
# Helper: build VideoGenConfig from request parameters
# ---------------------------------------------------------------------------

def _build_config(
    engine: str,
    model: str,
    resolution: str,
    num_frames: int,
    num_inference_steps: int,
    guidance_scale: float,
    fps: int,
    seed: int | None,
):
    from videogen import VideoGenConfig

    config = VideoGenConfig(
        engine=engine,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        fps=fps,
        seed=seed,
    )

    # Engine-specific settings
    if engine == "wan":
        config.wan_model = model
        config.wan_resolution = resolution
    elif engine == "ltx":
        config.ltx_model = model
    elif engine == "cogvideox":
        config.model_variant = model
    elif engine == "hunyuan":
        config.hunyuan_model = model
        config.hunyuan_resolution = resolution

    return config


# ---------------------------------------------------------------------------
# Background workers
# ---------------------------------------------------------------------------

def _run_text2video(job_id: str, req: TextToVideoRequest):
    try:
        _update_job(job_id, status="running", message="Initializing video generator...")
        from videogen import VideoGenerator

        config = _build_config(
            engine=req.engine,
            model=req.model,
            resolution=req.resolution,
            num_frames=req.num_frames,
            num_inference_steps=req.num_inference_steps,
            guidance_scale=req.guidance_scale,
            fps=req.fps,
            seed=req.seed,
        )
        gen = VideoGenerator(config)

        def progress_cb(step, total, message):
            pct = (step / max(total, 1)) * 100
            _update_job(job_id, progress=round(pct, 1), message=message)

        output_path = gen.generate_text2video(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            progress_callback=progress_cb,
        )

        _update_job(
            job_id,
            status="completed",
            progress=100,
            message="Video generation complete.",
            result={"video_path": output_path, "engine": req.engine},
        )
    except Exception as exc:
        _update_job(
            job_id,
            status="failed",
            error=str(exc),
            message=f"Generation failed: {exc}",
        )


def _run_image2video(job_id: str, req: ImageToVideoRequest):
    try:
        _update_job(job_id, status="running", message="Initializing image-to-video...")
        from videogen import VideoGenerator

        if not os.path.isfile(req.image_path):
            raise FileNotFoundError(f"Image not found: {req.image_path}")

        config = _build_config(
            engine=req.engine,
            model=req.model,
            resolution=req.resolution,
            num_frames=req.num_frames,
            num_inference_steps=req.num_inference_steps,
            guidance_scale=req.guidance_scale,
            fps=req.fps,
            seed=req.seed,
        )
        gen = VideoGenerator(config)

        def progress_cb(step, total, message):
            pct = (step / max(total, 1)) * 100
            _update_job(job_id, progress=round(pct, 1), message=message)

        output_path = gen.generate_image2video(
            image_path=req.image_path,
            prompt=req.prompt,
            progress_callback=progress_cb,
        )

        _update_job(
            job_id,
            status="completed",
            progress=100,
            message="Image-to-video complete.",
            result={"video_path": output_path, "engine": req.engine},
        )
    except Exception as exc:
        _update_job(
            job_id,
            status="failed",
            error=str(exc),
            message=f"Image-to-video failed: {exc}",
        )


def _run_story(job_id: str, req: StoryRequest):
    try:
        _update_job(job_id, status="running", message="Starting story generation...")
        from videogen import VideoGenerator, VideoGenConfig, concatenate_videos
        from mpv2.classes.TtsFactory import get_tts_instance
        from reelforge import rf_clean_text_for_tts
        from mpv2.llm_provider import generate_text, select_backend, select_model
        import json
        import re

        # Step 1 -- Use user-provided scenes or generate via LLM
        # If frontend sent scenes (from the storyboard editor), use those directly
        _use_user_scenes = req.scenes and len(req.scenes) > 0

        if _use_user_scenes:
            _update_job(job_id, progress=10, message=f"Using {len(req.scenes)} user-defined scenes")
        else:
            _update_job(job_id, progress=5, message="Generating story script with AI...")

        prompt = f"""You are a professional cinematographer. Write a {req.num_scenes}-scene shot list for this film:

STORY: {req.concept}
GENRE: {req.genre} | MOOD: {req.mood}

CRITICAL: Return a JSON object with TWO keys.

1. "visual_identity": Describe the MAIN SUBJECT in extreme detail so it looks IDENTICAL in every scene. Include: exact physical description (breed, color, size, distinguishing marks), exact clothing/accessories, exact setting details that repeat. This paragraph is appended to EVERY scene prompt.

2. "scenes": JSON array where each scene has:
   - "title": 3-5 word title
   - "visual": MUST start with the exact same subject description from visual_identity, then add: specific camera angle (close-up/medium/wide/tracking), specific continuous motion (what the subject is doing), specific lighting, specific background. Describe ONE continuous smooth action per scene.
   - "narration": 2-3 sentence engaging voiceover that connects to previous scene
   - "duration": 4-6 seconds

Return ONLY valid JSON (close all brackets):
{{
  "visual_identity": "...",
  "scenes": [
    {{"title": "...", "visual": "...", "narration": "...", "duration": 5}}
  ]
}}"""

        try:
            from reelforge import load_config
            cfg = load_config()
            backend = cfg.get("llm_backend", "ollama")
            select_backend(backend)
            if backend == "ollama":
                select_model(cfg.get("ollama_model") or "llama3.2:3b")
            else:
                gguf = cfg.get("gguf_model", "")
                if gguf:
                    select_model(gguf)
        except Exception:
            pass

        response = generate_text(prompt)
        response = response.replace("```json", "").replace("```", "").strip()

        # Parse response
        parsed = None
        try:
            parsed = json.loads(response)
        except json.JSONDecodeError:
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group())
                except json.JSONDecodeError:
                    pass

        visual_identity = ""
        scene_list = []

        if isinstance(parsed, dict):
            visual_identity = parsed.get("visual_identity", "")
            scene_list = parsed.get("scenes", [])
        elif isinstance(parsed, list):
            scene_list = parsed

        if not scene_list:
            # Try array with closing bracket
            match = re.search(r'\[.*\]', response, re.DOTALL)
            if match:
                try:
                    scene_list = json.loads(match.group())
                except json.JSONDecodeError:
                    pass

        if not scene_list:
            # Fix incomplete JSON (missing closing bracket)
            match = re.search(r'\[', response)
            if match:
                arr_text = response[match.start():]
                if not arr_text.rstrip().endswith(']'):
                    last_brace = arr_text.rfind('}')
                    if last_brace > 0:
                        arr_text = arr_text[:last_brace + 1] + ']'
                try:
                    scene_list = json.loads(arr_text)
                except json.JSONDecodeError:
                    pass

        if not scene_list:
            # Extract individual objects
            obj_matches = re.findall(r'\{[^{}]*\}', response)
            for obj_str in obj_matches:
                try:
                    obj = json.loads(obj_str)
                    if "title" in obj or "visual" in obj:
                        scene_list.append(obj)
                except json.JSONDecodeError:
                    pass

        # Normalize fields (visual might be array)
        for s in scene_list:
            if isinstance(s.get("visual"), list):
                s["visual"] = ", ".join(str(v) for v in s["visual"])
            if isinstance(s.get("narration"), list):
                s["narration"] = " ".join(str(n) for n in s["narration"])

        if not scene_list:
            _update_job(job_id, status="failed",
                error="Script generation failed: LLM returned no usable scenes. Please provide scenes manually in the storyboard editor.",
                message="Script generation failed — no scenes returned from LLM.")
            return

        scene_list = scene_list[: req.num_scenes]

        # Override with user-provided scenes (from the storyboard editor)
        # These have user-edited titles, prompts, narrations, and DURATIONS
        if _use_user_scenes:
            scene_list = [s.model_dump() for s in req.scenes]
            visual_identity = req.visual_anchor or visual_identity or ""

        _update_job(job_id, progress=15, message=f"Script ready: {len(scene_list)} scenes")

        # Step 2 -- Generate narration audio per scene (if enabled)
        narration_paths = []
        narration_durations = []  # actual duration of each narration in seconds
        if req.enable_narration:
            _update_job(job_id, progress=18, message="Generating narration audio...")
            tts = get_tts_instance(req.engine if req.engine in ("piper", "kitten") else "piper")
            for idx, scene in enumerate(scene_list):
                narration_text = rf_clean_text_for_tts(scene.get("narration", ""))
                audio_path = os.path.join(OUTPUT_DIR, f"story_narration_{job_id}_{idx}.wav")
                tts.synthesize(narration_text, audio_path)
                narration_paths.append(audio_path)
                # Measure actual narration duration so video generation can
                # produce a clip long enough to cover the audio (no abrupt cutoff).
                try:
                    import soundfile as _sf
                    info = _sf.info(audio_path)
                    narration_durations.append(float(info.duration))
                except Exception:
                    narration_durations.append(0.0)
                pct = 18 + int((idx + 1) / len(scene_list) * 12)
                _update_job(job_id, progress=pct, message=f"Narration {idx + 1}/{len(scene_list)} done ({narration_durations[-1]:.1f}s)")

        # Step 3 -- Generate video for each scene
        # Free any SDXL/other pipelines from portrait generation to reclaim VRAM
        import gc
        import torch as _torch
        try:
            from reelforge import _sdxl_pipe
            import reelforge
            if reelforge._sdxl_pipe is not None:
                del reelforge._sdxl_pipe
                reelforge._sdxl_pipe = None
        except Exception:
            pass
        gc.collect()
        if _torch.cuda.is_available():
            _torch.cuda.empty_cache()
        _update_job(job_id, progress=29, message="VRAM cleared for video generation...")

        # Use shared seed for visual consistency across scenes
        import random as _rng
        shared_seed = _rng.randint(0, 2**31)

        # Use request params or sensible defaults
        req_frames = getattr(req, "num_frames", 49) or 49
        req_fps = getattr(req, "fps", 16) or 16
        req_resolution = getattr(req, "resolution", "480p") or "480p"

        # Quality presets: control inference steps and guidance scale
        # Higher steps = more detail and coherence, higher guidance = stronger prompt adherence
        quality = getattr(req, "quality_preset", "standard") or "standard"
        QUALITY_PRESETS = {
            "draft":    {"steps": 25, "guidance": 5.5},
            "standard": {"steps": 40, "guidance": 6.5},
            "high":     {"steps": 50, "guidance": 7.5},
        }
        qp = QUALITY_PRESETS.get(quality, QUALITY_PRESETS["standard"])
        _update_job(job_id, progress=29, message=f"Quality: {quality} ({qp['steps']} steps, guidance {qp['guidance']})")

        video_paths = []

        # Strong negative prompt to reduce AI artifacts
        neg_prompt = (
            "low quality, blurry, distorted, disfigured, bad anatomy, worst quality, "
            "objects appearing from nowhere, teleporting, sudden scene change, "
            "inconsistent subject, morphing, glitch, watermark, text overlay, subtitle, "
            "extra limbs, missing limbs, floating objects, unnatural movement, "
            "deformed face, ugly, duplicate, morbid, mutilated, poorly drawn face, "
            "bad proportions, cloned face, gross proportions, "
            "malformed limbs, missing arms, missing legs, fused fingers, too many fingers, "
            "mutation, extra fingers, extra arms, extra legs, long neck, "
            "cross-eyed, out of frame, cut off, jpeg artifacts, pixelated, grainy, "
            "static image, frozen, no motion, still frame, slideshow"
        )

        # --- Character reference images (for prompt injection, not IP-Adapter) ---
        # Wan T2V does not support IP-Adapter. Character consistency comes from
        # strong prompt-based descriptions injected per scene (handled below).
        # No silent fallback chains — if something fails, it fails visibly.
        ip_adapter_strength = getattr(req, "ip_adapter_strength", 0.6)
        ip_adapter_enabled = getattr(req, "ip_adapter_enabled", True)

        # Build character appearance lookup from request characters
        # (name -> detailed appearance description for prompt injection)
        char_appearances = {}
        if hasattr(req, "scenes") and req.scenes:
            # Characters come from the storyboard's character list
            for sc in req.scenes:
                for cname in (sc.characters or []):
                    if cname not in char_appearances:
                        char_appearances[cname] = cname  # placeholder
        # Also check for characters defined via CharacterDef in script request
        for ch in getattr(req, "characters", []) or []:
            if hasattr(ch, "name") and hasattr(ch, "appearance"):
                char_appearances[ch.name] = ch.appearance or ch.name

        # Merge with visual_anchor which often contains character descriptions
        global_char_desc = visual_identity or ""

        for idx, scene in enumerate(scene_list):
            scene_prompt = scene.get("visual", req.concept)

            # --- Character injection: embed appearance descriptions directly in prompt ---
            scene_chars = scene.get("characters", [])
            char_injection = ""
            if scene_chars:
                # Inject specific character descriptions for this scene
                char_parts = []
                for cname in scene_chars:
                    appearance = char_appearances.get(cname, "")
                    if appearance and appearance != cname:
                        char_parts.append(f"{cname} ({appearance})")
                    else:
                        char_parts.append(cname)
                char_injection = "Characters: " + ", ".join(char_parts) + ". "
            elif global_char_desc:
                char_injection = f"{global_char_desc}. "

            # Also inject any user-provided visual anchor
            user_anchor = getattr(req, "visual_anchor", "")
            if user_anchor and user_anchor != global_char_desc:
                char_injection += f"{user_anchor}. "

            # Build final prompt: character desc FIRST (most important), then scene visual
            scene_prompt = f"{char_injection}{scene_prompt}"

            # Add quality boosters
            scene_prompt = (
                f"{scene_prompt}, smooth continuous motion, cinematic quality, "
                f"consistent character appearance throughout, same person same face same outfit, "
                f"photorealistic, professional cinematography, sharp focus"
            )

            pct_base = 30 + int(idx / len(scene_list) * 55)
            char_info = f" [{len(scene_chars)} character(s)]" if scene_chars else ""
            _update_job(
                job_id,
                progress=pct_base,
                message=f"Generating scene {idx + 1}/{len(scene_list)}: {scene.get('title', '')}{char_info}",
            )

            scene_dur = scene.get("duration", 5)
            if not isinstance(scene_dur, (int, float)):
                scene_dur = 5
            scene_dur = max(3, min(30, float(scene_dur)))
            # Stretch to cover the actual narration so audio isn't truncated.
            # +0.4s for inter-scene silence padding the assembler adds later.
            if idx < len(narration_durations) and narration_durations[idx] > 0:
                scene_dur = max(scene_dur, narration_durations[idx] + 0.4)

            scene_clips = []

            # --- SDXL → Wan VACE pipeline ---
            # Step 1: SDXL generates scene IMAGE with characters baked in
            # Step 2: VACE animates it (first frame preserved, rest generated)
            # Characters WILL appear because they're in the source image.
            from scene_generator import generate_scene_video as _gen_scene

            _update_job(
                job_id, progress=pct_base + 1,
                message=f"Scene {idx+1}: SDXL image → VACE animation ({scene_dur:.0f}s)...",
            )

            vp = _gen_scene(
                prompt=scene_prompt,
                negative_prompt=neg_prompt,
                target_duration=scene_dur,
                fps=req_fps,
                num_inference_steps=qp["steps"],
                guidance_scale=qp["guidance"],
                seed=shared_seed + idx,
                preferred_engine=getattr(req, "video_engine", "auto") or "auto",
                progress_callback=lambda s, t, m: _update_job(
                    job_id,
                    progress=pct_base + int((s / t) * (55 / len(scene_list))),
                    message=f"Scene {idx+1}/{len(scene_list)}: {m}",
                ),
            )
            scene_clips.append(vp)

            video_paths.append(scene_clips)

        # Step 4 -- Assemble video: concatenate clips per scene, then build
        # a single audio track with proper timing (numpy-based, no moviepy audio bugs)
        from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips
        import numpy as np
        import soundfile as sf

        _update_job(job_id, progress=86, message="Assembling scenes...")

        # 4a: Build silent video for each scene (clips merged, trimmed to duration)
        final_scene_paths = []
        scene_durations = []  # actual duration of each scene video

        for idx, scene_clips in enumerate(video_paths):
            scene = scene_list[idx] if idx < len(scene_list) else {}
            target_dur = scene.get("duration", 5)
            if not isinstance(target_dur, (int, float)):
                target_dur = 5
            target_dur = max(3, min(30, float(target_dur)))

            try:
                clips = [VideoFileClip(cp) for cp in scene_clips]
                # Always strip audio from video clips - we build audio separately
                silent_clips = [c.without_audio() for c in clips]
                if len(silent_clips) > 1:
                    combined = concatenate_videoclips(silent_clips, method="compose")
                else:
                    combined = silent_clips[0]

                # Trim to exact target duration
                if combined.duration > target_dur:
                    combined = combined.subclip(0, target_dur)

                scene_out = os.path.join(OUTPUT_DIR, f"story_scene_{job_id}_{idx}.mp4")
                combined.write_videofile(
                    scene_out, fps=req_fps, codec="libx264",
                    audio=False,  # No audio in scene clips
                    logger=None,
                )
                scene_durations.append(combined.duration)
                final_scene_paths.append(scene_out)

                for c in clips:
                    c.close()
                combined.close()
            except Exception as asm_err:
                _update_job(job_id, progress=86,
                    message=f"WARNING: Scene {idx+1} assembly failed ({asm_err}). Using raw clip.")
                if scene_clips:
                    final_scene_paths.append(scene_clips[0])
                    scene_durations.append(target_dur)

            pct = 86 + int((idx + 1) / len(video_paths) * 4)
            _update_job(job_id, progress=pct, message=f"Scene {idx+1} video assembled ({target_dur:.0f}s)")

        # 4a2: Optional lip sync (LatentSync 1.6) — apply per scene using its narration audio
        if getattr(req, "enable_lip_sync", False) and narration_paths:
            try:
                import lipsync as _lipsync
                if not _lipsync.available():
                    _update_job(
                        job_id, progress=89,
                        message=f"WARNING: Lip sync requested but not available. {_lipsync.install_hint()}",
                    )
                else:
                    _update_job(job_id, progress=89, message="Applying LatentSync 1.6 lip sync...")
                    # Free VRAM before loading LatentSync UNet
                    try:
                        import scene_generator as _sg
                        _sg._unload_i2v()
                    except Exception:
                        pass
                    synced_paths = _lipsync.lip_sync_scene_clips(
                        final_scene_paths,
                        narration_paths,
                        progress_callback=lambda i, n, m: _update_job(
                            job_id, progress=89 + int((i / max(n, 1)) * 1),
                            message=f"Lip sync {i+1}/{n}: {m}",
                        ),
                    )
                    final_scene_paths = synced_paths
            except Exception as lipsync_err:
                _update_job(
                    job_id, progress=89,
                    message=f"WARNING: Lip sync failed — {lipsync_err}. Continuing with original video.",
                )

        # 4b: Concatenate all scene videos into one silent video
        _update_job(job_id, progress=90, message="Merging scenes into final video...")
        final_video = concatenate_videos(final_scene_paths, fps=req_fps)

        # 4c: Build a single audio track with proper per-scene timing and gaps
        # This avoids moviepy's audio concatenation bugs (overlap, drift, repetition)
        if narration_paths:
            _update_job(job_id, progress=91, message="Building synchronized audio track...")
            INTER_SCENE_SILENCE = 0.4  # seconds of silence between scenes
            TTS_SAMPLE_RATE = 22050  # Piper TTS output rate

            # Detect sample rate from first valid narration file
            for np_path in narration_paths:
                if np_path and os.path.exists(np_path) and os.path.getsize(np_path) > 100:
                    try:
                        _, sr = sf.read(np_path, frames=1)
                        TTS_SAMPLE_RATE = sr
                    except Exception:
                        pass
                    break

            audio_segments = []
            for idx, (narr_path, scene_dur) in enumerate(zip(narration_paths, scene_durations)):
                scene_samples = int(scene_dur * TTS_SAMPLE_RATE)

                if narr_path and os.path.exists(narr_path) and os.path.getsize(narr_path) > 100:
                    try:
                        narr_audio, sr = sf.read(narr_path)
                        # Convert stereo to mono if needed
                        if len(narr_audio.shape) > 1:
                            narr_audio = np.mean(narr_audio, axis=1)
                        # Resample if needed
                        if sr != TTS_SAMPLE_RATE:
                            from scipy import signal as sp_signal
                            narr_audio = sp_signal.resample(
                                narr_audio, int(len(narr_audio) * TTS_SAMPLE_RATE / sr)
                            )

                        if len(narr_audio) > scene_samples:
                            # Narration longer than scene: trim with fade-out
                            narr_audio = narr_audio[:scene_samples]
                            fade_samples = min(int(0.3 * TTS_SAMPLE_RATE), len(narr_audio))
                            narr_audio[-fade_samples:] *= np.linspace(1, 0, fade_samples)
                        elif len(narr_audio) < scene_samples:
                            # Narration shorter: pad with silence to fill scene duration
                            padding = np.zeros(scene_samples - len(narr_audio), dtype=np.float32)
                            narr_audio = np.concatenate([narr_audio, padding])

                        audio_segments.append(narr_audio.astype(np.float32))
                    except Exception as narr_err:
                        _update_job(job_id, progress=91,
                            message=f"WARNING: Scene {idx+1} narration failed to load ({narr_err}). Using silence.")
                        audio_segments.append(np.zeros(scene_samples, dtype=np.float32))
                else:
                    # No narration for this scene, fill with silence
                    audio_segments.append(np.zeros(scene_samples, dtype=np.float32))

                # Add inter-scene silence gap (except after last scene)
                if idx < len(scene_durations) - 1:
                    gap = np.zeros(int(INTER_SCENE_SILENCE * TTS_SAMPLE_RATE), dtype=np.float32)
                    audio_segments.append(gap)

            # Combine all audio segments
            combined_audio = np.concatenate(audio_segments)

            # Normalize to prevent clipping
            peak = np.max(np.abs(combined_audio))
            if peak > 0.95:
                combined_audio = combined_audio * 0.92 / peak

            # Write combined audio
            combined_audio_path = os.path.join(OUTPUT_DIR, f"story_audio_{job_id}.wav")
            sf.write(combined_audio_path, combined_audio, TTS_SAMPLE_RATE)

            # Merge audio onto the silent video using ffmpeg (most reliable)
            final_with_audio = os.path.join(OUTPUT_DIR, f"story_final_{job_id}.mp4")
            try:
                import subprocess
                # Get video duration
                vid_clip = VideoFileClip(final_video)
                vid_duration = vid_clip.duration
                vid_clip.close()

                subprocess.run([
                    "ffmpeg", "-y",
                    "-i", final_video,
                    "-i", combined_audio_path,
                    "-c:v", "copy",  # Don't re-encode video
                    "-c:a", "aac", "-b:a", "192k",
                    "-t", str(vid_duration),  # Match video duration exactly
                    "-shortest",
                    final_with_audio,
                ], capture_output=True, check=True, timeout=120)
                final_video = final_with_audio
            except Exception as e:
                _update_job(
                    job_id, progress=92,
                    message=f"WARNING: Audio merge failed ({e}). Video will have no audio.",
                )

        # Step 6 -- Post-processing: upscale and frame interpolation
        enable_upscale = getattr(req, "enable_upscale", False)
        enable_interp = getattr(req, "enable_interpolation", False)

        if enable_upscale:
            _update_job(job_id, progress=93, message="Upscaling video with Real-ESRGAN (2x)...")
            try:
                from upscaler import upscale_video, check_upscaler_available
                available, backends = check_upscaler_available()
                method = "realesrgan" if "realesrgan" in backends else "lanczos"
                upscaled_path = upscale_video(
                    final_video, scale=2, method=method,
                    progress_callback=lambda cur, tot, msg: _update_job(
                        job_id, progress=93 + int((cur / max(tot, 1)) * 3),
                        message=f"Upscaling: {msg}",
                    ),
                )
                if upscaled_path and os.path.exists(upscaled_path):
                    final_video = upscaled_path
                    _update_job(job_id, progress=96, message=f"Upscaled with {method}")
            except Exception as e:
                _update_job(job_id, status="failed", error=f"Upscale failed: {e}",
                    message=f"Upscale failed: {e}. Install Real-ESRGAN or disable upscaling.")
                return

        if enable_interp:
            _update_job(job_id, progress=97, message="Interpolating frames (RIFE 2x)...")
            try:
                from rife_ncnn_vulkan_python import Rife
                # Double the frame rate by interpolating between every pair of frames
                from video_editor import extract_frames, frames_to_video
                from PIL import Image as PILImage
                import numpy as np
                import tempfile

                frames_dir = tempfile.mkdtemp()
                frame_paths = extract_frames(final_video, frames_dir)

                if len(frame_paths) > 1:
                    rife = Rife(gpuid=0)
                    interp_frames = []
                    for fi in range(len(frame_paths)):
                        interp_frames.append(frame_paths[fi])
                        if fi < len(frame_paths) - 1:
                            img0 = PILImage.open(frame_paths[fi])
                            img1 = PILImage.open(frame_paths[fi + 1])
                            mid = rife.process(np.array(img0), np.array(img1))
                            mid_path = os.path.join(frames_dir, f"interp_{fi:05d}.png")
                            PILImage.fromarray(mid).save(mid_path)
                            interp_frames.append(mid_path)
                        _update_job(
                            job_id, progress=97 + int((fi / len(frame_paths)) * 2),
                            message=f"Interpolating frame {fi+1}/{len(frame_paths)}",
                        )

                    interp_video = os.path.join(OUTPUT_DIR, f"story_interp_{job_id}.mp4")
                    frames_to_video(interp_frames, interp_video, fps=req_fps * 2)

                    # Preserve audio from original
                    from moviepy.editor import VideoFileClip, AudioFileClip
                    orig_clip = VideoFileClip(final_video)
                    if orig_clip.audio:
                        interp_clip = VideoFileClip(interp_video)
                        interp_clip = interp_clip.set_audio(orig_clip.audio)
                        final_interp = os.path.join(OUTPUT_DIR, f"story_smooth_{job_id}.mp4")
                        interp_clip.write_videofile(final_interp, codec="libx264", audio_codec="aac", logger=None)
                        interp_clip.close()
                        final_video = final_interp
                    else:
                        final_video = interp_video
                    orig_clip.close()

                    _update_job(job_id, progress=99, message=f"Frame interpolation done ({req_fps}→{req_fps*2} fps)")
            except ImportError:
                _update_job(job_id, progress=99, message="RIFE not installed, skipping interpolation (pip install rife-ncnn-vulkan-python)")
            except Exception as e:
                _update_job(job_id, status="failed", error=f"Frame interpolation failed: {e}",
                    message=f"RIFE interpolation failed: {e}. Install rife-ncnn-vulkan-python or disable interpolation.")
                return

        _update_job(
            job_id,
            status="completed",
            progress=100,
            message=f"Story generation complete! Quality: {quality}"
            + (" + upscaled" if enable_upscale else "")
            + (" + smoothed" if enable_interp else ""),
            result={
                "video_path": final_video,
                "scene_videos": video_paths,
                "narration_audio": narration_paths,
                "scenes": scene_list,
                "visual_identity": visual_identity,
                "quality_preset": quality,
            },
        )
    except Exception as exc:
        _update_job(
            job_id,
            status="failed",
            error=str(exc),
            message=f"Story generation failed: {exc}",
        )


def _run_sfx(job_id: str, req: SFXRequest):
    try:
        label = req.sfx_type or req.prompt or "sound effect"
        _update_job(job_id, status="running", message=f"Generating {label} ({req.duration}s)...")
        from audio_studio import generate_sfx

        output_path = os.path.join(OUTPUT_DIR, f"sfx_{job_id}.wav")
        info = generate_sfx(
            sfx_type=req.sfx_type,
            prompt=req.prompt,
            duration=req.duration,
            output_path=output_path,
        )
        if not info or not os.path.isfile(info["path"]):
            raise RuntimeError("SFX generator returned no file")

        _update_job(
            job_id,
            status="completed",
            progress=100,
            message=f"{info['sfx_type']} ready ({info['method']}, {info['channels']}ch @ {info['sample_rate']}Hz)",
            result={"audio_path": info["path"], **info},
        )
    except Exception as exc:
        _update_job(job_id, status="failed", error=str(exc), message=f"SFX failed: {exc}")


def _run_tts(job_id: str, req: TTSRequest):
    try:
        _update_job(job_id, status="running", message="Initializing TTS engine...")
        from mpv2.classes.TtsFactory import get_tts_instance
        from reelforge import rf_clean_text_for_tts
        from audio_studio import TTS_PERSONAS, apply_audio_effects

        # Persona is a label only at this stage. The frontend snaps the
        # controls when a persona is picked, so by the time we get the
        # request the values on the sliders ARE what should run. No
        # silent overrides on the server.
        persona = TTS_PERSONAS.get(req.persona) if req.persona else None
        voice = req.voice or "Amy"
        speed = req.speed
        pitch = req.pitch
        volume = req.volume

        tts = get_tts_instance(req.engine)
        if req.engine == "piper" and voice:
            from mpv2.classes.PiperTts import PiperTTS
            tts = PiperTTS(voice_name=voice)

        clean_text = rf_clean_text_for_tts(req.text)
        raw_path = os.path.join(OUTPUT_DIR, f"tts_{job_id}_raw.wav")

        _update_job(job_id, progress=30, message=f"Synthesizing with {voice}...")
        tts.synthesize(clean_text, raw_path)

        # Post-process if any effect is non-identity OR the user wants mp3.
        needs_proc = (
            abs(speed - 1.0) > 0.005
            or abs(pitch) > 0.05
            or abs(volume - 1.0) > 0.005
            or req.output_format == "mp3"
        )
        if needs_proc:
            _update_job(job_id, progress=70, message="Applying voice effects...")
            ext = "mp3" if req.output_format == "mp3" else "wav"
            output_path = os.path.join(OUTPUT_DIR, f"tts_{job_id}.{ext}")
            apply_audio_effects(
                raw_path, output_path,
                speed=speed, pitch_semitones=pitch, volume=volume,
                output_format=ext,
            )
            try:
                if os.path.abspath(raw_path) != os.path.abspath(output_path):
                    os.remove(raw_path)
            except OSError:
                pass
        else:
            output_path = raw_path

        _update_job(
            job_id,
            status="completed",
            progress=100,
            message=f"TTS ready ({voice}, persona={req.persona or 'none'}).",
            result={
                "audio_path": output_path,
                "voice": voice,
                "engine": req.engine,
                "persona": req.persona or None,
                "persona_label": persona["label"] if persona else None,
                "speed": speed,
                "pitch": pitch,
                "volume": volume,
                "output_format": "mp3" if req.output_format == "mp3" else "wav",
            },
        )
    except Exception as exc:
        _update_job(
            job_id,
            status="failed",
            error=str(exc),
            message=f"TTS failed: {exc}",
        )


def _run_trim(job_id: str, req: TrimRequest):
    try:
        _update_job(job_id, status="running", message="Trimming video...")

        if not os.path.isfile(req.video_path):
            raise FileNotFoundError(f"Video not found: {req.video_path}")
        if req.end_time <= req.start_time:
            raise ValueError("end_time must be greater than start_time")

        from moviepy.editor import VideoFileClip

        clip = VideoFileClip(req.video_path)
        if req.end_time > clip.duration:
            clip.close()
            raise ValueError(
                f"end_time ({req.end_time}s) exceeds video duration ({clip.duration:.2f}s)"
            )

        _update_job(job_id, progress=30, message="Cutting clip...")
        trimmed = clip.subclip(req.start_time, req.end_time)

        output_path = os.path.join(OUTPUT_DIR, f"trimmed_{job_id}.mp4")
        trimmed.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac" if trimmed.audio else None,
            logger=None,
        )
        trimmed.close()
        clip.close()

        _update_job(
            job_id,
            status="completed",
            progress=100,
            message="Trim complete.",
            result={
                "video_path": output_path,
                "start_time": req.start_time,
                "end_time": req.end_time,
            },
        )
    except Exception as exc:
        _update_job(
            job_id,
            status="failed",
            error=str(exc),
            message=f"Trim failed: {exc}",
        )


def _run_merge(job_id: str, req: MergeRequest):
    try:
        _update_job(job_id, status="running", message="Merging videos...")

        if len(req.video_paths) < 2:
            raise ValueError("At least 2 video paths are required for merging")
        for p in req.video_paths:
            if not os.path.isfile(p):
                raise FileNotFoundError(f"Video not found: {p}")

        from moviepy.editor import VideoFileClip, concatenate_videoclips

        clips = [VideoFileClip(p) for p in req.video_paths]
        _update_job(job_id, progress=30, message=f"Loaded {len(clips)} clips, concatenating...")

        if req.transition == "crossfade" and req.transition_duration > 0:
            # Apply crossfade padding between clips
            padded = []
            for i, clip in enumerate(clips):
                if i > 0:
                    padded.append(
                        clip.crossfadein(req.transition_duration)
                    )
                else:
                    padded.append(clip)
            final = concatenate_videoclips(padded, method="compose", padding=-req.transition_duration)
        else:
            final = concatenate_videoclips(clips, method="compose")

        output_path = os.path.join(OUTPUT_DIR, f"merged_{job_id}.mp4")
        final.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac" if final.audio else None,
            logger=None,
        )
        final.close()
        for c in clips:
            c.close()

        _update_job(
            job_id,
            status="completed",
            progress=100,
            message="Merge complete.",
            result={
                "video_path": output_path,
                "input_count": len(req.video_paths),
                "transition": req.transition,
            },
        )
    except Exception as exc:
        _update_job(
            job_id,
            status="failed",
            error=str(exc),
            message=f"Merge failed: {exc}",
        )


def _run_upscale(job_id: str, req: UpscaleRequest):
    try:
        _update_job(job_id, status="running", message="Upscaling video...")

        if not os.path.isfile(req.video_path):
            raise FileNotFoundError(f"Video not found: {req.video_path}")

        from moviepy.editor import VideoFileClip

        clip = VideoFileClip(req.video_path)
        new_width = int(clip.w * req.scale)
        new_height = int(clip.h * req.scale)

        _update_job(
            job_id,
            progress=20,
            message=f"Resizing from {clip.w}x{clip.h} to {new_width}x{new_height}...",
        )

        resized = clip.resize((new_width, new_height))
        output_path = os.path.join(OUTPUT_DIR, f"upscaled_{job_id}.mp4")
        resized.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac" if resized.audio else None,
            bitrate="8000k",
            logger=None,
        )
        resized.close()
        clip.close()

        _update_job(
            job_id,
            status="completed",
            progress=100,
            message="Upscale complete.",
            result={
                "video_path": output_path,
                "original_resolution": f"{clip.w}x{clip.h}",
                "new_resolution": f"{new_width}x{new_height}",
                "scale": req.scale,
            },
        )
    except Exception as exc:
        _update_job(
            job_id,
            status="failed",
            error=str(exc),
            message=f"Upscale failed: {exc}",
        )


def _run_ffmpeg(cmd: list[str]) -> None:
    import subprocess

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "unknown ffmpeg error").strip()
        raise RuntimeError(detail.splitlines()[-1] if detail else "unknown ffmpeg error")


def _require_video_path(video_path: str) -> None:
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")


def _probe_has_audio(path: str) -> bool:
    import subprocess

    proc = subprocess.run([
        "ffprobe", "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "stream=index",
        "-of", "csv=p=0",
        path,
    ], capture_output=True, text=True)
    return proc.returncode == 0 and bool(proc.stdout.strip())


def _atempo_chain(speed: float) -> str:
    parts = []
    remaining = speed
    while remaining > 2.0:
        parts.append("atempo=2.0")
        remaining /= 2.0
    while remaining < 0.5:
        parts.append("atempo=0.5")
        remaining /= 0.5
    parts.append(f"atempo={remaining:.4f}")
    return ",".join(parts)


def _run_extract_audio(job_id: str, req: ExtractAudioRequest):
    try:
        _update_job(job_id, status="running", progress=10, message="Extracting audio...")
        _require_video_path(req.video_path)
        fmt = req.format.lower().strip()
        if fmt not in {"wav", "mp3"}:
            raise ValueError("format must be wav or mp3")

        output_path = os.path.join(OUTPUT_DIR, f"audio_{job_id}.{fmt}")
        cmd = ["ffmpeg", "-y", "-i", req.video_path, "-vn"]
        if fmt == "mp3":
            cmd += ["-codec:a", "libmp3lame", "-q:a", "2"]
        else:
            cmd += ["-acodec", "pcm_s16le"]
        cmd.append(output_path)
        _run_ffmpeg(cmd)

        if not os.path.exists(output_path) or os.path.getsize(output_path) <= 100:
            raise RuntimeError("No audio track was extracted from this video")
        _update_job(
            job_id,
            status="completed",
            progress=100,
            message="Audio extracted.",
            result={"audio_path": output_path, "format": fmt},
        )
    except Exception as exc:
        _update_job(job_id, status="failed", error=str(exc), message=f"Audio extraction failed: {exc}")


def _run_remove_audio(job_id: str, req: VideoPathRequest):
    try:
        _update_job(job_id, status="running", progress=10, message="Removing audio...")
        _require_video_path(req.video_path)
        output_path = os.path.join(OUTPUT_DIR, f"muted_{job_id}.mp4")
        _run_ffmpeg(["ffmpeg", "-y", "-i", req.video_path, "-c:v", "copy", "-an", output_path])
        _update_job(
            job_id,
            status="completed",
            progress=100,
            message="Audio removed.",
            result={"video_path": output_path},
        )
    except Exception as exc:
        _update_job(job_id, status="failed", error=str(exc), message=f"Remove audio failed: {exc}")


def _run_compress(job_id: str, req: CompressRequest):
    try:
        _update_job(job_id, status="running", progress=10, message="Compressing video...")
        _require_video_path(req.video_path)
        crf_by_preset = {"high": "23", "medium": "28", "low": "32"}
        preset = req.preset.lower().strip()
        if preset not in crf_by_preset:
            raise ValueError("preset must be high, medium, or low")

        output_path = os.path.join(OUTPUT_DIR, f"compressed_{job_id}.mp4")
        _run_ffmpeg([
            "ffmpeg", "-y", "-i", req.video_path,
            "-c:v", "libx264", "-preset", "medium", "-crf", crf_by_preset[preset],
            "-c:a", "aac", "-b:a", "128k",
            output_path,
        ])
        _update_job(
            job_id,
            status="completed",
            progress=100,
            message="Compression complete.",
            result={"video_path": output_path, "preset": preset},
        )
    except Exception as exc:
        _update_job(job_id, status="failed", error=str(exc), message=f"Compression failed: {exc}")


def _run_rotate_flip(job_id: str, req: RotateFlipRequest):
    try:
        _update_job(job_id, status="running", progress=10, message="Transforming video...")
        _require_video_path(req.video_path)
        if req.rotate not in {0, 90, 180, 270}:
            raise ValueError("rotate must be one of 0, 90, 180, 270")
        flip = req.flip.lower().strip()
        if flip not in {"none", "horizontal", "vertical"}:
            raise ValueError("flip must be none, horizontal, or vertical")

        filters = []
        if req.rotate == 90:
            filters.append("transpose=1")
        elif req.rotate == 180:
            filters.append("transpose=1,transpose=1")
        elif req.rotate == 270:
            filters.append("transpose=2")
        if flip == "horizontal":
            filters.append("hflip")
        elif flip == "vertical":
            filters.append("vflip")

        output_path = os.path.join(OUTPUT_DIR, f"transformed_{job_id}.mp4")
        cmd = ["ffmpeg", "-y", "-i", req.video_path]
        if filters:
            cmd += ["-vf", ",".join(filters), "-c:v", "libx264", "-preset", "fast", "-crf", "18"]
        else:
            cmd += ["-c:v", "copy"]
        cmd += ["-c:a", "copy", output_path]
        _run_ffmpeg(cmd)
        _update_job(
            job_id,
            status="completed",
            progress=100,
            message="Transform complete.",
            result={"video_path": output_path, "rotate": req.rotate, "flip": flip},
        )
    except Exception as exc:
        _update_job(job_id, status="failed", error=str(exc), message=f"Transform failed: {exc}")


def _run_reverse(job_id: str, req: ReverseRequest):
    try:
        _update_job(job_id, status="running", progress=10, message="Reversing video...")
        _require_video_path(req.video_path)
        output_path = os.path.join(OUTPUT_DIR, f"reversed_{job_id}.mp4")
        filter_complex = "[0:v]reverse[v];[0:a]areverse[a]" if req.include_audio else "[0:v]reverse[v]"
        cmd = ["ffmpeg", "-y", "-i", req.video_path, "-filter_complex", filter_complex, "-map", "[v]"]
        if req.include_audio:
            cmd += ["-map", "[a]"]
        else:
            cmd += ["-an"]
        cmd += ["-c:v", "libx264", "-preset", "fast", "-crf", "18", "-c:a", "aac", output_path]
        _run_ffmpeg(cmd)
        _update_job(
            job_id,
            status="completed",
            progress=100,
            message="Reverse complete.",
            result={"video_path": output_path, "include_audio": req.include_audio},
        )
    except Exception as exc:
        _update_job(job_id, status="failed", error=str(exc), message=f"Reverse failed: {exc}")


def _run_loop(job_id: str, req: LoopRequest):
    try:
        _update_job(job_id, status="running", progress=10, message=f"Looping video {req.count} times...")
        _require_video_path(req.video_path)
        output_path = os.path.join(OUTPUT_DIR, f"looped_{job_id}.mp4")
        _run_ffmpeg([
            "ffmpeg", "-y", "-stream_loop", str(req.count - 1), "-i", req.video_path,
            "-c", "copy", output_path,
        ])
        _update_job(
            job_id,
            status="completed",
            progress=100,
            message="Loop complete.",
            result={"video_path": output_path, "count": req.count},
        )
    except Exception as exc:
        _update_job(job_id, status="failed", error=str(exc), message=f"Loop failed: {exc}")


def _run_stabilize(job_id: str, req: StabilizeRequest):
    try:
        _update_job(job_id, status="running", progress=10, message="Analyzing camera motion...")
        _require_video_path(req.video_path)
        transforms = f"stabilize_{job_id}.trf"
        output_path = os.path.join(OUTPUT_DIR, f"stabilized_{job_id}.mp4")
        _run_ffmpeg([
            "ffmpeg", "-y", "-i", req.video_path,
            "-vf", f"vidstabdetect=shakiness={req.shakiness}:accuracy={req.accuracy}:result={transforms}",
            "-f", "null", os.devnull,
        ])
        _update_job(job_id, progress=55, message="Applying stabilization...")
        _run_ffmpeg([
            "ffmpeg", "-y", "-i", req.video_path,
            "-vf", f"vidstabtransform=input={transforms}:zoom=5:smoothing=30,unsharp=5:5:0.8:3:3:0.4",
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-c:a", "copy", output_path,
        ])
        _update_job(
            job_id,
            status="completed",
            progress=100,
            message="Stabilization complete.",
            result={"video_path": output_path, "shakiness": req.shakiness, "accuracy": req.accuracy},
        )
    except Exception as exc:
        _update_job(job_id, status="failed", error=str(exc), message=f"Stabilization failed: {exc}")


def _run_color_correction(job_id: str, req: ColorCorrectionRequest):
    try:
        _update_job(job_id, status="running", progress=10, message="Applying color correction...")
        _require_video_path(req.video_path)
        output_path = os.path.join(OUTPUT_DIR, f"color_{job_id}.mp4")
        vf = (
            f"eq=brightness={req.brightness}:contrast={req.contrast}:saturation={req.saturation},"
            f"hue=h={req.hue}"
        )
        _run_ffmpeg([
            "ffmpeg", "-y", "-i", req.video_path,
            "-vf", vf,
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-c:a", "copy", output_path,
        ])
        _update_job(
            job_id,
            status="completed",
            progress=100,
            message="Color correction complete.",
            result={"video_path": output_path},
        )
    except Exception as exc:
        _update_job(job_id, status="failed", error=str(exc), message=f"Color correction failed: {exc}")


def _run_region_effect(job_id: str, req: RegionEffectRequest):
    try:
        _update_job(job_id, status="running", progress=10, message="Applying region effect...")
        _require_video_path(req.video_path)
        mode = req.mode.lower().strip()
        if mode not in {"blur", "pixelate"}:
            raise ValueError("mode must be blur or pixelate")

        output_path = os.path.join(OUTPUT_DIR, f"region_{mode}_{job_id}.mp4")
        crop = f"crop={req.width}:{req.height}:{req.x}:{req.y}"
        if mode == "blur":
            effect = f"{crop},gblur=sigma={req.strength}"
        else:
            block = max(4, min(req.strength, 80))
            effect = f"{crop},scale=iw/{block}:ih/{block},scale={req.width}:{req.height}:flags=neighbor"
        filter_complex = f"[0:v]split[base][tmp];[tmp]{effect}[fg];[base][fg]overlay={req.x}:{req.y}[v]"
        _run_ffmpeg([
            "ffmpeg", "-y", "-i", req.video_path,
            "-filter_complex", filter_complex,
            "-map", "[v]", "-map", "0:a?",
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-c:a", "copy", output_path,
        ])
        _update_job(
            job_id,
            status="completed",
            progress=100,
            message="Region effect complete.",
            result={"video_path": output_path, "mode": mode},
        )
    except Exception as exc:
        _update_job(job_id, status="failed", error=str(exc), message=f"Region effect failed: {exc}")


def _run_picture_in_picture(job_id: str, req: PictureInPictureRequest):
    try:
        _update_job(job_id, status="running", progress=10, message="Compositing picture-in-picture...")
        _require_video_path(req.video_path)
        _require_video_path(req.overlay_path)
        positions = {
            "top_left": f"{req.margin}:{req.margin}",
            "top_right": f"main_w-overlay_w-{req.margin}:{req.margin}",
            "bottom_left": f"{req.margin}:main_h-overlay_h-{req.margin}",
            "bottom_right": f"main_w-overlay_w-{req.margin}:main_h-overlay_h-{req.margin}",
            "center": "(main_w-overlay_w)/2:(main_h-overlay_h)/2",
        }
        position = req.position.lower().strip()
        if position not in positions:
            raise ValueError("position must be top_left, top_right, bottom_left, bottom_right, or center")

        output_path = os.path.join(OUTPUT_DIR, f"pip_{job_id}.mp4")
        scale_expr = f"[1:v][0:v]scale2ref=w=main_w*{req.size_percent}/100:h=-1[ov][base]"
        filter_complex = f"{scale_expr};[base][ov]overlay={positions[position]}[v]"
        _run_ffmpeg([
            "ffmpeg", "-y", "-i", req.video_path, "-i", req.overlay_path,
            "-filter_complex", filter_complex,
            "-map", "[v]", "-map", "0:a?",
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-c:a", "copy", "-shortest", output_path,
        ])
        _update_job(
            job_id,
            status="completed",
            progress=100,
            message="Picture-in-picture complete.",
            result={"video_path": output_path, "position": position, "size_percent": req.size_percent},
        )
    except Exception as exc:
        _update_job(job_id, status="failed", error=str(exc), message=f"Picture-in-picture failed: {exc}")


def _run_background_music(job_id: str, req: BackgroundMusicRequest):
    try:
        _update_job(job_id, status="running", progress=10, message="Adding background music...")
        _require_video_path(req.video_path)
        if not os.path.isfile(req.audio_path):
            raise FileNotFoundError(f"Audio not found: {req.audio_path}")

        output_path = os.path.join(OUTPUT_DIR, f"music_{job_id}.mp4")
        if _probe_has_audio(req.video_path):
            filter_complex = (
                f"[0:a]volume={req.video_volume}[a0];"
                f"[1:a]volume={req.music_volume}[a1];"
                "[a0][a1]amix=inputs=2:duration=first:dropout_transition=2[a]"
            )
            cmd = [
                "ffmpeg", "-y", "-i", req.video_path, "-stream_loop", "-1", "-i", req.audio_path,
                "-filter_complex", filter_complex,
                "-map", "0:v", "-map", "[a]",
                "-c:v", "copy", "-c:a", "aac", "-shortest", output_path,
            ]
        else:
            cmd = [
                "ffmpeg", "-y", "-i", req.video_path, "-stream_loop", "-1", "-i", req.audio_path,
                "-map", "0:v", "-map", "1:a",
                "-filter:a", f"volume={req.music_volume}",
                "-c:v", "copy", "-c:a", "aac", "-shortest", output_path,
            ]
        _run_ffmpeg(cmd)
        _update_job(
            job_id,
            status="completed",
            progress=100,
            message="Background music added.",
            result={"video_path": output_path},
        )
    except Exception as exc:
        _update_job(job_id, status="failed", error=str(exc), message=f"Background music failed: {exc}")


def _run_speed(job_id: str, req: SpeedRequest):
    try:
        _update_job(job_id, status="running", progress=10, message=f"Changing speed to {req.speed}x...")
        _require_video_path(req.video_path)
        output_path = os.path.join(OUTPUT_DIR, f"speed_{job_id}.mp4")
        has_audio = req.keep_audio and _probe_has_audio(req.video_path)
        if has_audio:
            filter_complex = f"[0:v]setpts=PTS/{req.speed}[v];[0:a]{_atempo_chain(req.speed)}[a]"
            cmd = [
                "ffmpeg", "-y", "-i", req.video_path,
                "-filter_complex", filter_complex,
                "-map", "[v]", "-map", "[a]",
                "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                "-c:a", "aac", output_path,
            ]
        else:
            cmd = [
                "ffmpeg", "-y", "-i", req.video_path,
                "-vf", f"setpts=PTS/{req.speed}",
                "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                output_path,
            ]
        _run_ffmpeg(cmd)
        _update_job(
            job_id,
            status="completed",
            progress=100,
            message="Speed change complete.",
            result={"video_path": output_path, "speed": req.speed, "keep_audio": req.keep_audio},
        )
    except Exception as exc:
        _update_job(job_id, status="failed", error=str(exc), message=f"Speed change failed: {exc}")


def _run_gif(job_id: str, req: GifRequest):
    try:
        _update_job(job_id, status="running", progress=10, message="Creating GIF...")
        _require_video_path(req.video_path)
        output_path = os.path.join(OUTPUT_DIR, f"gif_{job_id}.gif")
        filter_complex = (
            f"fps={req.fps},scale={req.width}:-1:flags=lanczos,"
            "split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse"
        )
        _run_ffmpeg([
            "ffmpeg", "-y",
            "-ss", str(req.start_time), "-t", str(req.duration),
            "-i", req.video_path,
            "-filter_complex", filter_complex,
            output_path,
        ])
        _update_job(
            job_id,
            status="completed",
            progress=100,
            message="GIF complete.",
            result={"image_path": output_path, "start_time": req.start_time, "duration": req.duration},
        )
    except Exception as exc:
        _update_job(job_id, status="failed", error=str(exc), message=f"GIF creation failed: {exc}")


def _run_thumbnail(job_id: str, req: ThumbnailRequest):
    try:
        _update_job(job_id, status="running", progress=10, message="Extracting thumbnail...")
        _require_video_path(req.video_path)
        output_path = os.path.join(OUTPUT_DIR, f"thumbnail_{job_id}.png")
        _run_ffmpeg([
            "ffmpeg", "-y",
            "-ss", str(req.timestamp),
            "-i", req.video_path,
            "-frames:v", "1",
            "-vf", f"scale={req.width}:-1",
            output_path,
        ])
        _update_job(
            job_id,
            status="completed",
            progress=100,
            message="Thumbnail extracted.",
            result={"image_path": output_path, "timestamp": req.timestamp},
        )
    except Exception as exc:
        _update_job(job_id, status="failed", error=str(exc), message=f"Thumbnail extraction failed: {exc}")


# ---------------------------------------------------------------------------
# Image Studio runners (background workers)
# ---------------------------------------------------------------------------

def _img_progress_cb(job_id: str):
    return lambda s, t, m: _update_job(
        job_id, progress=int((s / max(t, 1)) * 100), message=m,
    )


def _run_image_generate(job_id: str, req: ImageGenerateRequest):
    try:
        _update_job(job_id, status="running", message="Loading image generator...")
        import imagegen

        prompt = req.prompt
        negative = req.negative_prompt
        if req.enhance_prompt and prompt.strip():
            _update_job(job_id, progress=4, message="Enhancing prompt with LLM...")
            res = imagegen.enhance_prompt(prompt, style=req.style, mode="expand",
                                          negative_prompt=negative)
            prompt = res.get("enhanced", prompt)
            if res.get("negative") and not negative:
                negative = res["negative"]

        out = imagegen.text_to_image(
            prompt=prompt,
            negative_prompt=negative or "",
            size=req.size,
            width=req.width,
            height=req.height,
            style=req.style,
            seed=req.seed,
            steps=req.steps,
            guidance=req.guidance,
            provider=req.provider,
            model=req.model,
            progress_callback=_img_progress_cb(job_id),
        )
        _update_job(
            job_id, status="completed", progress=100, message="Generated.",
            result={
                "image_path": out,
                "url": f"/static/images/{os.path.basename(out)}",
                "prompt_used": prompt,
                "negative_used": negative or "",
            },
        )
    except Exception as exc:
        logger.exception("image generate failed")
        _update_job(job_id, status="failed", error=str(exc), message=f"Failed: {exc}")


def _run_image_edit(job_id: str, req: ImageEditRequest):
    try:
        _update_job(job_id, status="running", message=f"Edit ({req.technique})...")
        import imagegen
        out = imagegen.image_to_image(
            image_path=req.image_path, prompt=req.prompt,
            negative_prompt=req.negative_prompt, strength=req.strength,
            style=req.style, seed=req.seed, steps=req.steps, guidance=req.guidance,
            provider=req.provider, technique=req.technique,
            image_guidance=req.image_guidance,
            progress_callback=_img_progress_cb(job_id),
        )
        _update_job(
            job_id, status="completed", progress=100, message="Edited.",
            result={"image_path": out, "url": f"/static/images/{os.path.basename(out)}"},
        )
    except Exception as exc:
        logger.exception("image edit failed")
        _update_job(job_id, status="failed", error=str(exc), message=f"Failed: {exc}")


def _run_image_inpaint(job_id: str, req: ImageInpaintRequest):
    try:
        _update_job(job_id, status="running", message="Loading inpaint pipeline...")
        import imagegen
        out = imagegen.inpaint(
            image_path=req.image_path, mask_path=req.mask_path,
            prompt=req.prompt, negative_prompt=req.negative_prompt,
            style=req.style, seed=req.seed, steps=req.steps, guidance=req.guidance,
            invert_mask=req.invert_mask, progress_callback=_img_progress_cb(job_id),
        )
        _update_job(
            job_id, status="completed", progress=100, message="Inpainted.",
            result={"image_path": out, "url": f"/static/images/{os.path.basename(out)}"},
        )
    except Exception as exc:
        logger.exception("image inpaint failed")
        _update_job(job_id, status="failed", error=str(exc), message=f"Failed: {exc}")


def _run_image_variation(job_id: str, req: ImageVariationRequest):
    try:
        _update_job(job_id, status="running", message="Generating variation...")
        import imagegen
        out = imagegen.variation(
            image_path=req.image_path, prompt=req.prompt,
            strength=req.strength, style=req.style, seed=req.seed,
            progress_callback=_img_progress_cb(job_id),
        )
        _update_job(
            job_id, status="completed", progress=100, message="Done.",
            result={"image_path": out, "url": f"/static/images/{os.path.basename(out)}"},
        )
    except Exception as exc:
        logger.exception("image variation failed")
        _update_job(job_id, status="failed", error=str(exc), message=f"Failed: {exc}")


def _run_image_upscale(job_id: str, req: ImageUpscaleRequest):
    try:
        _update_job(job_id, status="running", message=f"Upscaling {req.scale}x...")
        import imagegen
        out = imagegen.upscale(
            image_path=req.image_path, scale=req.scale, method=req.method,
            progress_callback=_img_progress_cb(job_id),
        )
        _update_job(
            job_id, status="completed", progress=100, message="Upscaled.",
            result={"image_path": out, "url": f"/static/images/{os.path.basename(out)}"},
        )
    except Exception as exc:
        logger.exception("image upscale failed")
        _update_job(job_id, status="failed", error=str(exc), message=f"Failed: {exc}")


def _run_image_remove_bg(job_id: str, req: ImageBgRemoveRequest):
    try:
        _update_job(job_id, status="running", message="Removing background...")
        import imagegen
        out = imagegen.remove_background(
            image_path=req.image_path, progress_callback=_img_progress_cb(job_id),
        )
        _update_job(
            job_id, status="completed", progress=100, message="Done.",
            result={"image_path": out, "url": f"/static/images/{os.path.basename(out)}"},
        )
    except Exception as exc:
        logger.exception("image remove-bg failed")
        _update_job(job_id, status="failed", error=str(exc), message=f"Failed: {exc}")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/v1/system/status")
async def system_status():
    """Get system status: GPU, VRAM, loaded models."""
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            total_vram = props.total_memory / (1024 ** 3)
            # Use nvidia-smi for actual system-wide free VRAM
            free_vram = total_vram
            try:
                import subprocess
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5,
                )
                if result.returncode == 0:
                    free_vram = float(result.stdout.strip()) / 1024
            except Exception:
                reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)
                free_vram = total_vram - reserved
            gpu_info = {
                "available": True,
                "gpu_name": props.name,
                "total_vram_gb": round(total_vram, 2),
                "free_vram_gb": round(free_vram, 2),
                "cuda_version": torch.version.cuda,
            }
        else:
            gpu_info = {"available": False, "message": "No CUDA GPU detected"}
    except ImportError:
        gpu_info = {"available": False, "message": "PyTorch not installed"}

    active_jobs = sum(1 for j in jobs.values() if j["status"] in ("queued", "running"))
    return {
        "status": "ok",
        "gpu": gpu_info,
        "active_jobs": active_jobs,
        "total_jobs": len(jobs),
        "output_dir": OUTPUT_DIR,
    }


@app.get("/api/v1/system/engines")
async def list_engines():
    """List available video generation engines (live availability) and TTS engines."""
    # Live availability from scene_generator (checks disk for weights)
    try:
        import scene_generator as _sg
        engine_list = _sg.list_available_engines()
    except Exception as e:
        logger.error("scene_generator import failed: %s", e)
        engine_list = []

    # Add rich metadata
    ENGINE_META = {
        "wan22_14b_gguf": {
            "tier": "quality",
            "description": "Wan 2.2 14B I2V (GGUF Q3) — highest quality, ~3-5 min/clip.",
            "resolution": "480p",
        },
        "hunyuan15_i2v_gguf": {
            "tier": "balanced",
            "description": "HunyuanVideo 1.5 I2V 480p (GGUF Q4) — cinematic motion, ~1-2 min/clip.",
            "resolution": "480p",
        },
        "ltx23_distilled_gguf": {
            "tier": "draft",
            "description": "LTX-2.3 distilled (GGUF Q3) — fastest, ~45s/clip, 24fps native.",
            "resolution": "480p",
        },
        "vace_1.3b": {
            "tier": "fallback",
            "description": "Wan VACE 1.3B — local fallback, lower quality.",
            "resolution": "480p",
        },
        "t2v_1.3b": {
            "tier": "last_resort",
            "description": "Wan T2V 1.3B — text-only, no image conditioning.",
            "resolution": "480p",
        },
    }
    video_engines = []
    for e in engine_list:
        meta = ENGINE_META.get(e["id"], {})
        video_engines.append({**e, **meta})

    # LatentSync lip-sync availability
    lip_sync = {"available": False, "install_hint": None}
    try:
        import lipsync as _ls
        lip_sync = {
            "available": _ls.available(),
            "install_hint": _ls.install_hint() if not _ls.available() else None,
        }
    except Exception:
        pass

    tts_engines = []
    try:
        from mpv2.classes.TtsFactory import list_available_engines

        tts_engines = list_available_engines()
    except ImportError:
        pass

    return {
        "video_engines": video_engines,
        "tts_engines": tts_engines,
        "lip_sync": lip_sync,
    }


@app.post("/api/v1/generate/text2video", response_model=JobResponse)
async def generate_text2video(req: TextToVideoRequest, background_tasks: BackgroundTasks):
    """Start a text-to-video generation job."""
    if not req.prompt.strip():
        raise HTTPException(status_code=422, detail="prompt must not be empty")

    job_id = _create_job("text2video", req.model_dump())
    thread = threading.Thread(target=_run_text2video, args=(job_id, req), daemon=True)
    thread.start()
    return _job_response(job_id)


@app.post("/api/v1/generate/image2video", response_model=JobResponse)
async def generate_image2video(req: ImageToVideoRequest, background_tasks: BackgroundTasks):
    """Start an image-to-video generation job."""
    if not req.prompt.strip():
        raise HTTPException(status_code=422, detail="prompt must not be empty")
    if not os.path.isfile(req.image_path):
        raise HTTPException(status_code=422, detail=f"Image file not found: {req.image_path}")

    job_id = _create_job("image2video", req.model_dump())
    thread = threading.Thread(target=_run_image2video, args=(job_id, req), daemon=True)
    thread.start()
    return _job_response(job_id)


class CharacterDef(BaseModel):
    name: str = ""
    role: str = "Lead"
    appearance: str = ""

class ScriptRequest(BaseModel):
    concept: str
    num_scenes: int = 4
    genre: str = "Cinematic"
    mood: str = "Epic"
    characters: list[CharacterDef] = []  # Characters to include in the script


@app.post("/api/v1/generate/script")
async def generate_script(req: ScriptRequest):
    """Generate a story script (scenes with titles, visuals, narration) WITHOUT video generation.
    This is synchronous and returns scenes directly."""
    if not req.concept.strip():
        raise HTTPException(status_code=422, detail="concept must not be empty")

    try:
        from mpv2.llm_provider import generate_text, select_backend, select_model
        from reelforge import load_config
        import json as _json, re as _re

        cfg = load_config()
        backend = cfg.get("llm_backend", "ollama")
        select_backend(backend)
        if backend == "ollama":
            select_model(cfg.get("ollama_model") or "llama3.2:3b")
        else:
            gguf = cfg.get("gguf_model", "")
            if gguf:
                select_model(gguf)

        # Build character block if characters provided
        char_block = ""
        if req.characters:
            char_lines = []
            for c in req.characters:
                if c.name.strip() and c.appearance.strip():
                    char_lines.append(f"- {c.name} ({c.role}): {c.appearance}")
            if char_lines:
                char_block = f"""
CHARACTERS (use these EXACT descriptions in EVERY scene):
{chr(10).join(char_lines)}

CRITICAL: Every scene's visual description MUST include the character name followed by their EXACT appearance description above. Copy the appearance word-for-word. Characters must look IDENTICAL across all scenes.
"""

        prompt = f"""You are a professional cinematographer writing a shot list for a {req.num_scenes}-scene short film.

STORY: {req.concept}
GENRE: {req.genre}
MOOD: {req.mood}
{char_block}
CRITICAL RULES FOR VISUAL DESCRIPTIONS:
- The MAIN SUBJECT must be described with IDENTICAL words in EVERY scene (same breed, same color, same clothing, same features)
- Example: if scene 1 says "a fluffy white and gray husky with bright blue eyes", then ALL scenes must use those EXACT same words for the dog
- NEVER introduce new characters or subjects that weren't established in scene 1
- Each visual description must specify: exact camera angle (close-up, medium, wide, aerial), camera movement (static, slow pan left, tracking), lighting (warm, cool, natural, dramatic), what the subject is DOING (specific action, not vague)
- Describe CONTINUOUS motion that the AI can generate as video - subjects moving smoothly, not teleporting
- Avoid describing rapid scene changes within one shot - each scene is ONE continuous camera shot

Return ONLY a JSON array (make sure to close the array with ]):
[
  {{"title": "Scene Title", "visual": "DETAILED shot description with EXACT same subject description, specific camera angle, specific motion, specific lighting", "narration": "Engaging voiceover 2-3 sentences", "duration": 5}},
  ...
]

The narration should tell a cohesive, engaging story that flows naturally scene to scene. Each scene's narration should connect to the previous one."""

        response = generate_text(prompt)
        response = response.replace("```json", "").replace("```", "").strip()

        # Parse - handle incomplete JSON from LLM
        scenes = []
        try:
            scenes = _json.loads(response)
        except _json.JSONDecodeError:
            # Try finding array with closing bracket
            match = _re.search(r'\[.*\]', response, _re.DOTALL)
            if match:
                try:
                    scenes = _json.loads(match.group())
                except _json.JSONDecodeError:
                    pass

            # LLM often returns array without closing bracket - fix it
            if not scenes:
                match = _re.search(r'\[', response)
                if match:
                    arr_text = response[match.start():]
                    # Try adding missing closing bracket
                    if not arr_text.rstrip().endswith(']'):
                        # Find the last complete object (ends with })
                        last_brace = arr_text.rfind('}')
                        if last_brace > 0:
                            arr_text = arr_text[:last_brace + 1] + ']'
                    try:
                        scenes = _json.loads(arr_text)
                    except _json.JSONDecodeError:
                        pass

            # Last resort: extract individual JSON objects
            if not scenes:
                obj_matches = _re.findall(r'\{[^{}]*\}', response)
                for obj_str in obj_matches:
                    try:
                        obj = _json.loads(obj_str)
                        if "title" in obj or "visual" in obj:
                            scenes.append(obj)
                    except _json.JSONDecodeError:
                        pass

        if not isinstance(scenes, list) or not scenes:
            # Fallback
            scenes = [
                {"title": f"Scene {i+1}", "visual": f"{req.concept}, scene {i+1}, {req.genre.lower()}, {req.mood.lower()}", "narration": "", "duration": 5}
                for i in range(req.num_scenes)
            ]

        # Normalize scene data - LLM sometimes returns arrays for fields
        clean_scenes = []
        for s in scenes[:req.num_scenes]:
            if not isinstance(s, dict):
                continue
            visual = s.get("visual", "")
            if isinstance(visual, list):
                visual = ", ".join(str(v) for v in visual)
            narration = s.get("narration", "")
            if isinstance(narration, list):
                narration = " ".join(str(n) for n in narration)
            duration = s.get("duration", 5)
            if not isinstance(duration, (int, float)):
                try:
                    duration = float(duration)
                except (ValueError, TypeError):
                    duration = 5
            duration = max(2, min(12, int(duration)))
            clean_scenes.append({
                "title": str(s.get("title", f"Scene {len(clean_scenes)+1}"))[:50],
                "visual": str(visual),
                "narration": str(narration),
                "duration": duration,
            })

        return {"scenes": clean_scenes, "concept": req.concept, "genre": req.genre, "mood": req.mood}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/generate/story", response_model=JobResponse)
async def generate_story(req: StoryRequest, background_tasks: BackgroundTasks):
    """Start a multi-scene story generation job."""
    if not req.concept.strip():
        raise HTTPException(status_code=422, detail="concept must not be empty")

    job_id = _create_job("story", req.model_dump())
    thread = threading.Thread(target=_run_story, args=(job_id, req), daemon=True)
    thread.start()
    return _job_response(job_id)


@app.get("/api/v1/jobs/{job_id}", response_model=JobResponse)
async def get_job(job_id: str):
    """Get job status and progress."""
    return _job_response(job_id)


@app.get("/api/v1/jobs/{job_id}/download")
async def download_result(job_id: str):
    """Download the generated video or audio file."""
    with _jobs_lock:
        job = jobs.get(job_id)

    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    if job["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Job is not completed (current status: {job['status']})",
        )
    if not job.get("result"):
        raise HTTPException(status_code=404, detail="No result available for this job")

    # Determine file path from result
    result = job["result"]
    file_path = result.get("video_path") or result.get("audio_path") or result.get("image_path")
    if not file_path or not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="Result file not found on disk")

    lower_path = file_path.lower()
    if lower_path.endswith(".mp4"):
        media_type = "video/mp4"
    elif lower_path.endswith(".gif"):
        media_type = "image/gif"
    elif lower_path.endswith(".png"):
        media_type = "image/png"
    elif lower_path.endswith(".mp3"):
        media_type = "audio/mpeg"
    else:
        media_type = "audio/wav"
    return FileResponse(
        path=file_path,
        media_type=media_type,
        filename=os.path.basename(file_path),
    )


@app.post("/api/v1/audio/tts", response_model=JobResponse)
async def text_to_speech(req: TTSRequest, background_tasks: BackgroundTasks):
    """Generate text-to-speech audio."""
    if not req.text.strip():
        raise HTTPException(status_code=422, detail="text must not be empty")

    job_id = _create_job("tts", req.model_dump())
    thread = threading.Thread(target=_run_tts, args=(job_id, req), daemon=True)
    thread.start()
    return _job_response(job_id)


@app.post("/api/v1/audio/sfx", response_model=JobResponse)
async def generate_sfx(req: SFXRequest, background_tasks: BackgroundTasks):
    """Generate a procedural sound effect (rain, thunder, etc.) as a .wav file."""
    job_id = _create_job("sfx", req.model_dump())
    thread = threading.Thread(target=_run_sfx, args=(job_id, req), daemon=True)
    thread.start()
    return _job_response(job_id)


@app.get("/api/v1/audio/voices")
async def audio_voices():
    """Catalog of TTS voices and personas for the UI's TTS panel."""
    from audio_studio import list_personas, list_piper_voices
    return {"personas": list_personas(), "voices": list_piper_voices()}


# ---- Voice isolation + normalize: file upload (multipart) ----------------

UPLOADS_DIR = os.path.join(OUTPUT_DIR, "uploads")
os.makedirs(UPLOADS_DIR, exist_ok=True)
app.mount("/static/uploads", StaticFiles(directory=UPLOADS_DIR), name="uploads")


def _save_upload(file: UploadFile, prefix: str) -> str:
    """Persist an UploadFile to disk and return its path."""
    safe_name = os.path.basename(file.filename or "input.wav").replace(" ", "_")
    dest = os.path.join(UPLOADS_DIR, f"{prefix}_{uuid.uuid4().hex[:8]}_{safe_name}")
    with open(dest, "wb") as f:
        while True:
            chunk = file.file.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    file.file.close()
    return dest


@app.post("/api/v1/edit/upload")
async def edit_upload(file: UploadFile = File(...)):
    """Upload a video for edit utility endpoints."""
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
        raise HTTPException(status_code=400, detail="Upload a video file: mp4, mov, avi, mkv, or webm")
    saved = _save_upload(file, prefix="edit")
    return {
        "video_path": saved,
        "filename": os.path.basename(saved),
        "size_bytes": os.path.getsize(saved),
    }


@app.post("/api/v1/edit/upload-audio")
async def edit_upload_audio(file: UploadFile = File(...)):
    """Upload an audio file for edit utility endpoints."""
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg"}:
        raise HTTPException(status_code=400, detail="Upload an audio file: mp3, wav, m4a, aac, flac, or ogg")
    saved = _save_upload(file, prefix="edit_audio")
    return {
        "audio_path": saved,
        "filename": os.path.basename(saved),
        "size_bytes": os.path.getsize(saved),
    }


def _run_isolate(job_id: str, input_path: str):
    try:
        _update_job(job_id, status="running", message="Separating vocals from background...")
        from audio_studio import isolate_voice

        result = isolate_voice(input_path)
        if "error" in result:
            raise RuntimeError(result["error"])

        # The output_path key in download_result expects audio_path; expose
        # the vocals there for backward-compat, plus full URLs for both stems.
        vocals = result.get("vocals")
        background = result.get("background")
        method = result.get("method", "ffmpeg_filter")

        # Move outputs under uploads dir so they can be served via static mount.
        def _to_static(path: str, label: str) -> str:
            if not path or not os.path.isfile(path):
                return ""
            name = f"{label}_{job_id}.wav"
            target = os.path.join(UPLOADS_DIR, name)
            try:
                import shutil
                shutil.move(path, target)
            except Exception:
                target = path
            return target

        vocals_path = _to_static(vocals, "vocals") if vocals else ""
        bg_path = _to_static(background, "background") if background else ""

        _update_job(
            job_id,
            status="completed",
            progress=100,
            message=f"Voice isolation complete ({method}).",
            result={
                "audio_path": vocals_path or bg_path,
                "vocals_path": vocals_path,
                "background_path": bg_path,
                "vocals_url": f"/static/uploads/{os.path.basename(vocals_path)}" if vocals_path else "",
                "background_url": f"/static/uploads/{os.path.basename(bg_path)}" if bg_path else "",
                "method": method,
            },
        )
    except Exception as exc:
        _update_job(job_id, status="failed", error=str(exc), message=f"Isolate failed: {exc}")
    finally:
        # Clean up upload after processing
        try:
            if os.path.isfile(input_path):
                os.remove(input_path)
        except OSError:
            pass


def _run_normalize(job_id: str, input_path: str, target_db: float):
    try:
        _update_job(job_id, status="running", message=f"Normalizing to {target_db:.1f} dB...")
        from audio_studio import normalize_audio

        out_name = f"normalized_{job_id}.wav"
        target_path = os.path.join(UPLOADS_DIR, out_name)
        normalize_audio(input_path, output_path=target_path, target_db=target_db)

        _update_job(
            job_id,
            status="completed",
            progress=100,
            message="Normalize complete.",
            result={
                "audio_path": target_path,
                "audio_url": f"/static/uploads/{out_name}",
                "target_db": target_db,
            },
        )
    except Exception as exc:
        _update_job(job_id, status="failed", error=str(exc), message=f"Normalize failed: {exc}")
    finally:
        try:
            if os.path.isfile(input_path):
                os.remove(input_path)
        except OSError:
            pass


@app.post("/api/v1/audio/isolate", response_model=JobResponse)
async def audio_isolate(file: UploadFile = File(...)):
    """Separate vocals from background. Accepts any audio/video file."""
    saved = _save_upload(file, prefix="iso")
    job_id = _create_job("isolate", {"filename": file.filename})
    thread = threading.Thread(target=_run_isolate, args=(job_id, saved), daemon=True)
    thread.start()
    return _job_response(job_id)


@app.post("/api/v1/audio/normalize", response_model=JobResponse)
async def audio_normalize(file: UploadFile = File(...), target_db: float = -3.0):
    """Normalize an audio file to a target dB level. Accepts any audio/video file."""
    target_db = max(-30.0, min(0.0, float(target_db)))
    saved = _save_upload(file, prefix="norm")
    job_id = _create_job("normalize", {"filename": file.filename, "target_db": target_db})
    thread = threading.Thread(target=_run_normalize, args=(job_id, saved, target_db), daemon=True)
    thread.start()
    return _job_response(job_id)


@app.post("/api/v1/edit/trim", response_model=JobResponse)
async def trim_video(req: TrimRequest, background_tasks: BackgroundTasks):
    """Trim a video to a specific time range."""
    if not os.path.isfile(req.video_path):
        raise HTTPException(status_code=422, detail=f"Video file not found: {req.video_path}")

    job_id = _create_job("trim", req.model_dump())
    thread = threading.Thread(target=_run_trim, args=(job_id, req), daemon=True)
    thread.start()
    return _job_response(job_id)


@app.post("/api/v1/edit/merge", response_model=JobResponse)
async def merge_videos(req: MergeRequest, background_tasks: BackgroundTasks):
    """Merge/concatenate multiple videos."""
    if len(req.video_paths) < 2:
        raise HTTPException(status_code=422, detail="At least 2 video paths are required")
    for p in req.video_paths:
        if not os.path.isfile(p):
            raise HTTPException(status_code=422, detail=f"Video file not found: {p}")

    job_id = _create_job("merge", req.model_dump())
    thread = threading.Thread(target=_run_merge, args=(job_id, req), daemon=True)
    thread.start()
    return _job_response(job_id)


@app.post("/api/v1/edit/upscale", response_model=JobResponse)
async def upscale_video(req: UpscaleRequest, background_tasks: BackgroundTasks):
    """Upscale video resolution."""
    if not os.path.isfile(req.video_path):
        raise HTTPException(status_code=422, detail=f"Video file not found: {req.video_path}")

    job_id = _create_job("upscale", req.model_dump())
    thread = threading.Thread(target=_run_upscale, args=(job_id, req), daemon=True)
    thread.start()
    return _job_response(job_id)


@app.post("/api/v1/edit/extract-audio", response_model=JobResponse)
async def edit_extract_audio(req: ExtractAudioRequest):
    """Extract a video's audio track as WAV or MP3."""
    if not os.path.isfile(req.video_path):
        raise HTTPException(status_code=422, detail=f"Video file not found: {req.video_path}")
    job_id = _create_job("extract_audio", req.model_dump())
    thread = threading.Thread(target=_run_extract_audio, args=(job_id, req), daemon=True)
    thread.start()
    return _job_response(job_id)


@app.post("/api/v1/edit/remove-audio", response_model=JobResponse)
async def edit_remove_audio(req: VideoPathRequest):
    """Remove the audio track from a video."""
    if not os.path.isfile(req.video_path):
        raise HTTPException(status_code=422, detail=f"Video file not found: {req.video_path}")
    job_id = _create_job("remove_audio", req.model_dump())
    thread = threading.Thread(target=_run_remove_audio, args=(job_id, req), daemon=True)
    thread.start()
    return _job_response(job_id)


@app.post("/api/v1/edit/compress", response_model=JobResponse)
async def edit_compress(req: CompressRequest):
    """Compress video using a quality preset."""
    if not os.path.isfile(req.video_path):
        raise HTTPException(status_code=422, detail=f"Video file not found: {req.video_path}")
    job_id = _create_job("compress", req.model_dump())
    thread = threading.Thread(target=_run_compress, args=(job_id, req), daemon=True)
    thread.start()
    return _job_response(job_id)


@app.post("/api/v1/edit/rotate-flip", response_model=JobResponse)
async def edit_rotate_flip(req: RotateFlipRequest):
    """Rotate and/or flip video orientation."""
    if not os.path.isfile(req.video_path):
        raise HTTPException(status_code=422, detail=f"Video file not found: {req.video_path}")
    job_id = _create_job("rotate_flip", req.model_dump())
    thread = threading.Thread(target=_run_rotate_flip, args=(job_id, req), daemon=True)
    thread.start()
    return _job_response(job_id)


@app.post("/api/v1/edit/reverse", response_model=JobResponse)
async def edit_reverse(req: ReverseRequest):
    """Reverse video playback, optionally including audio."""
    if not os.path.isfile(req.video_path):
        raise HTTPException(status_code=422, detail=f"Video file not found: {req.video_path}")
    job_id = _create_job("reverse", req.model_dump())
    thread = threading.Thread(target=_run_reverse, args=(job_id, req), daemon=True)
    thread.start()
    return _job_response(job_id)


@app.post("/api/v1/edit/loop", response_model=JobResponse)
async def edit_loop(req: LoopRequest):
    """Loop a video multiple times."""
    if not os.path.isfile(req.video_path):
        raise HTTPException(status_code=422, detail=f"Video file not found: {req.video_path}")
    job_id = _create_job("loop", req.model_dump())
    thread = threading.Thread(target=_run_loop, args=(job_id, req), daemon=True)
    thread.start()
    return _job_response(job_id)


@app.post("/api/v1/edit/stabilize", response_model=JobResponse)
async def edit_stabilize(req: StabilizeRequest):
    """Stabilize shaky footage using ffmpeg vidstab."""
    if not os.path.isfile(req.video_path):
        raise HTTPException(status_code=422, detail=f"Video file not found: {req.video_path}")
    job_id = _create_job("stabilize", req.model_dump())
    thread = threading.Thread(target=_run_stabilize, args=(job_id, req), daemon=True)
    thread.start()
    return _job_response(job_id)


@app.post("/api/v1/edit/color-correction", response_model=JobResponse)
async def edit_color_correction(req: ColorCorrectionRequest):
    """Adjust brightness, contrast, saturation, and hue."""
    if not os.path.isfile(req.video_path):
        raise HTTPException(status_code=422, detail=f"Video file not found: {req.video_path}")
    job_id = _create_job("color_correction", req.model_dump())
    thread = threading.Thread(target=_run_color_correction, args=(job_id, req), daemon=True)
    thread.start()
    return _job_response(job_id)


@app.post("/api/v1/edit/region-effect", response_model=JobResponse)
async def edit_region_effect(req: RegionEffectRequest):
    """Blur or pixelate a rectangular video region."""
    if not os.path.isfile(req.video_path):
        raise HTTPException(status_code=422, detail=f"Video file not found: {req.video_path}")
    job_id = _create_job("region_effect", req.model_dump())
    thread = threading.Thread(target=_run_region_effect, args=(job_id, req), daemon=True)
    thread.start()
    return _job_response(job_id)


@app.post("/api/v1/edit/picture-in-picture", response_model=JobResponse)
async def edit_picture_in_picture(req: PictureInPictureRequest):
    """Overlay one video on another."""
    if not os.path.isfile(req.video_path):
        raise HTTPException(status_code=422, detail=f"Video file not found: {req.video_path}")
    if not os.path.isfile(req.overlay_path):
        raise HTTPException(status_code=422, detail=f"Overlay video file not found: {req.overlay_path}")
    job_id = _create_job("picture_in_picture", req.model_dump())
    thread = threading.Thread(target=_run_picture_in_picture, args=(job_id, req), daemon=True)
    thread.start()
    return _job_response(job_id)


@app.post("/api/v1/edit/background-music", response_model=JobResponse)
async def edit_background_music(req: BackgroundMusicRequest):
    """Mix or add background music to a video."""
    if not os.path.isfile(req.video_path):
        raise HTTPException(status_code=422, detail=f"Video file not found: {req.video_path}")
    if not os.path.isfile(req.audio_path):
        raise HTTPException(status_code=422, detail=f"Audio file not found: {req.audio_path}")
    job_id = _create_job("background_music", req.model_dump())
    thread = threading.Thread(target=_run_background_music, args=(job_id, req), daemon=True)
    thread.start()
    return _job_response(job_id)


@app.post("/api/v1/edit/speed", response_model=JobResponse)
async def edit_speed(req: SpeedRequest):
    """Change video playback speed."""
    if not os.path.isfile(req.video_path):
        raise HTTPException(status_code=422, detail=f"Video file not found: {req.video_path}")
    job_id = _create_job("speed", req.model_dump())
    thread = threading.Thread(target=_run_speed, args=(job_id, req), daemon=True)
    thread.start()
    return _job_response(job_id)


@app.post("/api/v1/edit/gif", response_model=JobResponse)
async def edit_gif(req: GifRequest):
    """Convert a video segment to an animated GIF."""
    if not os.path.isfile(req.video_path):
        raise HTTPException(status_code=422, detail=f"Video file not found: {req.video_path}")
    job_id = _create_job("gif", req.model_dump())
    thread = threading.Thread(target=_run_gif, args=(job_id, req), daemon=True)
    thread.start()
    return _job_response(job_id)


@app.post("/api/v1/edit/thumbnail", response_model=JobResponse)
async def edit_thumbnail(req: ThumbnailRequest):
    """Extract a PNG thumbnail from a video timestamp."""
    if not os.path.isfile(req.video_path):
        raise HTTPException(status_code=422, detail=f"Video file not found: {req.video_path}")
    job_id = _create_job("thumbnail", req.model_dump())
    thread = threading.Thread(target=_run_thumbnail, args=(job_id, req), daemon=True)
    thread.start()
    return _job_response(job_id)


# ---------------------------------------------------------------------------
# Image Studio endpoints
# ---------------------------------------------------------------------------

@app.get("/api/v1/images/catalog")
async def images_catalog():
    """Return everything the Images UI needs: providers, models, sizes, styles."""
    import imagegen
    return imagegen.catalog()


@app.post("/api/v1/images/enhance-prompt")
async def images_enhance_prompt(req: PromptEnhanceRequest):
    """Run a raw prompt through the LLM to enrich (expand) or compress (shorten).
    When ``image_path`` is set, BLIP captions the image first so the LLM is
    grounded in actual scene content (used by Edit / Variation / Inpaint).
    Always returns both an enhanced positive prompt AND a negative prompt.
    Surfaces LLM errors to the caller instead of silently echoing the input."""
    import imagegen
    try:
        result = imagegen.enhance_prompt(
            req.prompt,
            style=req.style,
            mode=req.mode,
            image_path=req.image_path,
            negative_prompt=req.negative_prompt,
        )
    except Exception as e:
        logger.exception("enhance-prompt failed")
        raise HTTPException(status_code=502, detail=f"LLM call failed: {e}")
    return {
        "original": req.prompt,
        "enhanced": result.get("enhanced", ""),
        "negative": result.get("negative", ""),
        "image_caption": result.get("image_caption", ""),
        "mode": req.mode,
    }


@app.post("/api/v1/images/generate", response_model=JobResponse)
async def images_generate(req: ImageGenerateRequest):
    """Text-to-image. Returns a job; poll /api/v1/jobs/{id}."""
    job_id = _create_job("image_generate", req.model_dump())
    threading.Thread(target=_run_image_generate, args=(job_id, req), daemon=True).start()
    return _job_response(job_id)


@app.post("/api/v1/images/edit", response_model=JobResponse)
async def images_edit(req: ImageEditRequest):
    """Image-to-image: modify an existing image with a prompt."""
    if not os.path.isfile(req.image_path):
        raise HTTPException(status_code=422, detail=f"Image not found: {req.image_path}")
    job_id = _create_job("image_edit", req.model_dump())
    threading.Thread(target=_run_image_edit, args=(job_id, req), daemon=True).start()
    return _job_response(job_id)


@app.post("/api/v1/images/inpaint", response_model=JobResponse)
async def images_inpaint(req: ImageInpaintRequest):
    """Mask-based local edit. Mask: white = edit, black = keep."""
    if not os.path.isfile(req.image_path):
        raise HTTPException(status_code=422, detail=f"Image not found: {req.image_path}")
    if not os.path.isfile(req.mask_path):
        raise HTTPException(status_code=422, detail=f"Mask not found: {req.mask_path}")
    job_id = _create_job("image_inpaint", req.model_dump())
    threading.Thread(target=_run_image_inpaint, args=(job_id, req), daemon=True).start()
    return _job_response(job_id)


@app.post("/api/v1/images/variation", response_model=JobResponse)
async def images_variation(req: ImageVariationRequest):
    """\"More like this\" — low-strength img2img on the same prompt."""
    if not os.path.isfile(req.image_path):
        raise HTTPException(status_code=422, detail=f"Image not found: {req.image_path}")
    job_id = _create_job("image_variation", req.model_dump())
    threading.Thread(target=_run_image_variation, args=(job_id, req), daemon=True).start()
    return _job_response(job_id)


@app.post("/api/v1/images/upscale", response_model=JobResponse)
async def images_upscale(req: ImageUpscaleRequest):
    """Upscale 2x or 4x via Real-ESRGAN (if installed) or Lanczos."""
    if not os.path.isfile(req.image_path):
        raise HTTPException(status_code=422, detail=f"Image not found: {req.image_path}")
    job_id = _create_job("image_upscale", req.model_dump())
    threading.Thread(target=_run_image_upscale, args=(job_id, req), daemon=True).start()
    return _job_response(job_id)


@app.post("/api/v1/images/remove-bg", response_model=JobResponse)
async def images_remove_bg(req: ImageBgRemoveRequest):
    """Cut out subject; transparent PNG out. Requires rembg."""
    if not os.path.isfile(req.image_path):
        raise HTTPException(status_code=422, detail=f"Image not found: {req.image_path}")
    job_id = _create_job("image_remove_bg", req.model_dump())
    threading.Thread(target=_run_image_remove_bg, args=(job_id, req), daemon=True).start()
    return _job_response(job_id)


@app.post("/api/v1/images/upload")
async def images_upload(file: UploadFile = File(...)):
    """Upload an image so it can be referenced by image_path in subsequent calls."""
    ext = os.path.splitext(file.filename or "")[1].lower() or ".png"
    if ext not in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
        raise HTTPException(status_code=422, detail=f"Unsupported file type: {ext}")
    out_path = os.path.join(IMAGES_DIR, f"upload_{uuid.uuid4().hex[:8]}{ext}")
    contents = await file.read()
    with open(out_path, "wb") as f:
        f.write(contents)
    return {
        "image_path": out_path,
        "url": f"/static/images/{os.path.basename(out_path)}",
        "size_bytes": len(contents),
    }


@app.get("/api/v1/images/history")
async def images_history(limit: int = 50):
    """Recent image generations on disk, newest first."""
    import imagegen
    items = imagegen.list_history(limit=limit)
    return {
        "images": [
            {**i, "url": f"/static/images/{i['filename']}"} for i in items
        ],
    }


@app.get("/api/v1/jobs")
async def list_jobs(limit: int = 20):
    """List recent jobs, newest first."""
    with _jobs_lock:
        sorted_jobs = sorted(jobs.items(), key=lambda x: x[1]["created_at"], reverse=True)
    result = []
    for job_id, job in sorted_jobs[:limit]:
        result.append(
            {
                "job_id": job_id,
                "kind": job["kind"],
                "status": job["status"],
                "progress": job["progress"],
                "message": job.get("message", ""),
                "created_at": job["created_at"],
                "elapsed": round(time.time() - job["created_at"], 2),
            }
        )
    return {"jobs": result, "total": len(jobs)}


# ---------------------------------------------------------------------------
# IP-Adapter endpoints
# ---------------------------------------------------------------------------

@app.get("/api/v1/ip-adapter/status")
async def ip_adapter_status():
    """Get IP-Adapter status and dependency check."""
    try:
        from ip_adapter import get_ip_adapter_manager
        manager = get_ip_adapter_manager()
        return manager.get_status()
    except ImportError:
        return {"enabled": False, "error": "ip_adapter module not found"}
    except Exception as e:
        return {"enabled": False, "error": str(e)}


@app.post("/api/v1/ip-adapter/upload-reference")
async def upload_reference_image(
    file: UploadFile = File(...),
    char_id: str = "main_subject",
):
    """Upload a character reference image for IP-Adapter."""
    ref_dir = os.path.join(OUTPUT_DIR, "ip_adapter_refs")
    os.makedirs(ref_dir, exist_ok=True)

    ext = os.path.splitext(file.filename or "ref.png")[1] or ".png"
    ref_path = os.path.join(ref_dir, f"{char_id}_{int(time.time())}{ext}")

    content = await file.read()
    with open(ref_path, "wb") as f:
        f.write(content)

    # Register with IP-Adapter manager
    try:
        from ip_adapter import get_ip_adapter_manager
        manager = get_ip_adapter_manager()
        manager.register_character(char_id, reference_images=[ref_path])
        manager.enable()
        return {
            "status": "ok",
            "char_id": char_id,
            "ref_path": ref_path,
            "message": f"Reference image registered for character '{char_id}'",
        }
    except Exception as e:
        return {"status": "ok", "ref_path": ref_path, "warning": f"Saved but registration failed: {e}"}


@app.post("/api/v1/ip-adapter/configure")
async def configure_ip_adapter(enabled: bool = True, strength: float = 0.6):
    """Enable/disable IP-Adapter and set strength."""
    try:
        from ip_adapter import get_ip_adapter_manager
        manager = get_ip_adapter_manager()
        if enabled:
            manager.enable(strength)
        else:
            manager.disable()
        return {"enabled": manager.enabled, "strength": manager.strength}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/ip-adapter/dependencies")
async def ip_adapter_dependencies():
    """Check which IP-Adapter dependencies are installed."""
    deps = {"insightface": False, "onnxruntime": False, "clip_vision": False, "diffusers": False, "cuda": False}
    try:
        import insightface
        deps["insightface"] = True
    except ImportError:
        pass
    try:
        import onnxruntime
        deps["onnxruntime"] = True
    except ImportError:
        pass
    try:
        from transformers import CLIPVisionModelWithProjection
        deps["clip_vision"] = True
    except ImportError:
        pass
    try:
        from diffusers import WanPipeline
        deps["diffusers"] = True
    except ImportError:
        pass
    import torch
    deps["cuda"] = torch.cuda.is_available()

    all_ok = all(deps.values())
    install_cmd = ""
    if not all_ok:
        missing = [k for k, v in deps.items() if not v and k not in ("cuda",)]
        pkg_map = {"insightface": "insightface", "onnxruntime": "onnxruntime-gpu", "clip_vision": "transformers", "diffusers": "diffusers"}
        install_cmd = "pip install " + " ".join(pkg_map.get(m, m) for m in missing)

    return {"dependencies": deps, "all_installed": all_ok, "install_command": install_cmd}


# ---------------------------------------------------------------------------
# Character Portrait Generation
# ---------------------------------------------------------------------------

class CharacterPortraitRequest(BaseModel):
    """Generate actual character reference images from description."""
    name: str
    description: str  # Brief traits like "30s woman, red hair, trench coat"
    visual_prompt: str = ""  # Detailed visual prompt (optional, auto-generated if empty)
    views: list[str] = ["front"]  # Which views to generate: front, three_quarter, side, back
    style: str = "photorealistic"  # photorealistic, anime, 3d_render
    register_ip_adapter: bool = True  # Auto-register as IP-Adapter reference


VIEW_PROMPT_TEMPLATES = {
    "front": "front view portrait, looking directly at camera, centered composition, studio lighting, neutral background",
    "three_quarter": "three-quarter view portrait, slightly turned to the side, elegant pose, studio lighting, neutral background",
    "side": "side profile view, facing left, clean silhouette visible, studio lighting, neutral background",
    "back": "back view, showing hair and outfit from behind, turned away from camera, studio lighting, neutral background",
}


def _build_character_prompt(name: str, description: str, visual_prompt: str, view: str, style: str) -> str:
    """Build a detailed image generation prompt for a character view."""
    base = visual_prompt if visual_prompt.strip() else description

    view_suffix = VIEW_PROMPT_TEMPLATES.get(view, VIEW_PROMPT_TEMPLATES["front"])

    style_map = {
        "photorealistic": "photorealistic, highly detailed skin texture, real person, professional photography, 8k, sharp focus",
        "anime": "anime style, cel shaded, clean lines, vibrant colors, detailed anime character design, high quality illustration",
        "3d_render": "3D rendered character, Pixar style, high quality 3D model, subsurface scattering, ambient occlusion, detailed textures",
    }
    style_suffix = style_map.get(style, style_map["photorealistic"])

    prompt = f"Character portrait of {name}: {base}, {view_suffix}, {style_suffix}, character reference sheet quality, consistent design"
    return prompt


def _run_character_portrait(job_id: str, req: CharacterPortraitRequest):
    """Generate character portrait images in background thread.

    Uses a shared seed across all views for consistency.
    Generates front view first, then uses it as IP-Adapter reference for other views.
    """
    try:
        _update_job(job_id, status="running", progress=5, message="Loading image generation model...")

        from reelforge import rf_generate_image, rf_load_sdxl_pipeline
        from PIL import Image as PILImage
        import shutil
        import random

        # Shared seed ensures same "identity" interpretation across views
        shared_seed = random.randint(0, 2**31)

        views = req.views or ["front"]
        total_views = len(views)
        generated_paths = {}

        # Always generate front view first (it becomes the reference for other views)
        if "front" in views and views[0] != "front":
            views.remove("front")
            views.insert(0, "front")

        front_image = None  # PIL Image of front view, used as IP-Adapter ref

        # Multi-view consistency: rely on shared seed (IP-Adapter download
        # for SDXL can hang for 10+ minutes, so skip it entirely)
        ip_adapter_loaded = False

        for i, view in enumerate(views):
            progress = 10 + (80 * i // total_views)
            _update_job(job_id, progress=progress, message=f"Generating {view} view ({i+1}/{total_views})...")

            prompt = _build_character_prompt(req.name, req.description, req.visual_prompt, view, req.style)

            # For non-front views, use front image as IP-Adapter reference
            ip_ref = None
            if i > 0 and front_image is not None and ip_adapter_loaded:
                ip_ref = front_image

            image_path = rf_generate_image(
                prompt=prompt,
                provider="sdxl_turbo",
                style=req.style,
                aspect_ratio="1:1",
                seed=shared_seed,
                ip_adapter_image=ip_ref,
            )

            # Save front view as reference for subsequent views
            if view == "front" and front_image is None:
                try:
                    front_image = PILImage.open(image_path).convert("RGB").resize((224, 224), PILImage.LANCZOS)
                except Exception:
                    pass

            # Move to character portraits directory
            safe_name = req.name.lower().replace(" ", "_")[:30]
            char_filename = f"{safe_name}_{view}_{uuid.uuid4().hex[:8]}.png"
            dest_path = os.path.join(CHAR_PORTRAITS_DIR, char_filename)
            shutil.move(image_path, dest_path)

            generated_paths[view] = {
                "path": dest_path,
                "filename": char_filename,
                "url": f"/static/portraits/{char_filename}",
            }

        # (IP-Adapter for SDXL multi-view skipped — download too slow;
        #  shared seed provides adequate cross-view consistency)

        # Register with IP-Adapter if requested
        ip_adapter_status = None
        if req.register_ip_adapter:
            try:
                from ip_adapter import get_ip_adapter_manager
                manager = get_ip_adapter_manager()
                ref_paths = [v["path"] for v in generated_paths.values()]
                safe_id = req.name.lower().replace(" ", "_")[:30]
                manager.register_character(safe_id, reference_images=ref_paths)
                manager.enable()
                ip_adapter_status = f"Registered {len(ref_paths)} reference(s) for IP-Adapter"
            except Exception as e:
                ip_adapter_status = f"Generated but IP-Adapter registration failed: {e}"

        _update_job(
            job_id,
            status="completed",
            progress=100,
            message=f"Generated {total_views} character view(s) for {req.name}",
            result={
                "character_name": req.name,
                "portraits": generated_paths,
                "ip_adapter": ip_adapter_status,
            },
        )

    except Exception as e:
        _update_job(job_id, status="failed", error=str(e), message=f"Portrait generation failed: {e}")
        traceback.print_exc()


@app.post("/api/v1/characters/generate-portrait", response_model=JobResponse)
async def generate_character_portrait(req: CharacterPortraitRequest, background_tasks: BackgroundTasks):
    """Generate character reference portrait images using SDXL.

    Creates actual character images (front, side, 3/4 views) from text description.
    These are automatically registered as IP-Adapter references for consistency
    across movie scenes.
    """
    if not req.name.strip():
        raise HTTPException(status_code=422, detail="Character name is required")
    if not req.description.strip() and not req.visual_prompt.strip():
        raise HTTPException(status_code=422, detail="Either description or visual_prompt is required")

    job_id = _create_job("character_portrait", req.model_dump())
    thread = threading.Thread(target=_run_character_portrait, args=(job_id, req), daemon=True)
    thread.start()
    return _job_response(job_id)


@app.get("/api/v1/characters/portraits/{char_name}")
async def list_character_portraits(char_name: str):
    """List all generated portrait images for a character (use '_all' for all characters)."""
    portraits = []
    if os.path.isdir(CHAR_PORTRAITS_DIR):
        for f in sorted(os.listdir(CHAR_PORTRAITS_DIR)):
            if not f.endswith(".png"):
                continue
            # Filter by character name unless requesting all
            if char_name != "_all":
                safe_name = char_name.lower().replace(" ", "_")[:30]
                if not f.startswith(safe_name):
                    continue
            view = "unknown"
            for v in ["front", "three_quarter", "side", "back"]:
                if f"_{v}_" in f:
                    view = v
                    break
            portraits.append({
                "filename": f,
                "url": f"/static/portraits/{f}",
                "view": view,
            })
    return {"character": char_name, "portraits": portraits}


@app.get("/api/v1/characters/list")
async def list_characters():
    """List all characters that have generated portraits, grouped by name.
    Used by Story Mode to import characters from Character Studio."""
    characters = {}
    if os.path.isdir(CHAR_PORTRAITS_DIR):
        for f in sorted(os.listdir(CHAR_PORTRAITS_DIR)):
            if not f.endswith(".png"):
                continue
            # Extract character name from filename: "captain_wolf_front_abc123.png"
            # Views can have underscores (three_quarter), so match views first then extract name
            view = "front"
            char_key = None
            for v in ["three_quarter", "front", "side", "back"]:  # longest match first
                marker = f"_{v}_"
                if marker in f:
                    view = v
                    char_key = f[:f.index(marker)]
                    break
            if char_key is None:
                # No known view found, use filename minus last 2 parts (hash.png)
                parts = f.rsplit("_", 1)
                char_key = parts[0] if len(parts) >= 2 else f.replace(".png", "")

            if char_key not in characters:
                # Convert key back to display name: "captain_wolf" -> "Captain Wolf"
                display_name = " ".join(w.capitalize() for w in char_key.split("_"))
                characters[char_key] = {
                    "id": char_key,
                    "name": display_name,
                    "portraits": [],
                    "front_url": None,
                }
            characters[char_key]["portraits"].append({
                "filename": f,
                "url": f"/static/portraits/{f}",
                "view": view,
            })
            # Keep track of front view for thumbnail
            if view == "front" and not characters[char_key]["front_url"]:
                characters[char_key]["front_url"] = f"/static/portraits/{f}"

    # Also pull appearance info from IP-Adapter manager job history
    result = []
    for key, char in characters.items():
        # Try to find the original generation job for this character
        appearance = ""
        for job in jobs.values():
            if job.get("kind") == "character_portrait":
                params = job.get("params", {})
                if params.get("name", "").lower().replace(" ", "_")[:30] == key:
                    appearance = params.get("description", "")
                    if params.get("visual_prompt"):
                        appearance = params["visual_prompt"]
                    break
        char["appearance"] = appearance
        result.append(char)

    return {"characters": result}


# ---------------------------------------------------------------------------
# Environment Variables & Logs
# ---------------------------------------------------------------------------

# Env vars that are safe to expose/edit via the UI
MANAGED_ENV_VARS = [
    "HF_TOKEN", "HF_HOME", "NEXT_PUBLIC_API_URL",
    "CUDA_VISIBLE_DEVICES", "PYTORCH_CUDA_ALLOC_CONF",
]


@app.get("/api/v1/system/env")
async def get_env_vars():
    """Get current environment variables relevant to StudioLite."""
    env = {}
    for key in MANAGED_ENV_VARS:
        val = os.environ.get(key, "")
        # Mask tokens (show first 8 chars only)
        if "TOKEN" in key and val and len(val) > 8:
            env[key] = val[:8] + "..." + val[-4:]
        else:
            env[key] = val
    # Also include all HF_ and CUDA_ vars
    for key, val in os.environ.items():
        if key.startswith(("HF_", "CUDA_", "TORCH_", "PYTORCH_")) and key not in env:
            if "TOKEN" in key and val and len(val) > 8:
                env[key] = val[:8] + "..." + val[-4:]
            else:
                env[key] = val
    return {"env": env, "managed_keys": MANAGED_ENV_VARS}


@app.post("/api/v1/system/env")
async def set_env_var(key: str, value: str):
    """Set an environment variable. Takes effect immediately for this process."""
    if not key.strip():
        raise HTTPException(status_code=422, detail="Key cannot be empty")
    os.environ[key.strip()] = value
    # Persist to .env file for next startup
    env_file = os.path.join(ROOT_DIR, ".env")
    env_lines = []
    if os.path.exists(env_file):
        with open(env_file, "r") as f:
            env_lines = f.readlines()
    # Update or add the key
    found = False
    for i, line in enumerate(env_lines):
        if line.strip().startswith(f"{key.strip()}="):
            env_lines[i] = f"{key.strip()}={value}\n"
            found = True
            break
    if not found:
        env_lines.append(f"{key.strip()}={value}\n")
    with open(env_file, "w") as f:
        f.writelines(env_lines)
    return {"status": "ok", "key": key.strip(), "persisted": True}


@app.delete("/api/v1/system/env")
async def delete_env_var(key: str):
    """Remove an environment variable."""
    if key in os.environ:
        del os.environ[key]
    # Remove from .env file
    env_file = os.path.join(ROOT_DIR, ".env")
    if os.path.exists(env_file):
        with open(env_file, "r") as f:
            lines = f.readlines()
        lines = [l for l in lines if not l.strip().startswith(f"{key}=")]
        with open(env_file, "w") as f:
            f.writelines(lines)
    return {"status": "ok", "key": key, "deleted": True}


@app.get("/api/v1/system/logs")
async def get_logs(lines: int = 100):
    """Get recent application logs."""
    log_file = os.path.join(OUTPUT_DIR, "videogen.log")
    log_lines = []
    if os.path.exists(log_file):
        try:
            with open(log_file, "r", encoding="utf-8", errors="replace") as f:
                all_lines = f.readlines()
                log_lines = all_lines[-lines:]
        except Exception:
            pass
    # Also get recent job status as pseudo-logs
    job_logs = []
    for jid, j in sorted(jobs.items(), key=lambda x: x[1].get("created_at", 0), reverse=True)[:20]:
        job_logs.append({
            "job_id": jid[:8],
            "kind": j.get("kind", ""),
            "status": j["status"],
            "progress": j["progress"],
            "message": j.get("message", ""),
            "error": j.get("error"),
            "elapsed": round(time.time() - j["created_at"], 1),
        })
    return {
        "log_lines": [l.rstrip() for l in log_lines],
        "log_file": log_file,
        "recent_jobs": job_logs,
    }


@app.post("/api/v1/system/hf-token-test")
async def test_hf_token(token: str):
    """Test a HuggingFace token and save it if valid."""
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        info = api.whoami()
        # Token works — save it
        os.environ["HF_TOKEN"] = token
        # Persist to .env
        env_file = os.path.join(ROOT_DIR, ".env")
        env_lines = []
        if os.path.exists(env_file):
            with open(env_file, "r") as f:
                env_lines = f.readlines()
        found = False
        for i, line in enumerate(env_lines):
            if line.strip().startswith("HF_TOKEN="):
                env_lines[i] = f"HF_TOKEN={token}\n"
                found = True
                break
        if not found:
            env_lines.append(f"HF_TOKEN={token}\n")
        with open(env_file, "w") as f:
            f.writelines(env_lines)
        return {"status": "ok", "user": info.get("name", "unknown"), "message": f"Token valid. Authenticated as {info.get('name', 'unknown')}."}
    except Exception as e:
        return {"status": "error", "message": f"Token invalid: {e}"}


# ---------------------------------------------------------------------------
# Live transcription (WebSocket)
# ---------------------------------------------------------------------------

@app.websocket("/api/v1/transcribe/live")
async def live_transcribe_ws(websocket: WebSocket):
    """
    Streaming transcription over WebSocket.

    Query params:
        session: client-supplied session id (also used for output filenames)
        model: whisper model size (tiny|base|small|medium|large-v2|large-v3)
        language: ISO 639-1 source language code, or omit for auto-detect
        translate: "1" to translate to English

    Wire protocol:
        Binary frames -> raw PCM int16 little-endian, mono, 16 kHz
        Text frame "stop" -> finalize and close
        Server sends JSON:
            {"type":"ready","session":...}
            {"type":"status","stage":"loading_model"|"model_ready",...}
            {"type":"final"|"partial","start":..,"end":..,"text":..}
            {"type":"complete","transcripts":{"txt":..,"srt":..,"json":..}}
            {"type":"error","message":..}
    """
    import asyncio
    from transcriber import LiveTranscriber

    await websocket.accept()
    qp = websocket.query_params
    session_id = qp.get("session") or str(uuid.uuid4())
    model_size = qp.get("model", "base")
    language = qp.get("language") or None
    translate = qp.get("translate") == "1"

    transcriber: Optional["LiveTranscriber"] = None
    model_ready = asyncio.Event()
    stop_event = asyncio.Event()
    audio_queue: "asyncio.Queue[bytes]" = asyncio.Queue(maxsize=200)

    async def receiver_task():
        """Pull frames off the WS as fast as possible so the OS buffer never fills."""
        try:
            while not stop_event.is_set():
                msg = await websocket.receive()
                if msg.get("type") == "websocket.disconnect":
                    stop_event.set()
                    break
                if msg.get("bytes") is not None:
                    try:
                        audio_queue.put_nowait(msg["bytes"])
                    except asyncio.QueueFull:
                        # Drop oldest if processor is way behind
                        try:
                            audio_queue.get_nowait()
                            audio_queue.put_nowait(msg["bytes"])
                        except Exception:
                            pass
                elif msg.get("text"):
                    cmd = msg["text"].strip().lower()
                    if cmd == "stop":
                        stop_event.set()
                        break
                    elif cmd == "ping":
                        try:
                            await websocket.send_json({"type": "pong"})
                        except Exception:
                            pass
        except WebSocketDisconnect:
            stop_event.set()

    async def processor_task():
        """Drain the audio queue, run decode in a thread so it never blocks the loop."""
        await model_ready.wait()
        while not stop_event.is_set() or not audio_queue.empty():
            try:
                chunk = await asyncio.wait_for(audio_queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            transcriber.feed_pcm16(chunk)
            # Drain any other queued chunks before deciding to decode
            while True:
                try:
                    extra = audio_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                transcriber.feed_pcm16(extra)
            result = await asyncio.to_thread(transcriber.maybe_decode)
            if result:
                for fs in result["final"]:
                    await websocket.send_json({"type": "final", **fs})
                for ps in result["partial"]:
                    await websocket.send_json({"type": "partial", **ps})

    try:
        transcriber = LiveTranscriber(
            session_id=session_id,
            transcripts_dir=TRANSCRIPTS_DIR,
            model_size=model_size,
            language=language,
            translate_to_english=translate,
        )
        await websocket.send_json({"type": "ready", "session": session_id, "device": transcriber.device})
        await websocket.send_json({"type": "status", "stage": "loading_model", "model": model_size})

        # Load the model in a thread so the event loop stays responsive.
        await asyncio.to_thread(transcriber._ensure_model)
        await websocket.send_json({"type": "status", "stage": "model_ready"})
        model_ready.set()

        recv = asyncio.create_task(receiver_task())
        proc = asyncio.create_task(processor_task())
        client_alive = True
        try:
            await asyncio.gather(recv, proc)
        except WebSocketDisconnect:
            client_alive = False
            stop_event.set()

        # Final flush, bounded so a stuck decode can't pin the socket.
        try:
            result = await asyncio.wait_for(asyncio.to_thread(transcriber.flush), timeout=20.0)
        except asyncio.TimeoutError:
            logger.warning("Live transcribe flush timed out; using already-persisted transcript.")
            result = None
        except Exception:
            logger.exception("Live transcribe flush failed")
            result = None

        if client_alive and result:
            for fs in result.get("final", []):
                try:
                    await websocket.send_json({"type": "final", **fs})
                except Exception:
                    client_alive = False
                    break

        urls = transcriber.transcript_urls(base_url="/static/transcripts")
        if client_alive:
            try:
                await websocket.send_json({"type": "complete", "transcripts": urls})
            except Exception:
                pass

    except WebSocketDisconnect:
        if transcriber:
            try:
                await asyncio.wait_for(asyncio.to_thread(transcriber.flush), timeout=20.0)
            except Exception:
                pass
    except Exception as e:
        logger.exception("Live transcribe error")
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
        if transcriber:
            try:
                await asyncio.wait_for(asyncio.to_thread(transcriber.flush), timeout=20.0)
            except Exception:
                pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Live screen OCR (Screen Transcribe)
# ---------------------------------------------------------------------------

@app.get("/api/v1/screen/monitors")
async def screen_monitors():
    """List monitors available to the server for local screen capture."""
    from screen_ocr import enumerate_monitors
    return {"monitors": enumerate_monitors()}


# ---------------------------------------------------------------------------
# Local LLM post-processing (Restructure with LLM)
# ---------------------------------------------------------------------------

@app.get("/api/v1/llm/models")
async def llm_models(host: Optional[str] = None):
    """Probe an Ollama instance and return its installed models +
    the style presets the UI should expose.

    Query params:
        host - optional override (default: OLLAMA_HOST env, then http://localhost:11434)
    """
    from llm_filter import is_ollama_reachable, list_models, list_styles, OLLAMA_HOST, DEFAULT_MODEL, _resolve_host
    resolved = _resolve_host(host)
    reachable = is_ollama_reachable(resolved)
    return {
        "available": reachable,
        "host": resolved,
        "default_host": OLLAMA_HOST,
        "default_model": DEFAULT_MODEL,
        "models": list_models(resolved) if reachable else [],
        "styles": list_styles(),
    }


@app.post("/api/v1/screen/restructure")
async def screen_restructure(req: RestructureRequest):
    """Run the saved screen-transcript text through the chosen Ollama model
    with a style preset (or a custom prompt), then render Markdown / DOCX / PDF."""
    import asyncio
    from llm_filter import restructure
    from doc_writer import markdown_to_docx, markdown_to_pdf

    txt_path = os.path.join(SCREEN_TRANSCRIPTS_DIR, f"{req.session_id}.txt")
    if not os.path.isfile(txt_path):
        raise HTTPException(status_code=404, detail=f"session not found: {req.session_id}")
    with open(txt_path, encoding="utf-8") as f:
        raw = f.read()
    if not raw.strip():
        raise HTTPException(status_code=400, detail="session transcript is empty")

    try:
        result = await asyncio.to_thread(
            restructure,
            raw,
            model=req.model,
            style=req.style,
            custom_prompt=req.custom_prompt,
            temperature=req.temperature,
            host=req.host,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    md = result["markdown"]
    if not md.strip():
        raise HTTPException(status_code=502, detail="Model returned an empty response.")

    base = os.path.join(SCREEN_TRANSCRIPTS_DIR, f"{req.session_id}.restructured")
    md_path = base + ".md"
    docx_path = base + ".docx"
    pdf_path = base + ".pdf"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)

    title = req.title or f"Restructured: {req.session_id}"
    await asyncio.to_thread(markdown_to_docx, md, docx_path, title)
    await asyncio.to_thread(markdown_to_pdf, md, pdf_path, title)

    return {
        "markdown": md,
        "stats": {
            "input_chars": result["input_chars"],
            "output_chars": result["output_chars"],
            "elapsed_seconds": result["elapsed"],
        },
        "model": result["model"],
        "style": result["style"],
        "files": {
            "md": f"/static/screen_transcripts/{req.session_id}.restructured.md",
            "docx": f"/static/screen_transcripts/{req.session_id}.restructured.docx",
            "pdf": f"/static/screen_transcripts/{req.session_id}.restructured.pdf",
        },
    }


@app.websocket("/api/v1/screen/live")
async def live_screen_ws(websocket: WebSocket):
    """
    Streaming screen OCR over WebSocket.

    Query params:
        session     - client session id (also used for output filenames)
        source      - "browser" (default) or "local"
        confidence  - float, OCR confidence floor (default 0.5)
        diff        - int, perceptual-hash distance below which a frame is
                      treated as unchanged (default 4)
        monitor     - int, mss monitor index when source=local (default 1)
        fps         - float, capture rate when source=local (default 1.0)

    Wire protocol:
        Client (browser source):
            binary frames -> JPEG/PNG bytes
            text "stop"   -> finalize and close
        Client (local source):
            no binary frames needed - server samples on its own
            text "stop"   -> finalize and close

        Server -> client (JSON):
            {"type":"ready","session":...,"source":...}
            {"type":"status","stage":"loading_model"|"model_ready",...}
            {"type":"line","t":..,"text":..,"conf":..,"bbox":..}
            {"type":"frame","frame_skipped":bool,"reason":..,"phash_diff":..}
            {"type":"complete","transcripts":{...},"stats":{...}}
            {"type":"error","message":...}
    """
    import asyncio
    from screen_ocr import LiveScreenOCR, capture_local_jpeg

    await websocket.accept()
    qp = websocket.query_params
    session_id = qp.get("session") or str(uuid.uuid4())
    source = (qp.get("source") or "browser").lower()
    confidence = float(qp.get("confidence") or 0.6)
    diff = int(qp.get("diff") or 4)
    max_repeats = int(qp.get("max_repeats") or 2)
    drop_garbage = (qp.get("drop_garbage") or "1") != "0"
    monitor = int(qp.get("monitor") or 1)
    fps = float(qp.get("fps") or 1.0)
    fps = max(0.2, min(fps, 5.0))

    ocr: Optional["LiveScreenOCR"] = None
    stop_event = asyncio.Event()
    frame_queue: "asyncio.Queue[bytes]" = asyncio.Queue(maxsize=8)

    async def receiver_task():
        """Pull frames from the WS (browser source) or just watch for the
        stop signal (local source)."""
        try:
            while not stop_event.is_set():
                msg = await websocket.receive()
                if msg.get("type") == "websocket.disconnect":
                    stop_event.set()
                    break
                if source == "browser" and msg.get("bytes") is not None:
                    # Drop oldest if the processor can't keep up - we want the
                    # most recent screen state, not a backlog of stale frames.
                    try:
                        frame_queue.put_nowait(msg["bytes"])
                    except asyncio.QueueFull:
                        try:
                            frame_queue.get_nowait()
                            frame_queue.put_nowait(msg["bytes"])
                        except Exception:
                            pass
                elif msg.get("text"):
                    cmd = msg["text"].strip().lower()
                    if cmd == "stop":
                        stop_event.set()
                        break
                    elif cmd == "ping":
                        try:
                            await websocket.send_json({"type": "pong"})
                        except Exception:
                            pass
        except WebSocketDisconnect:
            stop_event.set()

    async def local_capture_task():
        """For source=local, sample the screen at `fps` and enqueue JPEGs."""
        interval = 1.0 / fps
        while not stop_event.is_set():
            try:
                jpeg = await asyncio.to_thread(capture_local_jpeg, monitor, None, 65)
                try:
                    frame_queue.put_nowait(jpeg)
                except asyncio.QueueFull:
                    try:
                        frame_queue.get_nowait()
                        frame_queue.put_nowait(jpeg)
                    except Exception:
                        pass
            except Exception as e:
                logger.warning(f"local screen capture failed: {e}")
                await websocket.send_json({"type": "error", "message": f"capture failed: {e}"})
                stop_event.set()
                return
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=interval)
            except asyncio.TimeoutError:
                pass

    async def processor_task():
        """Drain the frame queue and OCR each frame in a worker thread."""
        while not stop_event.is_set() or not frame_queue.empty():
            try:
                frame = await asyncio.wait_for(frame_queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            # Drop any older queued frames before OCR'ing - process the freshest one
            while True:
                try:
                    frame = frame_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
            result = await asyncio.to_thread(ocr.process_frame, frame)
            if result.get("frame_skipped"):
                await websocket.send_json({
                    "type": "frame",
                    "frame_skipped": True,
                    "reason": result.get("reason"),
                    "phash_diff": result.get("phash_diff"),
                })
                continue
            for line in result.get("new_lines", []):
                await websocket.send_json({"type": "line", **line})
            await websocket.send_json({
                "type": "frame",
                "frame_skipped": False,
                "lines": len(result.get("new_lines", [])),
            })

    try:
        ocr = LiveScreenOCR(
            session_id=session_id,
            transcripts_dir=SCREEN_TRANSCRIPTS_DIR,
            confidence_floor=confidence,
            frame_diff_threshold=diff,
            max_repeats=max_repeats,
            drop_garbage=drop_garbage,
        )
        await websocket.send_json({
            "type": "ready",
            "session": session_id,
            "source": source,
            "device": ocr.device,
        })
        await websocket.send_json({"type": "status", "stage": "loading_model"})
        await asyncio.to_thread(ocr.ensure_engine)
        await websocket.send_json({"type": "status", "stage": "model_ready"})

        tasks = [
            asyncio.create_task(receiver_task()),
            asyncio.create_task(processor_task()),
        ]
        if source == "local":
            tasks.append(asyncio.create_task(local_capture_task()))
        await asyncio.gather(*tasks)

        flush_info = await asyncio.to_thread(ocr.flush)
        urls = ocr.transcript_urls(base_url="/static/screen_transcripts")
        await websocket.send_json({
            "type": "complete",
            "transcripts": urls,
            "stats": flush_info.get("stats", {}),
        })

    except WebSocketDisconnect:
        if ocr:
            try:
                await asyncio.to_thread(ocr.flush)
            except Exception:
                pass
    except Exception as e:
        logger.exception("Live screen OCR error")
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
        if ocr:
            try:
                await asyncio.to_thread(ocr.flush)
            except Exception:
                pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Offline video transcription: audio (whisper) + screen OCR, separately
# ---------------------------------------------------------------------------

def _ffmpeg_extract_frames(video_path: str, frames_dir: str, fps: float, max_width: int = 1280) -> int:
    """Sample frames from a video at the given fps using ffmpeg.

    Writes frame_000001.jpg, frame_000002.jpg, ... into frames_dir. Scales
    down so OCR doesn't waste cycles on 4K. Returns the number of frames
    written. Raises RuntimeError if ffmpeg fails."""
    import subprocess
    import glob
    os.makedirs(frames_dir, exist_ok=True)
    pattern = os.path.join(frames_dir, "frame_%06d.jpg")
    vf = f"fps={fps:g},scale='min({max_width},iw)':-2"
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", video_path,
        "-vf", vf,
        "-q:v", "4",
        pattern,
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg frame extraction failed: {proc.stderr.decode('utf-8', 'replace')[:500]}")
    return len(glob.glob(os.path.join(frames_dir, "frame_*.jpg")))


def _video_probe_duration(video_path: str) -> Optional[float]:
    """Best-effort duration probe via ffprobe."""
    import subprocess
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", video_path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20,
        )
        if out.returncode == 0:
            return float(out.stdout.decode("utf-8", "replace").strip())
    except Exception:
        pass
    return None


def _run_video_transcribe(
    job_id: str,
    video_path: str,
    audio_model: str,
    language: Optional[str],
    translate: bool,
    do_audio: bool,
    do_screen: bool,
    fps: float,
    confidence: float,
    drop_garbage: bool,
):
    """Background worker: produces an audio transcript and/or a screen OCR
    transcript from a single video file. Both outputs land in the existing
    static mounts under the session id, so the existing download URLs work."""
    import json as _json
    import glob
    import shutil

    session_id = f"video-{job_id[:8]}"
    duration = _video_probe_duration(video_path)
    audio_urls: Dict[str, str] = {}
    screen_urls: Dict[str, str] = {}

    try:
        _update_job(job_id, status="running", progress=2, message="Preparing video...")

        # --- Audio leg --------------------------------------------------
        if do_audio:
            from transcriber import Transcriber, TranscriptionConfig
            _update_job(job_id, progress=5, message="Loading whisper model...")
            cfg = TranscriptionConfig(
                model_size=audio_model,
                language=(language or None),
                translate_to_english=bool(translate),
            )
            tx = Transcriber(cfg)

            last_msg = {"v": ""}
            def _audio_progress(msg: str):
                # Coarse progress: whisper has no real % until segments come in.
                last_msg["v"] = msg
                _update_job(job_id, message=f"Audio: {msg}")

            result = tx.transcribe(video_path, progress_callback=_audio_progress)
            if not result.get("success"):
                raise RuntimeError(f"Audio transcription failed: {result.get('error')}")

            segments = result.get("segments") or []
            text = result.get("text") or ""

            txt_path = os.path.join(TRANSCRIPTS_DIR, f"{session_id}.txt")
            srt_path = os.path.join(TRANSCRIPTS_DIR, f"{session_id}.srt")
            json_path = os.path.join(TRANSCRIPTS_DIR, f"{session_id}.json")
            with open(txt_path, "w", encoding="utf-8") as fh:
                # One segment per line so the .txt is a readable transcript,
                # not a single long blob.
                for seg in segments:
                    line = (seg.get("text") or "").strip()
                    if line:
                        fh.write(line + "\n")
                if not segments and text:
                    fh.write(text)
            with open(srt_path, "w", encoding="utf-8") as fh:
                fh.write(tx.generate_srt(segments))
            with open(json_path, "w", encoding="utf-8") as fh:
                _json.dump({
                    "session": session_id,
                    "language": result.get("language"),
                    "segments": segments,
                }, fh, indent=2, ensure_ascii=False)

            audio_urls = {
                "txt": f"/static/transcripts/{session_id}.txt",
                "srt": f"/static/transcripts/{session_id}.srt",
                "json": f"/static/transcripts/{session_id}.json",
            }
            _update_job(
                job_id,
                progress=55 if do_screen else 95,
                message=f"Audio done: {len(segments)} segment(s).",
            )

        # --- Screen OCR leg --------------------------------------------
        if do_screen:
            from screen_ocr import LiveScreenOCR
            _update_job(job_id, message="Sampling video frames...")

            frames_dir = os.path.join(UPLOADS_DIR, f"frames_{job_id[:8]}")
            try:
                n_frames = _ffmpeg_extract_frames(video_path, frames_dir, fps=fps)
            except Exception as e:
                raise RuntimeError(f"Frame sampling failed: {e}")

            if n_frames == 0:
                _update_job(job_id, message="No frames extracted (silent / unreadable video?).")
            else:
                ocr = LiveScreenOCR(
                    session_id=session_id,
                    transcripts_dir=SCREEN_TRANSCRIPTS_DIR,
                    confidence_floor=confidence,
                    drop_garbage=drop_garbage,
                )
                ocr.ensure_engine()
                _update_job(job_id, message=f"OCRing {n_frames} frames at {fps:g} fps...")

                frame_paths = sorted(glob.glob(os.path.join(frames_dir, "frame_*.jpg")))
                done = 0
                base = 55 if do_audio else 5
                span = (95 - base)
                for fp in frame_paths:
                    try:
                        with open(fp, "rb") as fh:
                            frame_bytes = fh.read()
                        ocr.process_frame(frame_bytes)
                    except Exception as e:
                        logger.warning(f"frame OCR failed for {os.path.basename(fp)}: {e}")
                    finally:
                        try:
                            os.remove(fp)
                        except OSError:
                            pass
                    done += 1
                    # Throttle progress updates to ~1/sec worth of frames
                    if done == 1 or done == n_frames or done % max(1, n_frames // 50) == 0:
                        pct = base + int(span * (done / max(1, n_frames)))
                        _update_job(job_id, progress=pct, message=f"OCR: frame {done}/{n_frames}")

                ocr.flush()
                screen_urls = {
                    "txt": f"/static/screen_transcripts/{session_id}.txt",
                    "json": f"/static/screen_transcripts/{session_id}.json",
                }

            # cleanup frames dir if it still exists
            try:
                shutil.rmtree(frames_dir, ignore_errors=True)
            except OSError:
                pass

        result_payload = {
            "session_id": session_id,
            "duration": duration,
            "audio_transcripts": audio_urls or None,
            "screen_transcripts": screen_urls or None,
        }
        _update_job(
            job_id,
            status="completed",
            progress=100,
            message="Video transcription complete.",
            result=result_payload,
        )
    except Exception as exc:
        logger.exception("Video transcription failed")
        _update_job(job_id, status="failed", error=str(exc), message=f"Video transcribe failed: {exc}")
    finally:
        try:
            if os.path.isfile(video_path):
                os.remove(video_path)
        except OSError:
            pass


@app.post("/api/v1/video/transcribe", response_model=JobResponse)
async def video_transcribe(
    file: UploadFile = File(...),
    audio_model: str = "base",
    language: str = "",
    translate: bool = False,
    do_audio: bool = True,
    do_screen: bool = True,
    fps: float = 1.0,
    confidence: float = 0.6,
    drop_garbage: bool = True,
):
    """Run faster-whisper on the audio and RapidOCR on sampled frames of a
    video, in one job. Each leg produces its own transcript files served
    via the existing static mounts."""
    if not (do_audio or do_screen):
        raise HTTPException(status_code=400, detail="Enable at least one of do_audio / do_screen.")
    fps = max(0.1, min(float(fps), 10.0))
    confidence = max(0.0, min(float(confidence), 1.0))
    lang = (language or "").strip() or None

    saved = _save_upload(file, prefix="vid")
    job_id = _create_job("video_transcribe", {
        "filename": file.filename,
        "do_audio": do_audio,
        "do_screen": do_screen,
        "audio_model": audio_model,
        "language": lang,
        "translate": translate,
        "fps": fps,
        "confidence": confidence,
    })
    thread = threading.Thread(
        target=_run_video_transcribe,
        args=(job_id, saved, audio_model, lang, translate, do_audio, do_screen,
              fps, confidence, drop_garbage),
        daemon=True,
    )
    thread.start()
    return _job_response(job_id)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# Make sure ffmpeg/ffprobe are findable. Some Windows shells don't pick up
# the registry PATH entry that winget adds, which causes audio_studio's
# ffmpeg subprocess calls to fail with WinError 2.
def _ensure_ffmpeg_on_path():
    import shutil
    if shutil.which("ffmpeg") and shutil.which("ffprobe"):
        return
    candidates = [
        r"C:\Program Files\WinGet\Links",
        r"C:\Program Files\Gyan.FFmpeg\bin",
        r"C:\ffmpeg\bin",
        r"C:\ProgramData\chocolatey\bin",
    ]
    for c in candidates:
        if os.path.isfile(os.path.join(c, "ffmpeg.exe")):
            os.environ["PATH"] = c + os.pathsep + os.environ.get("PATH", "")
            return

_ensure_ffmpeg_on_path()


# Load .env file on startup
_env_file = os.path.join(ROOT_DIR, ".env")
if os.path.exists(_env_file):
    with open(_env_file, "r") as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _key, _val = _line.split("=", 1)
                os.environ.setdefault(_key.strip(), _val.strip())

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
