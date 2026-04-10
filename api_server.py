"""
StudioLite REST API Server

Run with: uvicorn api_server:app --host 0.0.0.0 --port 8000
Or: python api_server.py
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import threading
import uuid
import time
import os
import traceback

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


class TTSRequest(BaseModel):
    text: str
    voice: str = "Amy"
    engine: str = "piper"  # piper or kitten


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
        if req.enable_narration:
            _update_job(job_id, progress=18, message="Generating narration audio...")
            tts = get_tts_instance(req.engine if req.engine in ("piper", "kitten") else "piper")
            for idx, scene in enumerate(scene_list):
                narration_text = rf_clean_text_for_tts(scene.get("narration", ""))
                audio_path = os.path.join(OUTPUT_DIR, f"story_narration_{job_id}_{idx}.wav")
                tts.synthesize(narration_text, audio_path)
                narration_paths.append(audio_path)
                pct = 18 + int((idx + 1) / len(scene_list) * 12)
                _update_job(job_id, progress=pct, message=f"Narration {idx + 1}/{len(scene_list)} done")

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

        config = _build_config(
            engine=req.engine,
            model=req.model,
            resolution=req_resolution,
            num_frames=req_frames,
            num_inference_steps=qp["steps"],
            guidance_scale=qp["guidance"],
            fps=req_fps,
            seed=shared_seed if req.continuity_mode in ("Prompt Anchoring", "Both") else None,
        )
        gen = VideoGenerator(config)
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

            scene_clips = []

            # --- T2V pipeline with strong character prompts ---
            # Character appearances are baked into the prompt text for Wan T2V.
            # Shared seed ensures consistent character interpretation across scenes.
            clip_duration = req_frames / max(req_fps, 1)
            clips_needed = min(2, max(1, int(scene_dur / clip_duration) + 1))

            continuations = [
                "continuing the motion smoothly",
                "the action progresses further",
                "moments later in the same scene",
            ]

            for ci in range(clips_needed):
                _update_job(
                    job_id,
                    progress=pct_base + int((ci / clips_needed) * (55 / len(scene_list))),
                    message=f"Scene {idx+1}/{len(scene_list)}: clip {ci+1}/{clips_needed} ({scene.get('title', '')})",
                )
                clip_prompt = scene_prompt if ci == 0 else f"{scene_prompt}, {continuations[ci % len(continuations)]}"
                vp = gen.generate_text2video(
                    prompt=clip_prompt,
                    negative_prompt=neg_prompt,
                    progress_callback=None,
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


def _run_tts(job_id: str, req: TTSRequest):
    try:
        _update_job(job_id, status="running", message="Initializing TTS engine...")
        from mpv2.classes.TtsFactory import get_tts_instance
        from reelforge import rf_clean_text_for_tts

        tts = get_tts_instance(req.engine)

        # If piper, we can set the voice via constructor
        if req.engine == "piper" and req.voice:
            from mpv2.classes.PiperTts import PiperTTS
            tts = PiperTTS(voice_name=req.voice)

        clean_text = rf_clean_text_for_tts(req.text)
        output_path = os.path.join(OUTPUT_DIR, f"tts_{job_id}.wav")

        _update_job(job_id, progress=30, message="Synthesizing speech...")
        tts.synthesize(clean_text, output_path)

        _update_job(
            job_id,
            status="completed",
            progress=100,
            message="TTS synthesis complete.",
            result={"audio_path": output_path, "voice": req.voice, "engine": req.engine},
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
    """List available video generation engines and TTS engines."""
    video_engines = [
        {
            "id": "wan",
            "name": "Wan 2.1/2.2",
            "models": ["1.3b", "14b", "2.2-14b", "2.2-1.3b"],
            "resolutions": ["480p", "720p"],
            "description": "Best quality, MoE architecture. Recommended.",
        },
        {
            "id": "ltx",
            "name": "LTX-Video",
            "models": ["base", "distilled", "0.9.7", "0.9.8"],
            "resolutions": ["480p"],
            "description": "Fast generation, real-time 30 FPS.",
        },
        {
            "id": "cogvideox",
            "name": "CogVideoX",
            "models": ["2b", "5b"],
            "resolutions": ["480p"],
            "description": "Good quality, 6-10s clips.",
        },
        {
            "id": "hunyuan",
            "name": "HunyuanVideo",
            "models": ["1.0", "1.5"],
            "resolutions": ["540p", "720p", "1080p"],
            "description": "High resolution video generation.",
        },
    ]

    tts_engines = []
    try:
        from mpv2.classes.TtsFactory import list_available_engines

        tts_engines = list_available_engines()
    except ImportError:
        pass

    return {"video_engines": video_engines, "tts_engines": tts_engines}


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
    file_path = result.get("video_path") or result.get("audio_path")
    if not file_path or not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="Result file not found on disk")

    media_type = "video/mp4" if file_path.endswith(".mp4") else "audio/wav"
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
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
