"""
StudioLite REST API Server

Run with: uvicorn api_server:app --host 0.0.0.0 --port 8000
Or: python api_server.py
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.responses import FileResponse
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
        _update_job(job_id, status="running", message="Generating story script...")
        from videogen import VideoGenerator, VideoGenConfig, concatenate_videos
        from mpv2.classes.TtsFactory import get_tts_instance
        from reelforge import rf_clean_text_for_tts
        from mpv2.llm_provider import generate_text, select_backend, select_model
        import json
        import re

        # Step 1 -- Generate story script via LLM
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
            scene_list = [
                {
                    "title": f"Scene {i + 1}",
                    "visual": req.concept,
                    "narration": f"Scene {i + 1} of the story.",
                    "duration": 5,
                }
                for i in range(req.num_scenes)
            ]

        scene_list = scene_list[: req.num_scenes]
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
        # Use shared seed for visual consistency across scenes
        import random as _rng
        shared_seed = _rng.randint(0, 2**31)

        # Use request params or sensible defaults
        req_frames = getattr(req, "num_frames", 49) or 49
        req_fps = getattr(req, "fps", 16) or 16
        req_resolution = getattr(req, "resolution", "480p") or "480p"

        config = _build_config(
            engine=req.engine,
            model=req.model,
            resolution=req_resolution,
            num_frames=req_frames,
            num_inference_steps=30,
            guidance_scale=6.0,  # Slightly higher for better prompt adherence
            fps=req_fps,
            seed=shared_seed if req.continuity_mode in ("Prompt Anchoring", "Both") else None,
        )
        gen = VideoGenerator(config)
        video_paths = []

        # Strong negative prompt to reduce AI artifacts
        neg_prompt = (
            "low quality, blurry, distorted, disfigured, bad anatomy, "
            "objects appearing from nowhere, teleporting, sudden scene change, "
            "inconsistent subject, morphing, glitch, watermark, text overlay, "
            "extra limbs, missing limbs, floating objects, unnatural movement"
        )

        for idx, scene in enumerate(scene_list):
            scene_prompt = scene.get("visual", req.concept)

            # Always inject visual identity for character consistency
            if visual_identity:
                scene_prompt = f"{visual_identity}. {scene_prompt}"

            # Also inject any user-provided visual anchor
            user_anchor = getattr(req, "visual_anchor", "")
            if user_anchor:
                scene_prompt = f"{scene_prompt}. {user_anchor}"

            # Add quality boosters
            scene_prompt = f"{scene_prompt}, smooth continuous motion, cinematic quality, consistent subject appearance, photorealistic"

            pct_base = 30 + int(idx / len(scene_list) * 55)
            _update_job(
                job_id,
                progress=pct_base,
                message=f"Generating scene {idx + 1}/{len(scene_list)}: {scene.get('title', '')}",
            )

            video_path = gen.generate_text2video(
                prompt=scene_prompt,
                negative_prompt=neg_prompt,
                progress_callback=None,
            )
            video_paths.append(video_path)

        # Step 4 -- Stretch each video to match scene duration + add narration
        from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips, vfx
        import soundfile as sf
        import numpy as np

        _update_job(job_id, progress=86, message="Syncing video duration and narration...")
        final_scene_paths = []

        for idx, vp in enumerate(video_paths):
            scene = scene_list[idx] if idx < len(scene_list) else {}
            target_duration = scene.get("duration", 5)
            if not isinstance(target_duration, (int, float)):
                target_duration = 5
            target_duration = max(3, min(15, float(target_duration)))

            try:
                clip = VideoFileClip(vp)
                clip_dur = clip.duration

                # Stretch or loop video to match target scene duration
                if clip_dur < target_duration:
                    # Slow down the clip to fill target duration
                    speed_factor = clip_dur / target_duration
                    if speed_factor > 0.3:  # Don't slow down more than 3x
                        stretched = clip.fx(vfx.speedx, speed_factor)
                    else:
                        # Loop the clip to fill duration
                        loops_needed = int(target_duration / clip_dur) + 1
                        stretched = concatenate_videoclips([clip] * loops_needed)
                        stretched = stretched.subclip(0, target_duration)
                else:
                    # Trim to target duration
                    stretched = clip.subclip(0, target_duration)

                # Add narration audio to this scene if available
                if idx < len(narration_paths):
                    ap = narration_paths[idx]
                    if ap and os.path.exists(ap) and os.path.getsize(ap) > 100:
                        narr_audio = AudioFileClip(ap)
                        # If narration is shorter than video, pad with silence
                        if narr_audio.duration < stretched.duration:
                            stretched = stretched.set_audio(narr_audio)
                        else:
                            # Trim narration to video length
                            narr_audio = narr_audio.subclip(0, stretched.duration)
                            stretched = stretched.set_audio(narr_audio)

                # Write final scene
                scene_out = os.path.join(OUTPUT_DIR, f"story_scene_{job_id}_{idx}.mp4")
                stretched.write_videofile(
                    scene_out, fps=req_fps, codec="libx264",
                    audio_codec="aac" if stretched.audio else None,
                    logger=None,
                )
                final_scene_paths.append(scene_out)

                clip.close()
                stretched.close()
            except Exception as e:
                # Fallback: use original video as-is
                final_scene_paths.append(vp)

            pct = 86 + int((idx + 1) / len(video_paths) * 6)
            _update_job(job_id, progress=pct, message=f"Scene {idx+1} synced ({target_duration:.0f}s)")

        video_paths = final_scene_paths

        # Step 5 -- Concatenate all scenes
        _update_job(job_id, progress=93, message="Assembling final movie...")

        # Use transitions if configured
        transition_type = getattr(req, "transition_type", "cut") or "cut"
        if transition_type != "cut" and len(video_paths) > 1:
            try:
                from transitions import concatenate_with_transitions
                final_video = concatenate_with_transitions(
                    video_paths, transition_type=transition_type,
                    duration=0.75, fps=config.fps,
                )
            except Exception:
                final_video = concatenate_videos(video_paths, fps=config.fps)
        else:
            final_video = concatenate_videos(video_paths, fps=config.fps)

        _update_job(
            job_id,
            status="completed",
            progress=100,
            message="Story generation complete.",
            result={
                "video_path": final_video,
                "scene_videos": video_paths,
                "narration_audio": narration_paths,
                "scenes": scene_list,
                "visual_identity": visual_identity,
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
            allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)
            reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)
            free_vram = total_vram - reserved
            gpu_info = {
                "available": True,
                "gpu_name": props.name,
                "total_vram_gb": round(total_vram, 2),
                "allocated_vram_gb": round(allocated, 2),
                "reserved_vram_gb": round(reserved, 2),
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


class ScriptRequest(BaseModel):
    concept: str
    num_scenes: int = 4
    genre: str = "Cinematic"
    mood: str = "Epic"


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

        prompt = f"""You are a professional cinematographer writing a shot list for a {req.num_scenes}-scene short film.

STORY: {req.concept}
GENRE: {req.genre}
MOOD: {req.mood}

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
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
