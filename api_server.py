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

        prompt = f"""Create a {req.num_scenes}-scene cinematic story for a short film.

Concept: {req.concept}
Genre: {req.genre}
Mood: {req.mood}

You MUST return a JSON object with two keys:

1. "visual_identity": A single paragraph describing the CONSISTENT visual elements across ALL scenes.
2. "scenes": A JSON array where each scene has:
   - "title": Short scene title (3-5 words)
   - "visual": Detailed cinematic description for AI video generation.
   - "narration": Voiceover text (1-3 sentences)
   - "duration": Seconds (3-8)

Return ONLY this JSON:
{{
  "visual_identity": "Consistent visual description...",
  "scenes": [
    {{"title": "...", "visual": "...", "narration": "...", "duration": 5}},
    ...
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
            match = re.search(r'\[.*\]', response, re.DOTALL)
            if match:
                try:
                    scene_list = json.loads(match.group())
                except json.JSONDecodeError:
                    pass

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
        config = _build_config(
            engine=req.engine,
            model=req.model,
            resolution="480p",
            num_frames=49,
            num_inference_steps=30,
            guidance_scale=5.0,
            fps=16,
            seed=None,
        )
        gen = VideoGenerator(config)
        video_paths = []

        for idx, scene in enumerate(scene_list):
            scene_prompt = scene.get("visual", req.concept)
            if visual_identity and req.continuity_mode == "Prompt Anchoring":
                scene_prompt = f"{scene_prompt}. {visual_identity}"

            pct_base = 30 + int(idx / len(scene_list) * 55)
            _update_job(
                job_id,
                progress=pct_base,
                message=f"Generating scene {idx + 1}/{len(scene_list)}: {scene.get('title', '')}",
            )

            video_path = gen.generate_text2video(
                prompt=scene_prompt,
                progress_callback=None,
            )
            video_paths.append(video_path)

        # Step 4 -- Add narration to individual clips if available
        if narration_paths:
            from videogen import add_audio_to_video

            _update_job(job_id, progress=88, message="Adding narration to scenes...")
            narrated_paths = []
            for vp, ap in zip(video_paths, narration_paths):
                try:
                    narrated = add_audio_to_video(vp, ap)
                    narrated_paths.append(narrated)
                except Exception:
                    narrated_paths.append(vp)
            video_paths = narrated_paths

        # Step 5 -- Concatenate all scenes
        _update_job(job_id, progress=93, message="Concatenating scenes...")
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
