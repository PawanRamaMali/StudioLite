"""
VideoGenerator - Real video generation using CogVideoX, LTX-Video, and Wan 2.1/2.2.

Supports:
- Text-to-Video: Generate video from text prompt
- Image-to-Video: Animate a still image
- Video-to-Video: Extend or modify existing video

Engines:
- Wan 2.1/2.2: Best quality, MoE architecture, 8-16GB VRAM (RECOMMENDED)
- LTX-Video: Fast generation, real-time 30 FPS, up to 161 frames
- CogVideoX: Good quality, 2B/5B models, 6-10s clips
"""
import os
import gc
import torch
import logging
from datetime import datetime
from collections import deque
from threading import Lock

# Use HDD for HuggingFace cache if available (for large model storage)
HDD_HF_CACHE = "/mnt/hdd/huggingface"
if os.path.exists("/mnt/hdd") and os.access("/mnt/hdd", os.W_OK):
    os.environ.setdefault("HF_HOME", HDD_HF_CACHE)
    os.makedirs(HDD_HF_CACHE, exist_ok=True)
from dataclasses import dataclass
from uuid import uuid4
from typing import Optional, Callable, Tuple, Dict, Any

# Root directory
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Logging configuration
LOG_FILE = os.path.join(ROOT_DIR, ".mp", "videogen.log")
MAX_LOG_ENTRIES = 500  # Keep last 500 log entries in memory


class VideoGenLogger:
    """Thread-safe logger for video generation with in-memory buffer."""

    _instance = None
    _lock = Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._logs = deque(maxlen=MAX_LOG_ENTRIES)
        self._log_lock = Lock()
        self._log_file = LOG_FILE

        # Ensure log directory exists
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

    @property
    def log_file(self) -> str:
        """Get the log file path."""
        return self._log_file

    def log(self, level: str, message: str) -> None:
        """Add a log entry."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = f"[{timestamp}] [{level.upper()}] {message}"

        with self._log_lock:
            self._logs.append(entry)
            # Also write to file
            try:
                with open(LOG_FILE, "a") as f:
                    f.write(entry + "\n")
            except Exception:
                pass  # Don't fail on log write errors

    def info(self, message: str) -> None:
        self.log("INFO", message)

    def warning(self, message: str) -> None:
        self.log("WARNING", message)

    def error(self, message: str) -> None:
        self.log("ERROR", message)

    def debug(self, message: str) -> None:
        self.log("DEBUG", message)

    def get_logs(self, limit: int = 100) -> list:
        """Get recent log entries."""
        with self._log_lock:
            logs = list(self._logs)
            return logs[-limit:] if limit else logs

    def clear(self) -> None:
        """Clear in-memory logs."""
        with self._log_lock:
            self._logs.clear()

    def get_log_file_contents(self, lines: int = 200) -> str:
        """Read last N lines from log file."""
        try:
            if not os.path.exists(LOG_FILE):
                return ""
            with open(LOG_FILE, "r") as f:
                all_lines = f.readlines()
                return "".join(all_lines[-lines:])
        except Exception as e:
            return f"Error reading log file: {e}"


# Global logger instance
videogen_logger = VideoGenLogger()


@dataclass
class VideoGenConfig:
    """Configuration for video generation."""
    # Engine selection
    engine: str = "wan"                 # "wan", "ltx", "cogvideox", or "hunyuan"

    # CogVideoX settings
    model_variant: str = "2b"           # "2b" or "5b" for CogVideoX

    # LTX-Video settings
    ltx_model: str = "base"             # "base", "distilled", "0.9.7", "0.9.8"

    # Wan 2.1/2.2 settings
    wan_model: str = "1.3b"             # "1.3b", "14b", "2.2-14b", "2.2-1.3b"
    wan_resolution: str = "480p"        # "480p" or "720p"

    # HunyuanVideo settings
    hunyuan_model: str = "1.5"          # "1.0" or "1.5"
    hunyuan_resolution: str = "720p"    # "540p", "720p", "1080p"

    # Common settings
    num_frames: int = 49                # CogVideoX: 49/81, LTX: 161, Wan: 81, Hunyuan: 129
    num_inference_steps: int = 50       # LTX distilled: 4-10, others: 50
    guidance_scale: float = 6.0         # LTX distilled: 1.0, CogVideoX: 6.0, Wan: 5.0
    width: int = 720                    # LTX: 704, CogVideoX: 720, Wan: 832/1280
    height: int = 480                   # LTX: 512, CogVideoX: 480, Wan: 480/720
    fps: int = 24                       # LTX: 24-30, CogVideoX: 8, Wan: 16
    enable_cpu_offload: bool = True
    quantization: str = "auto"          # "none", "int8", "auto"
    use_vae_slicing: bool = True
    use_vae_tiling: bool = True         # For LTX/Wan memory optimization
    seed: Optional[int] = None


class VideoGenerator:
    """
    Video Generation Engine supporting CogVideoX and LTX-Video.

    Features:
    - Auto-configures based on available VRAM
    - Supports text-to-video, image-to-video, video-to-video
    - Memory-efficient with CPU offloading and VAE optimizations
    """

    def __init__(self, config: VideoGenConfig = None):
        self.config = config or VideoGenConfig()
        self._pipeline = None
        self._current_mode = None
        self._loaded_model_id = None
        self._current_engine = None

    def estimate_vram_required(self) -> float:
        """
        Estimate VRAM required for current configuration in GB.

        Based on real-world testing:
        - Wan 1.3B with 121 frames @ 480p uses ~42-45GB VRAM
        - Frame count significantly impacts memory (latent space growth)
        - Resolution scales ~2.25x from 480p to 720p

        Returns:
            Estimated VRAM needed in GB
        """
        # Base VRAM for model weights + inference overhead
        base_vram = {
            "wan": {
                "1.3b": 12.0,   # Actual: ~12GB base for model
                "14b": 28.0,    # ~28GB for 14B model
                "2.2-14b": 30.0,
                "2.2-1.3b": 12.0,
            },
            "hunyuan": {
                "1.0": 24.0,    # HunyuanVideo 1.0 needs ~24GB
                "1.5": 28.0,    # HunyuanVideo 1.5 needs ~28GB
            },
            "ltx": {
                "base": 12.0,
                "distilled": 10.0,
                "0.9.7": 12.0,
                "0.9.8": 18.0,
            },
            "cogvideox": {
                "2b": 10.0,
                "5b": 18.0,
            },
        }

        engine = self.config.engine
        if engine == "wan":
            model_key = self.config.wan_model.lower()
            model_vram = base_vram["wan"].get(model_key, 12.0)
        elif engine == "hunyuan":
            model_key = self.config.hunyuan_model.lower()
            model_vram = base_vram["hunyuan"].get(model_key, 24.0)
        elif engine == "ltx":
            model_key = self.config.ltx_model.lower()
            model_vram = base_vram["ltx"].get(model_key, 12.0)
        else:
            model_key = self.config.model_variant.lower()
            model_vram = base_vram["cogvideox"].get(model_key, 12.0)

        # Frame count scaling - critical for Wan!
        # Based on testing: 121 frames needs ~42GB vs 49 frames ~18GB
        # Roughly: 0.25GB per frame beyond 49 frames
        num_frames = self.config.num_frames
        if num_frames <= 49:
            frame_overhead = 0
        elif num_frames <= 81:
            frame_overhead = (num_frames - 49) * 0.2  # ~6GB for 81 frames
        else:
            frame_overhead = 6.0 + (num_frames - 81) * 0.35  # ~20GB more for 121 frames

        # Resolution scaling (480p baseline)
        if engine == "wan" and self.config.wan_resolution == "720p":
            resolution_factor = 2.25  # 720p uses ~2.25x more VRAM
        elif engine == "hunyuan":
            if self.config.hunyuan_resolution == "1080p":
                resolution_factor = 4.0  # 1080p is much larger
            elif self.config.hunyuan_resolution == "720p":
                resolution_factor = 2.0
            else:  # 540p
                resolution_factor = 1.0
        else:
            resolution_factor = 1.0

        # VAE decoding spike (temporary but can cause OOM)
        vae_overhead = 4.0

        # CPU offloading dramatically reduces VRAM - only one layer on GPU at a time
        if self.config.enable_cpu_offload:
            # With CPU offloading, peak VRAM is much lower
            offload_factor = 0.4  # ~40% of full VRAM needed
            model_vram = model_vram * offload_factor
            vae_overhead = 2.0  # VAE still needs some VRAM

        # Total with 10% safety margin
        total = (model_vram + frame_overhead * resolution_factor + vae_overhead) * 1.1

        return round(total, 1)

    def check_memory_for_generation(self) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Check if there's enough VRAM for generation with current config.

        Returns:
            Tuple of (can_proceed, message, info_dict)
        """
        if not torch.cuda.is_available():
            return False, "CUDA GPU required", {}

        try:
            # Get current memory state
            props = torch.cuda.get_device_properties(0)
            total_vram = props.total_memory / (1024**3)
            allocated = torch.cuda.memory_allocated(0) / (1024**3)
            reserved = torch.cuda.memory_reserved(0) / (1024**3)
            free_vram = total_vram - reserved  # Use reserved as it's what's actually held

            # Estimate required VRAM
            required_vram = self.estimate_vram_required()

            info = {
                "gpu_name": props.name,
                "total_vram": round(total_vram, 2),
                "allocated_vram": round(allocated, 2),
                "reserved_vram": round(reserved, 2),
                "free_vram": round(free_vram, 2),
                "required_vram": required_vram,
                "engine": self.config.engine,
                "model": self.config.wan_model if self.config.engine == "wan" else
                         self.config.ltx_model if self.config.engine == "ltx" else
                         self.config.model_variant,
                "num_frames": self.config.num_frames,
            }

            if free_vram < required_vram:
                # Build helpful suggestion
                suggestions = []
                if self.config.num_frames > 49:
                    suggestions.append(f"reduce frames from {self.config.num_frames} to 49")
                if self.config.engine == "wan" and self.config.wan_resolution == "720p":
                    suggestions.append("use 480p instead of 720p")
                if self.config.engine == "wan" and "14b" in self.config.wan_model:
                    suggestions.append("use 1.3b model instead of 14b")
                if not suggestions:
                    suggestions.append("close other GPU applications")

                suggestion_text = " or ".join(suggestions)

                return False, (
                    f"Insufficient VRAM: {free_vram:.1f}GB free, ~{required_vram:.1f}GB required. "
                    f"Try: {suggestion_text}."
                ), info

            return True, "OK", info

        except Exception as e:
            return False, f"Memory check failed: {e}", {}

    def check_requirements(self) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Check system requirements for video generation.

        Returns:
            Tuple of (success, message, info_dict)
        """
        try:
            import torch
        except ImportError:
            return False, "PyTorch not installed", {}

        if not torch.cuda.is_available():
            return False, "CUDA GPU required for video generation", {}

        try:
            props = torch.cuda.get_device_properties(0)
            total_vram = props.total_memory / (1024**3)
            allocated = torch.cuda.memory_allocated(0) / (1024**3)
            free_vram = total_vram - allocated

            info = {
                "gpu_name": props.name,
                "total_vram": round(total_vram, 2),
                "free_vram": round(free_vram, 2),
                "cuda_version": torch.version.cuda,
            }

            if free_vram < 6:
                return False, f"Insufficient VRAM ({free_vram:.1f}GB). Minimum 6GB required.", info

            return True, "OK", info

        except Exception as e:
            return False, f"GPU check failed: {e}", {}

    def auto_configure(self) -> None:
        """Auto-configure settings based on available VRAM."""
        ok, msg, info = self.check_requirements()
        if not ok:
            raise RuntimeError(msg)

        vram = info["free_vram"]

        if vram >= 24:
            # High-end GPU: Use Wan 2.2 14B for best quality
            self.config.engine = "wan"
            self.config.wan_model = "2.2-14b"
            self.config.wan_resolution = "720p"
            self.config.num_frames = 81
            self.config.enable_cpu_offload = False
            self.config.fps = 16
            self.config.guidance_scale = 5.0
        elif vram >= 16:
            # Mid-range: Wan 2.1 14B with CPU offload
            self.config.engine = "wan"
            self.config.wan_model = "14b"
            self.config.wan_resolution = "480p"
            self.config.num_frames = 81
            self.config.enable_cpu_offload = True
            self.config.fps = 16
            self.config.guidance_scale = 5.0
        elif vram >= 12:
            # Lower mid-range: Wan 2.1 1.3B for best quality/VRAM ratio
            self.config.engine = "wan"
            self.config.wan_model = "1.3b"
            self.config.wan_resolution = "480p"
            self.config.num_frames = 81
            self.config.enable_cpu_offload = True
            self.config.fps = 16
            self.config.guidance_scale = 5.0
        elif vram >= 8:
            # Entry-level: Wan 2.1 1.3B (best for 8GB)
            self.config.engine = "wan"
            self.config.wan_model = "1.3b"
            self.config.wan_resolution = "480p"
            self.config.num_frames = 49
            self.config.enable_cpu_offload = True
            self.config.fps = 16
            self.config.guidance_scale = 5.0
        else:
            # Minimal: CogVideoX 2B with aggressive optimization
            self.config.engine = "cogvideox"
            self.config.model_variant = "2b"
            self.config.quantization = "int8"
            self.config.enable_cpu_offload = True
            self.config.num_frames = 49

    def _get_ltx_model_id(self, mode: str) -> str:
        """Get LTX-Video HuggingFace model ID."""
        if self.config.ltx_model == "distilled":
            return "Lightricks/LTX-Video-0.9.7-distilled"
        elif self.config.ltx_model == "0.9.7":
            return "Lightricks/LTX-Video-0.9.7-dev"
        elif self.config.ltx_model == "0.9.8":
            return "Lightricks/LTX-Video-0.9.8-13B-distilled"
        else:
            return "Lightricks/LTX-Video"

    def _get_cogvideox_model_id(self, mode: str) -> str:
        """Get CogVideoX HuggingFace model ID."""
        if mode == "text2video":
            return f"THUDM/CogVideoX-{self.config.model_variant}"
        elif mode == "image2video":
            return "THUDM/CogVideoX-5b-I2V"
        elif mode == "video2video":
            return f"THUDM/CogVideoX-{self.config.model_variant}"
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def _get_wan_model_id(self, mode: str) -> str:
        """Get Wan 2.1/2.2 HuggingFace model ID."""
        model = self.config.wan_model.lower()
        resolution = self.config.wan_resolution.lower()

        if mode == "text2video":
            if model == "2.2-14b" or model == "14b-2.2":
                return "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
            elif model == "14b":
                return "Wan-AI/Wan2.1-T2V-14B-Diffusers"
            elif model == "2.2-1.3b" or model == "1.3b-2.2":
                # Wan 2.2 doesn't have 1.3B, use 2.1
                return "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
            else:  # default to 1.3b (Wan 2.1)
                return "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
        elif mode == "image2video":
            # Wan I2V models available:
            # - Wan2.2-I2V-A14B-Diffusers (best quality, 14B)
            # - Wan2.1-I2V-14B-480P-Diffusers
            # - Wan2.1-I2V-14B-720P-Diffusers
            if "2.2" in model:
                return "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
            elif resolution == "720p":
                return "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers"
            else:  # 480p
                return "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
        else:
            raise ValueError(f"Wan does not support mode: {mode}")

    def _load_wan_pipeline(self, mode: str, progress_callback: Callable = None) -> None:
        """Load Wan 2.1/2.2 pipeline."""
        from diffusers import WanPipeline, WanImageToVideoPipeline, AutoencoderKLWan
        import os

        model_id = self._get_wan_model_id(mode)

        # Skip if already loaded
        if (self._pipeline is not None and
            self._current_mode == mode and
            self._loaded_model_id == model_id and
            self._current_engine == "wan"):
            return

        # Unload existing pipeline
        if self._pipeline is not None:
            if progress_callback:
                progress_callback(0, 100, "Unloading previous model...")
            self.unload()

        # Set longer timeout for large model downloads
        os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")

        if progress_callback:
            progress_callback(10, 100, f"Downloading Wan VAE (~1GB)...")

        # Wan requires VAE to be loaded separately in float32 for precision
        # Retry logic for network issues
        max_retries = 3
        for attempt in range(max_retries):
            try:
                vae = AutoencoderKLWan.from_pretrained(
                    model_id,
                    subfolder="vae",
                    torch_dtype=torch.float32,
                )
                break
            except OSError as e:
                if "Connection timed out" in str(e) or "timed out" in str(e).lower():
                    if attempt < max_retries - 1:
                        if progress_callback:
                            progress_callback(10, 100, f"Network timeout, retrying ({attempt + 2}/{max_retries})...")
                        continue
                    raise RuntimeError(
                        f"Download timed out after {max_retries} attempts. "
                        "Try again later or check your network connection. "
                        "You can also pre-download with: "
                        f"huggingface-cli download {model_id}"
                    ) from e
                raise

        if progress_callback:
            progress_callback(40, 100, f"Downloading Wan pipeline (~7GB)...")

        # Select pipeline class
        if mode == "image2video":
            PipelineClass = WanImageToVideoPipeline
        else:
            PipelineClass = WanPipeline

        for attempt in range(max_retries):
            try:
                self._pipeline = PipelineClass.from_pretrained(
                    model_id,
                    vae=vae,
                    torch_dtype=torch.bfloat16,
                )
                break
            except OSError as e:
                if "Connection timed out" in str(e) or "timed out" in str(e).lower():
                    if attempt < max_retries - 1:
                        if progress_callback:
                            progress_callback(40, 100, f"Network timeout, retrying ({attempt + 2}/{max_retries})...")
                        continue
                    raise RuntimeError(
                        f"Download timed out after {max_retries} attempts. "
                        "Try again later or check your network connection."
                    ) from e
                raise

        if progress_callback:
            progress_callback(70, 100, "Configuring memory optimization...")

        # Apply memory optimizations
        if self.config.enable_cpu_offload:
            self._pipeline.enable_model_cpu_offload()
        else:
            self._pipeline.to("cuda")

        if self.config.use_vae_tiling:
            self._pipeline.vae.enable_tiling()

        if self.config.use_vae_slicing:
            self._pipeline.vae.enable_slicing()

        self._current_mode = mode
        self._loaded_model_id = model_id
        self._current_engine = "wan"

        if progress_callback:
            progress_callback(100, 100, "Wan model loaded!")

    def _load_ltx_pipeline(self, mode: str, progress_callback: Callable = None) -> None:
        """Load LTX-Video pipeline."""
        from diffusers import LTXPipeline, LTXImageToVideoPipeline

        model_id = self._get_ltx_model_id(mode)

        # Skip if already loaded
        if (self._pipeline is not None and
            self._current_mode == mode and
            self._loaded_model_id == model_id and
            self._current_engine == "ltx"):
            return

        # Unload existing pipeline
        if self._pipeline is not None:
            if progress_callback:
                progress_callback(0, 100, "Unloading previous model...")
            self.unload()

        if progress_callback:
            progress_callback(10, 100, f"Loading LTX-Video ({model_id})...")

        # Select pipeline class
        if mode == "image2video":
            PipelineClass = LTXImageToVideoPipeline
        else:
            PipelineClass = LTXPipeline

        dtype = torch.bfloat16

        if progress_callback:
            progress_callback(30, 100, f"Downloading model weights...")

        self._pipeline = PipelineClass.from_pretrained(
            model_id,
            torch_dtype=dtype,
        )

        if progress_callback:
            progress_callback(70, 100, "Configuring memory optimization...")

        # Apply memory optimizations
        if self.config.enable_cpu_offload:
            self._pipeline.enable_model_cpu_offload()
        else:
            self._pipeline.to("cuda")

        if self.config.use_vae_tiling:
            self._pipeline.vae.enable_tiling()

        if self.config.use_vae_slicing:
            self._pipeline.vae.enable_slicing()

        self._current_mode = mode
        self._loaded_model_id = model_id
        self._current_engine = "ltx"

        if progress_callback:
            progress_callback(100, 100, "LTX-Video model loaded!")

    def _load_cogvideox_pipeline(self, mode: str, progress_callback: Callable = None) -> None:
        """Load CogVideoX pipeline."""
        from diffusers import (
            CogVideoXPipeline,
            CogVideoXImageToVideoPipeline,
            CogVideoXVideoToVideoPipeline,
        )

        model_id = self._get_cogvideox_model_id(mode)

        # Skip if already loaded
        if (self._pipeline is not None and
            self._current_mode == mode and
            self._loaded_model_id == model_id and
            self._current_engine == "cogvideox"):
            return

        # Unload existing pipeline
        if self._pipeline is not None:
            if progress_callback:
                progress_callback(0, 100, "Unloading previous model...")
            self.unload()

        if progress_callback:
            progress_callback(10, 100, f"Loading CogVideoX ({model_id})...")

        # Select pipeline class
        if mode == "text2video":
            PipelineClass = CogVideoXPipeline
        elif mode == "image2video":
            PipelineClass = CogVideoXImageToVideoPipeline
        elif mode == "video2video":
            PipelineClass = CogVideoXVideoToVideoPipeline
        else:
            raise ValueError(f"Unknown mode: {mode}")

        dtype = torch.bfloat16

        if progress_callback:
            progress_callback(20, 100, f"Downloading model weights...")

        self._pipeline = PipelineClass.from_pretrained(
            model_id,
            torch_dtype=dtype,
        )

        if progress_callback:
            progress_callback(70, 100, "Configuring memory optimization...")

        # Apply memory optimizations
        if self.config.enable_cpu_offload:
            self._pipeline.enable_model_cpu_offload()
        else:
            self._pipeline.to("cuda")

        if self.config.use_vae_slicing:
            self._pipeline.vae.enable_slicing()

        self._current_mode = mode
        self._loaded_model_id = model_id
        self._current_engine = "cogvideox"

        if progress_callback:
            progress_callback(100, 100, "CogVideoX model loaded!")

    def _load_hunyuan_pipeline(self, mode: str, progress_callback: Callable = None) -> None:
        """Load HunyuanVideo pipeline."""
        from diffusers import HunyuanVideoPipeline
        from transformers import LlamaModel, LlamaTokenizerFast, CLIPTextModel, CLIPTokenizer

        # Get model ID based on version
        if self.config.hunyuan_model == "1.5":
            model_id = "hunyuanvideo-community/HunyuanVideo-1.5"
        else:
            model_id = "hunyuanvideo-community/HunyuanVideo"

        # Skip if already loaded
        if (self._pipeline is not None and
            self._current_mode == mode and
            self._loaded_model_id == model_id and
            self._current_engine == "hunyuan"):
            return

        # Unload existing pipeline
        if self._pipeline is not None:
            if progress_callback:
                progress_callback(0, 100, "Unloading previous model...")
            self.unload()

        if progress_callback:
            progress_callback(10, 100, f"Loading HunyuanVideo ({model_id})...")

        # Load pipeline with bfloat16 for efficiency
        dtype = torch.bfloat16

        if progress_callback:
            progress_callback(30, 100, f"Downloading model weights (~14GB)...")

        self._pipeline = HunyuanVideoPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
        )

        if progress_callback:
            progress_callback(70, 100, "Configuring memory optimization...")

        # Apply memory optimizations
        if self.config.enable_cpu_offload:
            self._pipeline.enable_model_cpu_offload()
        else:
            self._pipeline.to("cuda")

        if self.config.use_vae_tiling:
            self._pipeline.vae.enable_tiling()

        if self.config.use_vae_slicing:
            self._pipeline.vae.enable_slicing()

        self._current_mode = mode
        self._loaded_model_id = model_id
        self._current_engine = "hunyuan"

        if progress_callback:
            progress_callback(100, 100, "HunyuanVideo model loaded!")

    def load_pipeline(self, mode: str = "text2video", progress_callback: Callable = None) -> None:
        """
        Load the appropriate pipeline for the specified mode.

        Args:
            mode: "text2video", "image2video", or "video2video"
            progress_callback: Optional callback for progress updates
        """
        if self.config.engine == "wan":
            self._load_wan_pipeline(mode, progress_callback)
        elif self.config.engine == "ltx":
            self._load_ltx_pipeline(mode, progress_callback)
        elif self.config.engine == "hunyuan":
            self._load_hunyuan_pipeline(mode, progress_callback)
        else:
            self._load_cogvideox_pipeline(mode, progress_callback)

    def _generate_ltx_text2video(
        self,
        prompt: str,
        negative_prompt: str = None,
        progress_callback: Callable = None,
    ) -> str:
        """Generate video using LTX-Video."""
        from diffusers.utils import export_to_video
        import time

        if progress_callback:
            progress_callback(0, 100, "Loading LTX-Video model...")

        # Load pipeline without passing progress_callback to avoid scale mismatch
        self._load_ltx_pipeline("text2video", None)

        if progress_callback:
            progress_callback(10, 100, "Model loaded. Starting inference...")

        generator = None
        if self.config.seed is not None:
            generator = torch.Generator(device="cuda").manual_seed(self.config.seed)

        # LTX-specific parameters
        is_distilled = "distilled" in self.config.ltx_model.lower()

        # Create step callback for progress reporting
        total_steps = self.config.num_inference_steps
        start_time = time.time()

        def step_callback(pipe, step_index, timestep, callback_kwargs):
            if progress_callback:
                # Map inference steps to 10-90% of progress bar
                step_progress = 10 + int((step_index / total_steps) * 80)
                elapsed = time.time() - start_time
                eta = (elapsed / max(step_index, 1)) * (total_steps - step_index)
                progress_callback(
                    step_progress, 100,
                    f"Inference step {step_index + 1}/{total_steps} (ETA: {int(eta)}s)"
                )
            return callback_kwargs

        output = self._pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt or "worst quality, inconsistent motion, blurry, jittery, distorted",
            width=self.config.width,
            height=self.config.height,
            num_frames=self.config.num_frames,
            num_inference_steps=self.config.num_inference_steps,
            guidance_scale=1.0 if is_distilled else self.config.guidance_scale,
            decode_timestep=0.03,
            decode_noise_scale=0.025,
            generator=generator,
            callback_on_step_end=step_callback,
        )

        if progress_callback:
            progress_callback(92, 100, "Encoding video to MP4...")

        output_dir = os.path.join(ROOT_DIR, ".mp")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"ltx_video_{uuid4()}.mp4")

        export_to_video(output.frames[0], output_path, fps=self.config.fps)

        if progress_callback:
            progress_callback(100, 100, "Complete!")

        return output_path

    def _generate_ltx_image2video(
        self,
        image_path: str,
        prompt: str,
        progress_callback: Callable = None,
    ) -> str:
        """Generate video from image using LTX-Video."""
        from PIL import Image
        from diffusers.utils import export_to_video
        import time

        if progress_callback:
            progress_callback(0, 100, "Loading image...")

        image = Image.open(image_path).convert("RGB")
        image = image.resize((self.config.width, self.config.height))

        if progress_callback:
            progress_callback(5, 100, "Loading LTX-Video model...")

        # Load pipeline without passing progress_callback to avoid scale mismatch
        self._load_ltx_pipeline("image2video", None)

        if progress_callback:
            progress_callback(10, 100, "Model loaded. Starting inference...")

        generator = None
        if self.config.seed is not None:
            generator = torch.Generator(device="cuda").manual_seed(self.config.seed)

        is_distilled = "distilled" in self.config.ltx_model.lower()

        # Create step callback for progress reporting
        total_steps = self.config.num_inference_steps
        start_time = time.time()

        def step_callback(pipe, step_index, timestep, callback_kwargs):
            if progress_callback:
                # Map inference steps to 10-90% of progress bar
                step_progress = 10 + int((step_index / total_steps) * 80)
                elapsed = time.time() - start_time
                eta = (elapsed / max(step_index, 1)) * (total_steps - step_index)
                progress_callback(
                    step_progress, 100,
                    f"Inference step {step_index + 1}/{total_steps} (ETA: {int(eta)}s)"
                )
            return callback_kwargs

        output = self._pipeline(
            image=image,
            prompt=prompt,
            negative_prompt="worst quality, inconsistent motion, blurry, jittery, distorted",
            width=self.config.width,
            height=self.config.height,
            num_frames=self.config.num_frames,
            num_inference_steps=self.config.num_inference_steps,
            guidance_scale=1.0 if is_distilled else self.config.guidance_scale,
            decode_timestep=0.03,
            decode_noise_scale=0.025,
            generator=generator,
            callback_on_step_end=step_callback,
        )

        if progress_callback:
            progress_callback(92, 100, "Encoding video to MP4...")

        output_dir = os.path.join(ROOT_DIR, ".mp")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"ltx_video_{uuid4()}.mp4")

        export_to_video(output.frames[0], output_path, fps=self.config.fps)

        if progress_callback:
            progress_callback(100, 100, "Complete!")

        return output_path

    def _generate_wan_text2video(
        self,
        prompt: str,
        negative_prompt: str = None,
        progress_callback: Callable = None,
    ) -> str:
        """Generate video using Wan 2.1/2.2."""
        from diffusers.utils import export_to_video
        from diffusers.callbacks import PipelineCallback
        import time

        if progress_callback:
            progress_callback(0, 100, "Loading Wan model...")

        # Load pipeline without passing progress_callback to avoid scale mismatch
        self._load_wan_pipeline("text2video", None)

        if progress_callback:
            progress_callback(10, 100, "Model loaded. Starting inference...")

        generator = None
        if self.config.seed is not None:
            generator = torch.Generator(device="cuda").manual_seed(self.config.seed)

        # Wan-specific resolution settings
        if self.config.wan_resolution == "720p":
            width, height = 1280, 720
        else:  # 480p
            width, height = 832, 480

        # Create step callback for progress reporting
        total_steps = self.config.num_inference_steps
        start_time = time.time()

        def step_callback(pipe, step_index, timestep, callback_kwargs):
            if progress_callback:
                # Map inference steps to 10-90% of progress bar
                step_progress = 10 + int((step_index / total_steps) * 80)
                elapsed = time.time() - start_time
                eta = (elapsed / max(step_index, 1)) * (total_steps - step_index)
                progress_callback(
                    step_progress, 100,
                    f"Inference step {step_index + 1}/{total_steps} (ETA: {int(eta)}s)"
                )
            return callback_kwargs

        output = self._pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt or "low quality, blurry, distorted, disfigured",
            width=width,
            height=height,
            num_frames=self.config.num_frames,
            num_inference_steps=self.config.num_inference_steps,
            guidance_scale=self.config.guidance_scale,
            generator=generator,
            callback_on_step_end=step_callback,
        )

        if progress_callback:
            progress_callback(92, 100, "Encoding video to MP4...")

        output_dir = os.path.join(ROOT_DIR, ".mp")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"wan_video_{uuid4()}.mp4")

        export_to_video(output.frames[0], output_path, fps=self.config.fps)

        if progress_callback:
            progress_callback(100, 100, "Complete!")

        return output_path

    def _generate_wan_image2video(
        self,
        image_path: str,
        prompt: str,
        progress_callback: Callable = None,
    ) -> str:
        """Generate video from image using Wan 2.1/2.2."""
        from PIL import Image
        from diffusers.utils import export_to_video
        import time

        if progress_callback:
            progress_callback(0, 100, "Loading image...")

        # Wan-specific resolution settings
        if self.config.wan_resolution == "720p":
            width, height = 1280, 720
        else:  # 480p
            width, height = 832, 480

        image = Image.open(image_path).convert("RGB")
        image = image.resize((width, height))

        if progress_callback:
            progress_callback(5, 100, "Loading Wan I2V model...")

        # Load pipeline without passing progress_callback to avoid scale mismatch
        self._load_wan_pipeline("image2video", None)

        if progress_callback:
            progress_callback(10, 100, "Model loaded. Starting inference...")

        generator = None
        if self.config.seed is not None:
            generator = torch.Generator(device="cuda").manual_seed(self.config.seed)

        # Create step callback for progress reporting
        total_steps = self.config.num_inference_steps
        start_time = time.time()

        def step_callback(pipe, step_index, timestep, callback_kwargs):
            if progress_callback:
                step_progress = 10 + int((step_index / total_steps) * 80)
                elapsed = time.time() - start_time
                eta = (elapsed / max(step_index, 1)) * (total_steps - step_index)
                progress_callback(
                    step_progress, 100,
                    f"Inference step {step_index + 1}/{total_steps} (ETA: {int(eta)}s)"
                )
            return callback_kwargs

        output = self._pipeline(
            image=image,
            prompt=prompt,
            negative_prompt="low quality, blurry, distorted, disfigured",
            width=width,
            height=height,
            num_frames=self.config.num_frames,
            num_inference_steps=self.config.num_inference_steps,
            guidance_scale=self.config.guidance_scale,
            generator=generator,
            callback_on_step_end=step_callback,
        )

        if progress_callback:
            progress_callback(92, 100, "Encoding video to MP4...")

        output_dir = os.path.join(ROOT_DIR, ".mp")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"wan_video_{uuid4()}.mp4")

        export_to_video(output.frames[0], output_path, fps=self.config.fps)

        if progress_callback:
            progress_callback(100, 100, "Complete!")

        return output_path

    def _generate_hunyuan_text2video(
        self,
        prompt: str,
        negative_prompt: str = None,
        progress_callback: Callable = None,
    ) -> str:
        """Generate video using HunyuanVideo."""
        from diffusers.utils import export_to_video
        import time

        if progress_callback:
            progress_callback(0, 100, "Loading HunyuanVideo model...")

        # Load pipeline without passing progress_callback to avoid scale mismatch
        self._load_hunyuan_pipeline("text2video", None)

        if progress_callback:
            progress_callback(10, 100, "Model loaded. Starting inference...")

        generator = None
        if self.config.seed is not None:
            generator = torch.Generator(device="cuda").manual_seed(self.config.seed)

        # HunyuanVideo resolution settings
        if self.config.hunyuan_resolution == "1080p":
            width, height = 1920, 1080
        elif self.config.hunyuan_resolution == "720p":
            width, height = 1280, 720
        else:  # 540p
            width, height = 960, 540

        # Create step callback for progress reporting
        total_steps = self.config.num_inference_steps
        start_time = time.time()

        def step_callback(pipe, step_index, timestep, callback_kwargs):
            if progress_callback:
                # Map inference steps to 10-90% of progress bar
                step_progress = 10 + int((step_index / total_steps) * 80)
                elapsed = time.time() - start_time
                eta = (elapsed / max(step_index, 1)) * (total_steps - step_index)
                progress_callback(
                    step_progress, 100,
                    f"Inference step {step_index + 1}/{total_steps} (ETA: {int(eta)}s)"
                )
            return callback_kwargs

        output = self._pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt or "low quality, blurry, distorted, watermark",
            width=width,
            height=height,
            num_frames=self.config.num_frames,
            num_inference_steps=self.config.num_inference_steps,
            guidance_scale=self.config.guidance_scale,
            generator=generator,
            callback_on_step_end=step_callback,
        )

        if progress_callback:
            progress_callback(92, 100, "Encoding video to MP4...")

        output_dir = os.path.join(ROOT_DIR, ".mp")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"hunyuan_video_{uuid4()}.mp4")

        export_to_video(output.frames[0], output_path, fps=self.config.fps)

        if progress_callback:
            progress_callback(100, 100, "Complete!")

        return output_path

    def generate_text2video(
        self,
        prompt: str,
        negative_prompt: str = None,
        progress_callback: Callable = None,
    ) -> str:
        """
        Generate video from text prompt.

        Args:
            prompt: Text description of the video
            negative_prompt: What to avoid in the video
            progress_callback: Callback(step, total, message)

        Returns:
            Path to generated video file

        Raises:
            RuntimeError: If insufficient VRAM available
        """
        # Pre-generation memory check
        can_proceed, msg, info = self.check_memory_for_generation()
        if not can_proceed:
            videogen_logger.error(f"Memory check failed: {msg}")
            raise RuntimeError(msg)

        videogen_logger.info(f"Starting T2V generation: engine={self.config.engine}, frames={self.config.num_frames}")
        videogen_logger.info(f"Prompt: {prompt[:100]}...")
        videogen_logger.info(f"VRAM info: {info.get('free_vram', 'N/A')}GB free / {info.get('total_vram', 'N/A')}GB total")

        if self.config.engine == "wan":
            return self._generate_wan_text2video(prompt, negative_prompt, progress_callback)
        elif self.config.engine == "ltx":
            return self._generate_ltx_text2video(prompt, negative_prompt, progress_callback)
        elif self.config.engine == "hunyuan":
            return self._generate_hunyuan_text2video(prompt, negative_prompt, progress_callback)

        # CogVideoX text2video
        import time
        from diffusers.utils import export_to_video

        if progress_callback:
            progress_callback(0, 100, "Loading CogVideoX model...")

        if self._current_mode != "text2video" or self._current_engine != "cogvideox":
            # Load pipeline without passing progress_callback to avoid scale mismatch
            self.load_pipeline("text2video", None)

        if progress_callback:
            progress_callback(10, 100, "Model loaded. Starting inference...")

        generator = None
        if self.config.seed is not None:
            generator = torch.Generator(device="cuda").manual_seed(self.config.seed)

        # Create step callback for progress reporting
        total_steps = self.config.num_inference_steps
        start_time = time.time()

        def step_callback(pipe, step_index, timestep, callback_kwargs):
            if progress_callback:
                # Map inference steps to 10-90% of progress bar
                step_progress = 10 + int((step_index / total_steps) * 80)
                elapsed = time.time() - start_time
                eta = (elapsed / max(step_index, 1)) * (total_steps - step_index)
                progress_callback(
                    step_progress, 100,
                    f"Inference step {step_index + 1}/{total_steps} (ETA: {int(eta)}s)"
                )
            return callback_kwargs

        output = self._pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt or "low quality, blurry, distorted",
            num_frames=self.config.num_frames,
            num_inference_steps=self.config.num_inference_steps,
            guidance_scale=self.config.guidance_scale,
            generator=generator,
            callback_on_step_end=step_callback,
        )

        if progress_callback:
            progress_callback(92, 100, "Encoding video to MP4...")

        output_dir = os.path.join(ROOT_DIR, ".mp")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"cogvideo_{uuid4()}.mp4")

        export_to_video(output.frames[0], output_path, fps=self.config.fps)

        if progress_callback:
            progress_callback(100, 100, "Complete!")

        return output_path

    def generate_image2video(
        self,
        image_path: str,
        prompt: str,
        progress_callback: Callable = None,
    ) -> str:
        """
        Generate video from an image (animate the image).

        Args:
            image_path: Path to input image
            prompt: Text description of the motion/video
            progress_callback: Callback(step, total, message)

        Returns:
            Path to generated video file

        Raises:
            RuntimeError: If insufficient VRAM available
        """
        # Pre-generation memory check
        can_proceed, msg, info = self.check_memory_for_generation()
        if not can_proceed:
            raise RuntimeError(msg)

        if self.config.engine == "wan":
            return self._generate_wan_image2video(image_path, prompt, progress_callback)
        elif self.config.engine == "ltx":
            return self._generate_ltx_image2video(image_path, prompt, progress_callback)

        # CogVideoX image2video
        from PIL import Image
        from diffusers.utils import export_to_video
        import time

        if progress_callback:
            progress_callback(0, 100, "Loading image...")

        image = Image.open(image_path).convert("RGB")
        image = image.resize((self.config.width, self.config.height))

        if progress_callback:
            progress_callback(5, 100, "Loading CogVideoX model...")

        if self._current_mode != "image2video" or self._current_engine != "cogvideox":
            # Load pipeline without passing progress_callback to avoid scale mismatch
            self.load_pipeline("image2video", None)

        if progress_callback:
            progress_callback(10, 100, "Model loaded. Starting inference...")

        generator = None
        if self.config.seed is not None:
            generator = torch.Generator(device="cuda").manual_seed(self.config.seed)

        # Create step callback for progress reporting
        total_steps = self.config.num_inference_steps
        start_time = time.time()

        def step_callback(pipe, step_index, timestep, callback_kwargs):
            if progress_callback:
                # Map inference steps to 10-90% of progress bar
                step_progress = 10 + int((step_index / total_steps) * 80)
                elapsed = time.time() - start_time
                eta = (elapsed / max(step_index, 1)) * (total_steps - step_index)
                progress_callback(
                    step_progress, 100,
                    f"Inference step {step_index + 1}/{total_steps} (ETA: {int(eta)}s)"
                )
            return callback_kwargs

        output = self._pipeline(
            prompt=prompt,
            image=image,
            num_frames=self.config.num_frames,
            num_inference_steps=self.config.num_inference_steps,
            guidance_scale=self.config.guidance_scale,
            generator=generator,
            callback_on_step_end=step_callback,
        )

        if progress_callback:
            progress_callback(92, 100, "Encoding video to MP4...")

        output_dir = os.path.join(ROOT_DIR, ".mp")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"cogvideo_{uuid4()}.mp4")

        export_to_video(output.frames[0], output_path, fps=self.config.fps)

        if progress_callback:
            progress_callback(100, 100, "Complete!")

        return output_path

    def _extract_last_frame(self, video_path: str) -> str:
        """
        Extract the last frame from a video for I2V continuation.

        Args:
            video_path: Path to input video

        Returns:
            Path to extracted frame image
        """
        import cv2
        from PIL import Image

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        # Get total frame count and seek to last frame
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)

        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise RuntimeError("Failed to extract last frame from video")

        # Convert BGR to RGB and save
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)

        output_dir = os.path.join(ROOT_DIR, ".mp")
        os.makedirs(output_dir, exist_ok=True)
        frame_path = os.path.join(output_dir, f"last_frame_{uuid4()}.png")
        img.save(frame_path)

        return frame_path

    def _generate_wan_video2video(
        self,
        video_path: str,
        prompt: str,
        strength: float = 0.8,
        progress_callback: Callable = None,
    ) -> str:
        """
        Extend video using Wan's I2V capability.
        Extracts last frame and generates continuation.
        """
        if progress_callback:
            progress_callback(0, 100, "Extracting last frame from video...")

        # Extract last frame for continuation
        last_frame_path = self._extract_last_frame(video_path)
        videogen_logger.info(f"Extracted last frame: {last_frame_path}")

        if progress_callback:
            progress_callback(5, 100, "Generating video continuation...")

        # Use I2V to continue from last frame
        continuation_prompt = f"Continue the motion: {prompt}"
        result = self._generate_wan_image2video(
            last_frame_path,
            continuation_prompt,
            progress_callback
        )

        # Clean up temp frame
        try:
            os.remove(last_frame_path)
        except Exception:
            pass

        return result

    def _generate_ltx_video2video(
        self,
        video_path: str,
        prompt: str,
        strength: float = 0.8,
        progress_callback: Callable = None,
    ) -> str:
        """
        Extend video using LTX's I2V capability.
        Extracts last frame and generates continuation.
        """
        if progress_callback:
            progress_callback(0, 100, "Extracting last frame from video...")

        # Extract last frame for continuation
        last_frame_path = self._extract_last_frame(video_path)
        videogen_logger.info(f"Extracted last frame: {last_frame_path}")

        if progress_callback:
            progress_callback(5, 100, "Generating video continuation...")

        # Use I2V to continue from last frame
        continuation_prompt = f"Continue the motion: {prompt}"
        result = self._generate_ltx_image2video(
            last_frame_path,
            continuation_prompt,
            progress_callback
        )

        # Clean up temp frame
        try:
            os.remove(last_frame_path)
        except Exception:
            pass

        return result

    def generate_video2video(
        self,
        video_path: str,
        prompt: str,
        strength: float = 0.8,
        progress_callback: Callable = None,
    ) -> str:
        """
        Transform or extend an existing video.

        For Wan and LTX: Uses I2V approach (extract last frame, generate continuation)
        For CogVideoX: Uses native video-to-video pipeline

        Args:
            video_path: Path to input video
            prompt: Text description of desired output
            strength: How much to transform (0-1, higher = more change)
            progress_callback: Callback(step, total, message)

        Returns:
            Path to generated video file

        Raises:
            RuntimeError: If insufficient VRAM available
        """
        # Pre-generation memory check
        can_proceed, msg, info = self.check_memory_for_generation()
        if not can_proceed:
            raise RuntimeError(msg)

        videogen_logger.info(f"Starting V2V generation: engine={self.config.engine}")
        videogen_logger.info(f"Input video: {video_path}")
        videogen_logger.info(f"Prompt: {prompt[:100]}...")

        # Wan uses I2V-based video extension
        if self.config.engine == "wan":
            return self._generate_wan_video2video(video_path, prompt, strength, progress_callback)

        # LTX uses I2V-based video extension
        if self.config.engine == "ltx":
            return self._generate_ltx_video2video(video_path, prompt, strength, progress_callback)

        # CogVideoX has native video-to-video support
        import time
        from diffusers.utils import load_video, export_to_video

        if progress_callback:
            progress_callback(0, 100, "Loading video...")

        video = load_video(video_path)

        if progress_callback:
            progress_callback(5, 100, "Loading model...")

        if self._current_mode != "video2video" or self._current_engine != "cogvideox":
            # Load pipeline without passing progress_callback to avoid scale mismatch
            self.load_pipeline("video2video", None)

        if progress_callback:
            progress_callback(10, 100, "Model loaded. Starting inference...")

        generator = None
        if self.config.seed is not None:
            generator = torch.Generator(device="cuda").manual_seed(self.config.seed)

        # Create step callback for progress reporting
        total_steps = self.config.num_inference_steps
        start_time = time.time()

        def step_callback(pipe, step_index, timestep, callback_kwargs):
            if progress_callback:
                # Map inference steps to 10-90% of progress bar
                step_progress = 10 + int((step_index / total_steps) * 80)
                elapsed = time.time() - start_time
                eta = (elapsed / max(step_index, 1)) * (total_steps - step_index)
                progress_callback(
                    step_progress, 100,
                    f"Inference step {step_index + 1}/{total_steps} (ETA: {int(eta)}s)"
                )
            return callback_kwargs

        output = self._pipeline(
            prompt=prompt,
            video=video,
            strength=strength,
            num_inference_steps=self.config.num_inference_steps,
            guidance_scale=self.config.guidance_scale,
            generator=generator,
            callback_on_step_end=step_callback,
        )

        if progress_callback:
            progress_callback(92, 100, "Encoding video to MP4...")

        output_dir = os.path.join(ROOT_DIR, ".mp")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"generated_video_{uuid4()}.mp4")

        export_to_video(output.frames[0], output_path, fps=self.config.fps)

        if progress_callback:
            progress_callback(100, 100, "Complete!")

        return output_path

    def unload(self) -> None:
        """Free GPU memory by unloading the pipeline."""
        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None
            self._current_mode = None
            self._loaded_model_id = None
            self._current_engine = None

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


def check_system_requirements() -> Tuple[bool, str, Dict[str, Any]]:
    """
    Quick system check for UI display.

    Returns:
        Tuple of (success, message, info_dict)
    """
    gen = VideoGenerator()
    return gen.check_requirements()


def get_recommended_config() -> VideoGenConfig:
    """
    Get recommended config based on system hardware.

    Returns:
        VideoGenConfig with optimal settings
    """
    gen = VideoGenerator()
    gen.auto_configure()
    return gen.config


def get_available_engines() -> Dict[str, Dict[str, Any]]:
    """
    Get available video generation engines and their features.

    Returns:
        Dict of engine info
    """
    return {
        "wan": {
            "name": "Wan 2.1/2.2 (Recommended)",
            "description": "Best quality, MoE architecture, 8-16GB VRAM",
            "models": ["1.3b", "14b", "2.2-14b"],
            "max_frames": 81,
            "fps": 16,
            "min_vram": 8,
        },
        "hunyuan": {
            "name": "HunyuanVideo (High VRAM)",
            "description": "Tencent's 8.3B model, 24GB+ VRAM, up to 1080p",
            "models": ["1.0", "1.5"],
            "max_frames": 129,
            "fps": 24,
            "min_vram": 24,
            "resolutions": ["540p", "720p", "1080p"],
        },
        "ltx": {
            "name": "LTX-Video",
            "description": "Fast generation, real-time 30 FPS, 161 frames",
            "models": ["base", "distilled", "0.9.7", "0.9.8"],
            "max_frames": 161,
            "fps": 24,
            "min_vram": 10,
        },
        "cogvideox": {
            "name": "CogVideoX",
            "description": "Good quality, works on 8GB+ VRAM",
            "models": ["2b", "5b"],
            "max_frames": 81,
            "fps": 8,
            "min_vram": 8,
        },
    }


def check_memory_for_config(config: VideoGenConfig) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Check if there's enough VRAM for the given configuration.

    Args:
        config: VideoGenConfig to check

    Returns:
        Tuple of (can_proceed, message, info_dict)
    """
    gen = VideoGenerator(config)
    return gen.check_memory_for_generation()


def get_vram_info() -> Dict[str, Any]:
    """
    Get current VRAM usage information.

    Returns:
        Dict with GPU info, or empty dict if no GPU
    """
    if not torch.cuda.is_available():
        return {"available": False, "message": "No CUDA GPU detected"}

    try:
        props = torch.cuda.get_device_properties(0)
        total_vram = props.total_memory / (1024**3)
        allocated = torch.cuda.memory_allocated(0) / (1024**3)
        reserved = torch.cuda.memory_reserved(0) / (1024**3)
        free_vram = total_vram - reserved

        return {
            "available": True,
            "gpu_name": props.name,
            "total_vram_gb": round(total_vram, 2),
            "allocated_vram_gb": round(allocated, 2),
            "reserved_vram_gb": round(reserved, 2),
            "free_vram_gb": round(free_vram, 2),
            "cuda_version": torch.version.cuda,
        }
    except Exception as e:
        return {"available": False, "message": str(e)}


def get_generation_logs(lines: int = 200) -> str:
    """
    Get recent video generation logs.

    Args:
        lines: Number of lines to return

    Returns:
        Log content as string
    """
    return videogen_logger.get_log_file_contents(lines)


def get_recent_logs(limit: int = 100) -> list:
    """
    Get recent log entries from memory.

    Args:
        limit: Maximum number of entries to return

    Returns:
        List of log entries
    """
    return videogen_logger.get_logs(limit)


def clear_logs() -> None:
    """Clear in-memory logs."""
    videogen_logger.clear()


def log_info(message: str) -> None:
    """Log an info message."""
    videogen_logger.info(message)


def log_error(message: str) -> None:
    """Log an error message."""
    videogen_logger.error(message)


def concatenate_videos(video_paths: list, output_path: str = None, fps: int = None) -> str:
    """
    Concatenate multiple video clips into a single video.

    Args:
        video_paths: List of paths to video files to concatenate
        output_path: Optional output path (auto-generated if not provided)
        fps: Optional FPS (uses first video's FPS if not provided)

    Returns:
        Path to concatenated video
    """
    from moviepy.editor import VideoFileClip, concatenate_videoclips

    if not video_paths:
        raise ValueError("No video paths provided")

    videogen_logger.info(f"Concatenating {len(video_paths)} videos...")

    clips = []
    for path in video_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Video not found: {path}")
        clips.append(VideoFileClip(path))

    # Concatenate clips
    final_clip = concatenate_videoclips(clips, method="compose")

    # Generate output path if not provided
    if output_path is None:
        output_dir = os.path.join(ROOT_DIR, ".mp")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"concatenated_{uuid4()}.mp4")

    # Write output
    target_fps = fps or clips[0].fps
    final_clip.write_videofile(
        output_path,
        fps=target_fps,
        codec="libx264",
        audio_codec="aac" if final_clip.audio else None,
        logger=None  # Suppress moviepy output
    )

    # Cleanup
    for clip in clips:
        clip.close()
    final_clip.close()

    videogen_logger.info(f"Concatenated video saved: {output_path}")
    return output_path


def add_audio_to_video(video_path: str, audio_path: str, output_path: str = None,
                       volume: float = 1.0) -> str:
    """
    Add audio track to a video.

    Args:
        video_path: Path to video file
        audio_path: Path to audio file (MP3, WAV, etc.)
        output_path: Optional output path
        volume: Audio volume multiplier (0.0 - 2.0)

    Returns:
        Path to video with audio
    """
    from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip

    videogen_logger.info(f"Adding audio to video: {video_path}")

    video = VideoFileClip(video_path)
    audio = AudioFileClip(audio_path)

    # Adjust audio duration to match video
    if audio.duration > video.duration:
        audio = audio.subclip(0, video.duration)
    elif audio.duration < video.duration:
        # Loop audio to fill video duration
        loops_needed = int(video.duration / audio.duration) + 1
        from moviepy.editor import concatenate_audioclips
        audio = concatenate_audioclips([audio] * loops_needed).subclip(0, video.duration)

    # Apply volume
    audio = audio.volumex(volume)

    # Combine with existing audio if present
    if video.audio is not None:
        combined_audio = CompositeAudioClip([video.audio, audio])
        video = video.set_audio(combined_audio)
    else:
        video = video.set_audio(audio)

    # Generate output path if not provided
    if output_path is None:
        output_dir = os.path.join(ROOT_DIR, ".mp")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"video_with_audio_{uuid4()}.mp4")

    video.write_videofile(
        output_path,
        codec="libx264",
        audio_codec="aac",
        logger=None
    )

    video.close()
    audio.close()

    videogen_logger.info(f"Video with audio saved: {output_path}")
    return output_path


def generate_scene_video(
    generator: VideoGenerator,
    scenes: list,
    progress_callback: Callable = None,
    concatenate: bool = True
) -> str:
    """
    Generate a longer video by creating multiple scenes and optionally concatenating them.

    Args:
        generator: VideoGenerator instance
        scenes: List of scene descriptions (strings) or dicts with 'prompt' and optional 'image'
        progress_callback: Optional callback for progress updates
        concatenate: If True, concatenate all scenes into one video

    Returns:
        Path to final video (concatenated) or list of video paths

    Example:
        scenes = [
            "A sunrise over mountains, golden light",
            "Birds flying across the sky",
            {"prompt": "A waterfall in forest", "image": "waterfall.jpg"},  # I2V for this scene
        ]
        video = generate_scene_video(gen, scenes)
    """
    if not scenes:
        raise ValueError("No scenes provided")

    videogen_logger.info(f"Generating {len(scenes)} scenes...")

    video_paths = []
    total_scenes = len(scenes)

    for i, scene in enumerate(scenes):
        if progress_callback:
            progress_callback(i, total_scenes, f"Generating scene {i+1}/{total_scenes}...")

        # Parse scene - can be string or dict
        if isinstance(scene, str):
            prompt = scene
            image_path = None
        else:
            prompt = scene.get("prompt", "")
            image_path = scene.get("image")

        try:
            if image_path and os.path.exists(image_path):
                # Image-to-video for this scene
                video_path = generator.generate_image2video(
                    image_path,
                    prompt,
                    progress_callback=None  # Don't pass callback to avoid conflicts
                )
            else:
                # Text-to-video
                video_path = generator.generate_text2video(
                    prompt,
                    progress_callback=None
                )

            video_paths.append(video_path)
            videogen_logger.info(f"Scene {i+1} generated: {video_path}")

        except Exception as e:
            videogen_logger.error(f"Failed to generate scene {i+1}: {e}")
            raise

    if progress_callback:
        progress_callback(total_scenes, total_scenes, "Scenes generated!")

    # Concatenate if requested
    if concatenate and len(video_paths) > 1:
        if progress_callback:
            progress_callback(total_scenes, total_scenes + 1, "Concatenating scenes...")

        final_video = concatenate_videos(video_paths, fps=generator.config.fps)

        if progress_callback:
            progress_callback(total_scenes + 1, total_scenes + 1, "Complete!")

        return final_video
    elif len(video_paths) == 1:
        return video_paths[0]
    else:
        return video_paths  # Return list if not concatenating
