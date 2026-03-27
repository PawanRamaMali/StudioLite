"""
VideoGenerator - Real video generation using CogVideoX and LTX-Video.

Supports:
- Text-to-Video: Generate video from text prompt
- Image-to-Video: Animate a still image
- Video-to-Video: Extend or modify existing video

Engines:
- CogVideoX: Good quality, 2B/5B models, 6-10s clips
- LTX-Video: Best quality, real-time generation, 30 FPS, up to 161 frames
"""
import os
import gc
import torch
from dataclasses import dataclass
from uuid import uuid4
from typing import Optional, Callable, Tuple, Dict, Any

# Root directory
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


@dataclass
class VideoGenConfig:
    """Configuration for video generation."""
    # Engine selection
    engine: str = "cogvideox"           # "cogvideox" or "ltx"

    # CogVideoX settings
    model_variant: str = "2b"           # "2b" or "5b" for CogVideoX

    # LTX-Video settings
    ltx_model: str = "base"             # "base", "distilled", "0.9.7", "0.9.8"

    # Common settings
    num_frames: int = 49                # CogVideoX: 49/81, LTX: 161 recommended
    num_inference_steps: int = 50       # LTX distilled: 4-10, others: 50
    guidance_scale: float = 6.0         # LTX distilled: 1.0, CogVideoX: 6.0
    width: int = 720                    # LTX: 704, CogVideoX: 720
    height: int = 480                   # LTX: 512, CogVideoX: 480
    fps: int = 24                       # LTX: 24-30, CogVideoX: 8
    enable_cpu_offload: bool = True
    quantization: str = "auto"          # "none", "int8", "auto"
    use_vae_slicing: bool = True
    use_vae_tiling: bool = True         # For LTX memory optimization
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
            # High-end GPU: Use LTX-Video for best quality
            self.config.engine = "ltx"
            self.config.ltx_model = "0.9.7"
            self.config.num_frames = 161
            self.config.enable_cpu_offload = False
            self.config.width = 768
            self.config.height = 512
            self.config.fps = 24
            self.config.guidance_scale = 5.0
        elif vram >= 16:
            # Mid-range: LTX distilled for speed
            self.config.engine = "ltx"
            self.config.ltx_model = "distilled"
            self.config.num_frames = 161
            self.config.enable_cpu_offload = True
            self.config.width = 704
            self.config.height = 512
            self.config.fps = 24
            self.config.guidance_scale = 1.0
            self.config.num_inference_steps = 8
        elif vram >= 12:
            # Lower mid-range: CogVideoX 5B or LTX base
            self.config.engine = "ltx"
            self.config.ltx_model = "base"
            self.config.num_frames = 81
            self.config.enable_cpu_offload = True
            self.config.width = 704
            self.config.height = 480
            self.config.fps = 24
        elif vram >= 8:
            # Entry-level: CogVideoX 2B
            self.config.engine = "cogvideox"
            self.config.model_variant = "2b"
            self.config.quantization = "int8"
            self.config.enable_cpu_offload = True
            self.config.num_frames = 49
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

    def load_pipeline(self, mode: str = "text2video", progress_callback: Callable = None) -> None:
        """
        Load the appropriate pipeline for the specified mode.

        Args:
            mode: "text2video", "image2video", or "video2video"
            progress_callback: Optional callback for progress updates
        """
        if self.config.engine == "ltx":
            self._load_ltx_pipeline(mode, progress_callback)
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

        if progress_callback:
            progress_callback(0, 5, "Loading LTX-Video model...")

        self._load_ltx_pipeline("text2video", progress_callback)

        if progress_callback:
            progress_callback(1, 5, "Generating video frames (this may take a while)...")

        generator = None
        if self.config.seed is not None:
            generator = torch.Generator(device="cuda").manual_seed(self.config.seed)

        # LTX-specific parameters
        is_distilled = "distilled" in self.config.ltx_model.lower()

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
        )

        if progress_callback:
            progress_callback(4, 5, "Encoding video...")

        output_dir = os.path.join(ROOT_DIR, ".mp")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"ltx_video_{uuid4()}.mp4")

        export_to_video(output.frames[0], output_path, fps=self.config.fps)

        if progress_callback:
            progress_callback(5, 5, "Complete!")

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

        if progress_callback:
            progress_callback(0, 5, "Loading image...")

        image = Image.open(image_path).convert("RGB")
        image = image.resize((self.config.width, self.config.height))

        if progress_callback:
            progress_callback(1, 5, "Loading LTX-Video model...")

        self._load_ltx_pipeline("image2video", progress_callback)

        if progress_callback:
            progress_callback(2, 5, "Generating video frames...")

        generator = None
        if self.config.seed is not None:
            generator = torch.Generator(device="cuda").manual_seed(self.config.seed)

        is_distilled = "distilled" in self.config.ltx_model.lower()

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
        )

        if progress_callback:
            progress_callback(4, 5, "Encoding video...")

        output_dir = os.path.join(ROOT_DIR, ".mp")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"ltx_video_{uuid4()}.mp4")

        export_to_video(output.frames[0], output_path, fps=self.config.fps)

        if progress_callback:
            progress_callback(5, 5, "Complete!")

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
        """
        if self.config.engine == "ltx":
            return self._generate_ltx_text2video(prompt, negative_prompt, progress_callback)

        # CogVideoX text2video
        if progress_callback:
            progress_callback(0, 5, "Loading CogVideoX model...")

        if self._current_mode != "text2video" or self._current_engine != "cogvideox":
            self.load_pipeline("text2video", progress_callback)

        if progress_callback:
            progress_callback(1, 5, "Generating video frames...")

        generator = None
        if self.config.seed is not None:
            generator = torch.Generator(device="cuda").manual_seed(self.config.seed)

        output = self._pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt or "low quality, blurry, distorted",
            num_frames=self.config.num_frames,
            num_inference_steps=self.config.num_inference_steps,
            guidance_scale=self.config.guidance_scale,
            generator=generator,
        )

        if progress_callback:
            progress_callback(4, 5, "Encoding video...")

        from diffusers.utils import export_to_video

        output_dir = os.path.join(ROOT_DIR, ".mp")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"cogvideo_{uuid4()}.mp4")

        export_to_video(output.frames[0], output_path, fps=self.config.fps)

        if progress_callback:
            progress_callback(5, 5, "Complete!")

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
        """
        if self.config.engine == "ltx":
            return self._generate_ltx_image2video(image_path, prompt, progress_callback)

        # CogVideoX image2video
        from PIL import Image

        if progress_callback:
            progress_callback(0, 5, "Loading image...")

        image = Image.open(image_path).convert("RGB")
        image = image.resize((self.config.width, self.config.height))

        if progress_callback:
            progress_callback(1, 5, "Loading CogVideoX model...")

        if self._current_mode != "image2video" or self._current_engine != "cogvideox":
            self.load_pipeline("image2video", progress_callback)

        if progress_callback:
            progress_callback(2, 5, "Generating video frames...")

        generator = None
        if self.config.seed is not None:
            generator = torch.Generator(device="cuda").manual_seed(self.config.seed)

        output = self._pipeline(
            prompt=prompt,
            image=image,
            num_frames=self.config.num_frames,
            num_inference_steps=self.config.num_inference_steps,
            guidance_scale=self.config.guidance_scale,
            generator=generator,
        )

        if progress_callback:
            progress_callback(4, 5, "Encoding video...")

        from diffusers.utils import export_to_video

        output_dir = os.path.join(ROOT_DIR, ".mp")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"cogvideo_{uuid4()}.mp4")

        export_to_video(output.frames[0], output_path, fps=self.config.fps)

        if progress_callback:
            progress_callback(5, 5, "Complete!")

        return output_path

    def generate_video2video(
        self,
        video_path: str,
        prompt: str,
        strength: float = 0.8,
        progress_callback: Callable = None,
    ) -> str:
        """
        Transform or extend an existing video.

        Args:
            video_path: Path to input video
            prompt: Text description of desired output
            strength: How much to transform (0-1, higher = more change)
            progress_callback: Callback(step, total, message)

        Returns:
            Path to generated video file
        """
        # Video2Video only supported by CogVideoX currently
        if self.config.engine == "ltx":
            # Fall back to CogVideoX for video2video
            original_engine = self.config.engine
            self.config.engine = "cogvideox"

        if progress_callback:
            progress_callback(0, 5, "Loading video...")

        from diffusers.utils import load_video

        video = load_video(video_path)

        if progress_callback:
            progress_callback(1, 5, "Loading model...")

        if self._current_mode != "video2video" or self._current_engine != "cogvideox":
            self.load_pipeline("video2video", progress_callback)

        if progress_callback:
            progress_callback(2, 5, "Transforming video...")

        generator = None
        if self.config.seed is not None:
            generator = torch.Generator(device="cuda").manual_seed(self.config.seed)

        output = self._pipeline(
            prompt=prompt,
            video=video,
            strength=strength,
            num_inference_steps=self.config.num_inference_steps,
            guidance_scale=self.config.guidance_scale,
            generator=generator,
        )

        if progress_callback:
            progress_callback(4, 5, "Encoding video...")

        from diffusers.utils import export_to_video

        output_dir = os.path.join(ROOT_DIR, ".mp")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"generated_video_{uuid4()}.mp4")

        export_to_video(output.frames[0], output_path, fps=self.config.fps)

        if progress_callback:
            progress_callback(5, 5, "Complete!")

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
        "cogvideox": {
            "name": "CogVideoX",
            "description": "Good quality, works on 8GB+ VRAM",
            "models": ["2b", "5b"],
            "max_frames": 81,
            "fps": 8,
            "min_vram": 8,
        },
        "ltx": {
            "name": "LTX-Video",
            "description": "Best quality, real-time 30 FPS, 161 frames",
            "models": ["base", "distilled", "0.9.7", "0.9.8"],
            "max_frames": 161,
            "fps": 24,
            "min_vram": 10,
        },
    }
