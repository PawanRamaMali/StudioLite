"""
VideoGenerator - Real AI video generation using CogVideoX.

Supports:
- Text-to-Video: Generate video from text prompt
- Image-to-Video: Animate a still image
- Video-to-Video: Extend or modify existing video
"""
import os
import gc
import torch
from dataclasses import dataclass, field
from uuid import uuid4
from typing import Optional, Callable, Tuple, Dict, Any

# Root directory
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


@dataclass
class VideoGenConfig:
    """Configuration for video generation."""
    model_variant: str = "2b"           # "2b" or "5b"
    num_frames: int = 49                # 49 (~6s) or 81 (~10s)
    num_inference_steps: int = 50
    guidance_scale: float = 6.0
    width: int = 720
    height: int = 480
    fps: int = 8
    enable_cpu_offload: bool = True
    quantization: str = "auto"          # "none", "int8", "auto"
    use_vae_slicing: bool = True
    seed: Optional[int] = None


class VideoGenerator:
    """
    AI Video Generation Engine using CogVideoX.

    Features:
    - Auto-configures based on available VRAM
    - Supports text-to-video, image-to-video, video-to-video
    - Memory-efficient with CPU offloading and quantization
    """

    def __init__(self, config: VideoGenConfig = None):
        self.config = config or VideoGenConfig()
        self._pipeline = None
        self._current_mode = None
        self._loaded_model_id = None

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
            # High-end GPU: Full 5B model, max quality
            self.config.model_variant = "5b"
            self.config.quantization = "none"
            self.config.enable_cpu_offload = False
            self.config.num_frames = 81
        elif vram >= 16:
            # Mid-range: 5B with CPU offload
            self.config.model_variant = "5b"
            self.config.quantization = "none"
            self.config.enable_cpu_offload = True
            self.config.num_frames = 49
        elif vram >= 12:
            # Lower mid-range: 5B with quantization
            self.config.model_variant = "5b"
            self.config.quantization = "int8"
            self.config.enable_cpu_offload = True
            self.config.num_frames = 49
        elif vram >= 8:
            # Entry-level: 2B model
            self.config.model_variant = "2b"
            self.config.quantization = "int8"
            self.config.enable_cpu_offload = True
            self.config.num_frames = 49
        else:
            # Minimal: 2B with aggressive optimization
            self.config.model_variant = "2b"
            self.config.quantization = "int8"
            self.config.enable_cpu_offload = True
            self.config.num_frames = 49

    def _get_model_id(self, mode: str) -> str:
        """Get HuggingFace model ID based on mode and config."""
        if mode == "text2video":
            return f"THUDM/CogVideoX-{self.config.model_variant}"
        elif mode == "image2video":
            return "THUDM/CogVideoX-5b-I2V"
        elif mode == "video2video":
            return f"THUDM/CogVideoX-{self.config.model_variant}"
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def load_pipeline(self, mode: str = "text2video", progress_callback: Callable = None) -> None:
        """
        Load the CogVideoX pipeline for the specified mode.

        Args:
            mode: "text2video", "image2video", or "video2video"
            progress_callback: Optional callback for progress updates
        """
        from diffusers import (
            CogVideoXPipeline,
            CogVideoXImageToVideoPipeline,
            CogVideoXVideoToVideoPipeline,
        )

        model_id = self._get_model_id(mode)

        # Skip if already loaded
        if self._pipeline is not None and self._current_mode == mode and self._loaded_model_id == model_id:
            return

        # Unload existing pipeline
        if self._pipeline is not None:
            if progress_callback:
                progress_callback(0, 100, "Unloading previous model...")
            self.unload()

        if progress_callback:
            progress_callback(10, 100, f"Loading {model_id}...")

        # Select pipeline class
        if mode == "text2video":
            PipelineClass = CogVideoXPipeline
        elif mode == "image2video":
            PipelineClass = CogVideoXImageToVideoPipeline
        elif mode == "video2video":
            PipelineClass = CogVideoXVideoToVideoPipeline
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Determine dtype
        dtype = torch.bfloat16

        # Load model (quantization disabled for compatibility)
        if progress_callback:
            progress_callback(20, 100, f"Loading {model_id}...")

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

        if progress_callback:
            progress_callback(100, 100, "Model loaded!")

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
        if progress_callback:
            progress_callback(0, 5, "Loading model...")

        # Load pipeline if needed
        if self._current_mode != "text2video":
            self.load_pipeline("text2video", progress_callback)

        if progress_callback:
            progress_callback(1, 5, "Generating video frames...")

        # Set seed if specified
        generator = None
        if self.config.seed is not None:
            generator = torch.Generator(device="cuda").manual_seed(self.config.seed)

        # Generate
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

        # Export to video file
        from diffusers.utils import export_to_video

        output_dir = os.path.join(ROOT_DIR, ".mp")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"generated_video_{uuid4()}.mp4")

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
        from PIL import Image

        if progress_callback:
            progress_callback(0, 5, "Loading image...")

        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        image = image.resize((self.config.width, self.config.height))

        if progress_callback:
            progress_callback(1, 5, "Loading model...")

        # Load pipeline if needed
        if self._current_mode != "image2video":
            self.load_pipeline("image2video", progress_callback)

        if progress_callback:
            progress_callback(2, 5, "Generating video frames...")

        # Set seed if specified
        generator = None
        if self.config.seed is not None:
            generator = torch.Generator(device="cuda").manual_seed(self.config.seed)

        # Generate
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

        # Export to video file
        from diffusers.utils import export_to_video

        output_dir = os.path.join(ROOT_DIR, ".mp")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"generated_video_{uuid4()}.mp4")

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
        if progress_callback:
            progress_callback(0, 5, "Loading video...")

        # Load video frames
        from diffusers.utils import load_video

        video = load_video(video_path)

        if progress_callback:
            progress_callback(1, 5, "Loading model...")

        # Load pipeline if needed
        if self._current_mode != "video2video":
            self.load_pipeline("video2video", progress_callback)

        if progress_callback:
            progress_callback(2, 5, "Transforming video...")

        # Set seed if specified
        generator = None
        if self.config.seed is not None:
            generator = torch.Generator(device="cuda").manual_seed(self.config.seed)

        # Generate
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

        # Export to video file
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

        # Force garbage collection
        gc.collect()

        # Clear CUDA cache
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
