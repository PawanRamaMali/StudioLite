"""
IP-Adapter Integration for StudioLite.

Provides character consistency across multiple generated scenes by injecting
face embeddings and reference image features into the diffusion pipeline.

Supports:
- IP-Adapter Plus: General image feature injection for style/subject consistency
- IP-Adapter FaceID: Face-specific embeddings for character face consistency
- Multi-reference: Combine multiple reference images per character
- Strength control: Adjustable influence (0.0-1.0) per reference

Requires:
- ip_adapter (pip install ip-adapter)
- insightface (pip install insightface)
- onnxruntime-gpu (pip install onnxruntime-gpu)
"""

import os
import gc
import logging
import numpy as np
import torch
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from PIL import Image

logger = logging.getLogger(__name__)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(ROOT_DIR, ".cache", "ip_adapter")
FACE_CACHE_DIR = os.path.join(ROOT_DIR, ".cache", "face_embeddings")


# ---------------------------------------------------------------------------
# Face analysis (InsightFace)
# ---------------------------------------------------------------------------

_face_analyzer = None


def _get_face_analyzer():
    """Lazy-load InsightFace face analyzer."""
    global _face_analyzer
    if _face_analyzer is not None:
        return _face_analyzer

    try:
        from insightface.app import FaceAnalysis

        _face_analyzer = FaceAnalysis(
            name="buffalo_l",
            root=os.path.join(CACHE_DIR, "insightface"),
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        _face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
        logger.info("InsightFace face analyzer loaded successfully")
        return _face_analyzer
    except ImportError:
        logger.warning(
            "insightface not installed. Install with: pip install insightface onnxruntime-gpu"
        )
        return None
    except Exception as e:
        logger.error(f"Failed to load InsightFace: {e}")
        return None


def extract_face_embedding(image_path: str) -> Optional[np.ndarray]:
    """
    Extract face embedding from an image using InsightFace/ArcFace.

    Args:
        image_path: Path to image containing a face

    Returns:
        512-dim face embedding numpy array, or None if no face found
    """
    analyzer = _get_face_analyzer()
    if analyzer is None:
        return None

    img = np.array(Image.open(image_path).convert("RGB"))
    # InsightFace expects BGR
    img_bgr = img[:, :, ::-1]

    faces = analyzer.get(img_bgr)
    if not faces:
        logger.warning(f"No face detected in {image_path}")
        return None

    # Use the largest face (highest detection area)
    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    embedding = face.normed_embedding

    logger.info(f"Extracted face embedding from {image_path} (dim={embedding.shape[0]})")
    return embedding


def extract_face_embeddings_batch(image_paths: List[str]) -> List[np.ndarray]:
    """Extract face embeddings from multiple images. Skips images with no face."""
    embeddings = []
    for path in image_paths:
        emb = extract_face_embedding(path)
        if emb is not None:
            embeddings.append(emb)
    return embeddings


def average_face_embeddings(embeddings: List[np.ndarray]) -> Optional[np.ndarray]:
    """Average multiple face embeddings for a more robust representation."""
    if not embeddings:
        return None
    avg = np.mean(embeddings, axis=0)
    # Re-normalize
    avg = avg / np.linalg.norm(avg)
    return avg


# ---------------------------------------------------------------------------
# Reference image processing
# ---------------------------------------------------------------------------

def prepare_reference_image(
    image_path: str,
    target_size: Tuple[int, int] = (224, 224),
) -> Optional[Image.Image]:
    """
    Load and prepare a reference image for IP-Adapter.

    Args:
        image_path: Path to reference image
        target_size: Target size for the image encoder (CLIP uses 224x224)

    Returns:
        Preprocessed PIL Image
    """
    if not os.path.exists(image_path):
        logger.error(f"Reference image not found: {image_path}")
        return None

    img = Image.open(image_path).convert("RGB")
    img = img.resize(target_size, Image.LANCZOS)
    return img


def extract_last_frame_as_reference(video_path: str, output_path: str = None) -> Optional[str]:
    """
    Extract the last frame from a video to use as reference for the next scene.

    Args:
        video_path: Path to video file
        output_path: Where to save the extracted frame (auto-generated if None)

    Returns:
        Path to the extracted frame image
    """
    try:
        from moviepy.editor import VideoFileClip

        clip = VideoFileClip(video_path)
        # Get frame at 90% of duration (avoids potential encoding artifacts at very end)
        t = clip.duration * 0.9
        frame = clip.get_frame(t)
        clip.close()

        if output_path is None:
            output_path = video_path.replace(".mp4", "_lastframe.png")

        img = Image.fromarray(frame)
        img.save(output_path)
        return output_path
    except Exception as e:
        logger.error(f"Failed to extract last frame from {video_path}: {e}")
        return None


# ---------------------------------------------------------------------------
# IP-Adapter loader & adapter management
# ---------------------------------------------------------------------------

class IPAdapterManager:
    """
    Manages IP-Adapter loading and injection into diffusion pipelines.

    Supports:
    - IP-Adapter Plus (general image features)
    - IP-Adapter FaceID Plus v2 (face-specific features)
    - Combined mode (both face + style reference)
    """

    # Known IP-Adapter model configs
    ADAPTER_CONFIGS = {
        "ip-adapter-plus": {
            "repo": "h94/IP-Adapter",
            "subfolder": "sdxl_models",
            "weight_name": "ip-adapter-plus_sdxl_vit-h.safetensors",
            "image_encoder": "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
            "type": "plus",
        },
        "ip-adapter-faceid-plusv2": {
            "repo": "h94/IP-Adapter-FaceID",
            "weight_name": "ip-adapter-faceid-plusv2_sdxl.bin",
            "image_encoder": "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
            "type": "faceid",
        },
        # For Wan/other diffusers pipelines - use the generic approach
        "ip-adapter-generic": {
            "type": "generic",
        },
    }

    def __init__(self):
        self._loaded_adapter = None
        self._adapter_type = None
        self._image_encoder = None
        self._face_embeddings: Dict[str, np.ndarray] = {}  # char_id -> embedding
        self._reference_images: Dict[str, List[Image.Image]] = {}  # char_id -> images
        self._enabled = False
        self._strength = 0.6  # Default strength

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def strength(self) -> float:
        return self._strength

    @strength.setter
    def strength(self, value: float):
        self._strength = max(0.0, min(1.0, value))

    def check_dependencies(self) -> Dict[str, bool]:
        """Check which IP-Adapter dependencies are available."""
        deps = {}
        try:
            import insightface
            deps["insightface"] = True
        except ImportError:
            deps["insightface"] = False

        try:
            import onnxruntime
            deps["onnxruntime"] = True
        except ImportError:
            deps["onnxruntime"] = False

        try:
            from transformers import CLIPVisionModelWithProjection
            deps["clip_vision"] = True
        except ImportError:
            deps["clip_vision"] = False

        try:
            from diffusers import WanPipeline
            deps["diffusers"] = True
        except ImportError:
            deps["diffusers"] = False

        deps["cuda"] = torch.cuda.is_available()
        return deps

    def register_character(
        self,
        char_id: str,
        reference_images: List[str] = None,
        face_embedding: np.ndarray = None,
    ):
        """
        Register a character with reference images and/or face embedding.

        Args:
            char_id: Unique character identifier
            reference_images: Paths to reference images
            face_embedding: Pre-computed face embedding (512-dim)
        """
        if reference_images:
            processed = []
            for img_path in reference_images:
                img = prepare_reference_image(img_path)
                if img is not None:
                    processed.append(img)
            if processed:
                self._reference_images[char_id] = processed
                logger.info(f"Registered {len(processed)} reference images for character {char_id}")

        if face_embedding is not None:
            self._face_embeddings[char_id] = face_embedding
            logger.info(f"Registered face embedding for character {char_id}")
        elif reference_images:
            # Auto-extract face embedding from reference images
            embeddings = extract_face_embeddings_batch(reference_images)
            if embeddings:
                avg_emb = average_face_embeddings(embeddings)
                if avg_emb is not None:
                    self._face_embeddings[char_id] = avg_emb
                    logger.info(f"Auto-extracted face embedding for character {char_id}")

    def register_from_video_frame(self, char_id: str, video_path: str) -> bool:
        """
        Extract a reference frame from a generated video and register it.
        Used for auto-chaining: first scene's output becomes the reference for subsequent scenes.

        Returns:
            True if registration succeeded
        """
        frame_path = extract_last_frame_as_reference(video_path)
        if frame_path is None:
            return False

        self.register_character(char_id, reference_images=[frame_path])
        return True

    def get_reference_images(self, char_ids: List[str] = None) -> List[Image.Image]:
        """Get all registered reference images, optionally filtered by character IDs."""
        images = []
        ids = char_ids or list(self._reference_images.keys())
        for cid in ids:
            if cid in self._reference_images:
                images.extend(self._reference_images[cid])
        return images

    def get_face_embeddings(self, char_ids: List[str] = None) -> List[np.ndarray]:
        """Get all registered face embeddings, optionally filtered by character IDs."""
        embeddings = []
        ids = char_ids or list(self._face_embeddings.keys())
        for cid in ids:
            if cid in self._face_embeddings:
                embeddings.append(self._face_embeddings[cid])
        return embeddings

    def load_ip_adapter_for_pipeline(self, pipeline, adapter_type: str = "plus"):
        """
        Load IP-Adapter weights into a diffusers pipeline.

        This uses the diffusers built-in IP-Adapter support which works with
        Stable Diffusion, SDXL, and compatible pipelines.

        Args:
            pipeline: A diffusers pipeline instance
            adapter_type: "plus" for general, "faceid" for face-specific
        """
        try:
            if adapter_type == "faceid":
                pipeline.load_ip_adapter(
                    "h94/IP-Adapter-FaceID",
                    subfolder=None,
                    weight_name="ip-adapter-faceid-plusv2_sdxl.bin",
                    cache_dir=CACHE_DIR,
                )
            else:
                pipeline.load_ip_adapter(
                    "h94/IP-Adapter",
                    subfolder="sdxl_models",
                    weight_name="ip-adapter-plus_sdxl_vit-h.safetensors",
                    cache_dir=CACHE_DIR,
                )

            pipeline.set_ip_adapter_scale(self._strength)
            self._loaded_adapter = adapter_type
            self._enabled = True
            logger.info(f"IP-Adapter ({adapter_type}) loaded with strength {self._strength}")
        except Exception as e:
            logger.error(f"Failed to load IP-Adapter: {e}")
            self._enabled = False
            raise

    def inject_into_pipeline_kwargs(
        self,
        pipeline_kwargs: dict,
        char_ids: List[str] = None,
    ) -> dict:
        """
        Add IP-Adapter arguments to pipeline call kwargs.

        Modifies pipeline_kwargs in-place and returns it.
        Adds ip_adapter_image and/or ip_adapter_image_embeds.

        Args:
            pipeline_kwargs: Existing kwargs dict for pipeline.__call__
            char_ids: Character IDs to include (None = all)

        Returns:
            Modified pipeline_kwargs
        """
        if not self._enabled:
            return pipeline_kwargs

        ref_images = self.get_reference_images(char_ids)
        face_embeds = self.get_face_embeddings(char_ids)

        if ref_images:
            # Use the first reference image (or composite)
            pipeline_kwargs["ip_adapter_image"] = ref_images[0] if len(ref_images) == 1 else ref_images
            logger.info(f"Injecting {len(ref_images)} reference image(s) into pipeline")

        if face_embeds and self._loaded_adapter == "faceid":
            # Stack face embeddings as tensor
            face_tensor = torch.tensor(np.stack(face_embeds)).to(dtype=torch.float16)
            pipeline_kwargs["ip_adapter_image_embeds"] = [face_tensor]
            logger.info(f"Injecting {len(face_embeds)} face embedding(s) into pipeline")

        return pipeline_kwargs

    def enable(self, strength: float = 0.6):
        """Enable IP-Adapter with given strength."""
        self._enabled = True
        self.strength = strength

    def disable(self):
        """Disable IP-Adapter (references are kept but not injected)."""
        self._enabled = False

    def clear(self):
        """Clear all registered references and embeddings."""
        self._face_embeddings.clear()
        self._reference_images.clear()
        self._loaded_adapter = None
        self._enabled = False

    def unload(self):
        """Fully unload IP-Adapter and free memory."""
        self.clear()
        self._image_encoder = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_status(self) -> dict:
        """Get current IP-Adapter status."""
        return {
            "enabled": self._enabled,
            "strength": self._strength,
            "loaded_adapter": self._loaded_adapter,
            "registered_characters": list(self._face_embeddings.keys()),
            "reference_image_count": sum(len(v) for v in self._reference_images.values()),
            "face_embedding_count": len(self._face_embeddings),
            "dependencies": self.check_dependencies(),
        }


# ---------------------------------------------------------------------------
# Wan-specific IP-Adapter integration
# ---------------------------------------------------------------------------

class WanIPAdapterHelper:
    """
    Specialized helper for integrating IP-Adapter with Wan 2.1 pipelines.

    Wan pipelines use a different architecture than SD/SDXL, so IP-Adapter
    integration requires a custom approach:
    1. Extract CLIP image features from reference images
    2. Inject them as cross-attention conditioning alongside text embeddings
    3. For face consistency, combine CLIP features with ArcFace embeddings
    """

    def __init__(self, ip_manager: IPAdapterManager):
        self.manager = ip_manager
        self._clip_model = None
        self._clip_processor = None

    def load_clip_encoder(self):
        """Load CLIP vision encoder for image feature extraction."""
        if self._clip_model is not None:
            return

        try:
            from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor

            model_name = "openai/clip-vit-large-patch14"
            self._clip_processor = CLIPImageProcessor.from_pretrained(
                model_name, cache_dir=CACHE_DIR
            )
            self._clip_model = CLIPVisionModelWithProjection.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                cache_dir=CACHE_DIR,
            )
            if torch.cuda.is_available():
                self._clip_model = self._clip_model.to("cuda")
            logger.info("CLIP vision encoder loaded for IP-Adapter")
        except Exception as e:
            logger.error(f"Failed to load CLIP encoder: {e}")
            raise

    def extract_image_features(self, images: List[Image.Image]) -> torch.Tensor:
        """
        Extract CLIP image features from reference images.

        Returns:
            Tensor of shape (N, 768) with image features
        """
        self.load_clip_encoder()

        inputs = self._clip_processor(images=images, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._clip_model(**inputs)

        # Use image_embeds (projected features)
        features = outputs.image_embeds
        return features

    def get_conditioning_for_scene(
        self,
        char_ids: List[str] = None,
        blend_with_text: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """
        Get IP-Adapter conditioning data for a scene generation.

        Returns a dict with:
        - clip_features: CLIP image embeddings
        - face_embeddings: ArcFace embeddings (if available)
        - strength: IP-Adapter strength value

        Returns None if no references are registered.
        """
        if not self.manager.enabled:
            return None

        ref_images = self.manager.get_reference_images(char_ids)
        face_embeds = self.manager.get_face_embeddings(char_ids)

        if not ref_images and not face_embeds:
            return None

        conditioning = {
            "strength": self.manager.strength,
        }

        if ref_images:
            clip_features = self.extract_image_features(ref_images)
            conditioning["clip_features"] = clip_features

        if face_embeds:
            conditioning["face_embeddings"] = torch.tensor(
                np.stack(face_embeds), dtype=torch.float16
            )
            if torch.cuda.is_available():
                conditioning["face_embeddings"] = conditioning["face_embeddings"].to("cuda")

        return conditioning

    def modify_prompt_with_reference(
        self,
        prompt: str,
        conditioning: Dict[str, Any],
    ) -> str:
        """
        Enhance prompt with reference-based descriptors.
        This is a text-level fallback when full IP-Adapter injection isn't available.
        """
        strength = conditioning.get("strength", 0.6)
        if strength > 0.7:
            prompt = f"{prompt}, exactly matching the reference character appearance, same face same features same clothing"
        elif strength > 0.4:
            prompt = f"{prompt}, consistent with reference character, maintaining same visual identity"
        return prompt

    def unload(self):
        """Free CLIP encoder memory."""
        self._clip_model = None
        self._clip_processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# High-level convenience API for Story Mode
# ---------------------------------------------------------------------------

# Global singleton
_ip_manager: Optional[IPAdapterManager] = None
_wan_helper: Optional[WanIPAdapterHelper] = None


def get_ip_adapter_manager() -> IPAdapterManager:
    """Get or create the global IP-Adapter manager."""
    global _ip_manager
    if _ip_manager is None:
        _ip_manager = IPAdapterManager()
    return _ip_manager


def get_wan_helper() -> WanIPAdapterHelper:
    """Get or create the Wan-specific IP-Adapter helper."""
    global _wan_helper
    if _wan_helper is None:
        _wan_helper = WanIPAdapterHelper(get_ip_adapter_manager())
    return _wan_helper


def setup_character_references(
    characters: List[Dict[str, Any]],
    strength: float = 0.6,
) -> IPAdapterManager:
    """
    Set up character references for a story generation session.

    Args:
        characters: List of character dicts with 'id', 'name', 'images' fields
        strength: IP-Adapter strength (0.0-1.0)

    Returns:
        Configured IPAdapterManager
    """
    manager = get_ip_adapter_manager()
    manager.clear()
    manager.enable(strength)

    for char in characters:
        char_id = char.get("id", char.get("name", "default"))
        images = char.get("images", char.get("reference_images", []))
        if images:
            manager.register_character(char_id, reference_images=images)
        else:
            logger.info(f"Character {char_id} has no reference images - will use first scene auto-capture")

    return manager


def auto_capture_reference_from_scene(
    scene_video_path: str,
    char_id: str = "main_subject",
) -> bool:
    """
    Auto-capture a reference from the first generated scene.
    Called after Scene 1 is generated to establish visual reference for remaining scenes.

    Args:
        scene_video_path: Path to the generated scene video
        char_id: Character ID to register the reference under

    Returns:
        True if capture succeeded
    """
    manager = get_ip_adapter_manager()
    return manager.register_from_video_frame(char_id, scene_video_path)


def get_scene_conditioning(
    char_ids: List[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Get IP-Adapter conditioning for the current scene.
    Returns None if IP-Adapter is disabled or no references available.
    """
    helper = get_wan_helper()
    return helper.get_conditioning_for_scene(char_ids)


def enhance_prompt_with_reference(prompt: str, conditioning: Dict[str, Any] = None) -> str:
    """Enhance a generation prompt with reference-based consistency terms."""
    if conditioning is None:
        conditioning = get_scene_conditioning()
    if conditioning is None:
        return prompt

    helper = get_wan_helper()
    return helper.modify_prompt_with_reference(prompt, conditioning)


def cleanup():
    """Clean up all IP-Adapter resources."""
    global _ip_manager, _wan_helper
    if _wan_helper:
        _wan_helper.unload()
        _wan_helper = None
    if _ip_manager:
        _ip_manager.unload()
        _ip_manager = None
