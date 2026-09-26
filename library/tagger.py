"""Zero-shot content tagging with CLIP.

Instead of loading a second model (BLIP captioning), we reuse the CLIP
text encoder against a curated taxonomy of ~90 tags. Each tag becomes a
sentence prompt; a video's tags are the top matches whose cosine score
crosses a threshold. Cheap because the tag-text embeddings only need to
be computed once and cached at import.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from . import embeddings as _emb

logger = logging.getLogger("studiolite.library.tagger")


# Curated tag taxonomy. Deliberately broad - the goal is to cover
# home-video-library territory without pretending we can name every
# possible concept.
TAG_TAXONOMY: List[str] = [
    # People
    "a person talking", "a group of people", "a child playing", "a baby",
    "a couple", "a family", "a crowd", "a portrait of a face",
    # Places
    "an indoor scene", "an outdoor scene", "a kitchen", "a living room",
    "an office", "a classroom", "a bedroom", "a bathroom", "a garage",
    # Nature / outdoors
    "a beach", "a mountain landscape", "a forest", "a river", "the ocean",
    "a lake", "a sunset", "a sunrise", "snow", "rain", "a garden",
    "flowers", "trees",
    # Cities / infrastructure
    "a city street", "a skyline", "traffic", "a bridge", "an airport",
    "a train station", "a car interior", "a parking lot",
    # Vehicles
    "a car", "a truck", "a motorcycle", "a bicycle", "an airplane",
    "a boat", "a ship",
    # Animals
    "a dog", "a cat", "a bird", "a horse", "a farm animal", "wildlife",
    # Food
    "food on a plate", "a restaurant", "cooking", "a bakery",
    # Events / activities
    "a birthday party", "a wedding", "a concert", "a sporting event",
    "a soccer match", "a basketball game", "a dance performance",
    "a graduation", "a live speaker on stage",
    # Everyday activities
    "reading a book", "using a laptop", "playing a video game",
    "watching television", "shopping",
    # Sports / recreation
    "running", "swimming", "hiking", "cycling", "skiing", "surfing",
    "yoga", "workout at a gym",
    # Weather / lighting
    "night footage", "low-light footage", "bright daylight", "golden hour lighting",
    # Content style
    "a screen recording of a computer desktop", "a video call",
    "a slide presentation", "a tutorial", "a news broadcast",
    "an interview", "a music video", "animated cartoon", "a video game clip",
    "a security camera view", "a drone aerial shot",
    # Meta
    "text on screen", "a blank screen", "very fast motion",
    "a static camera", "handheld camera",
]


@dataclass
class TagResult:
    tag: str
    score: float


_matrix_lock = threading.Lock()
_prompt_matrix: Optional[np.ndarray] = None
_prompt_labels: List[str] = list(TAG_TAXONOMY)


def _ensure_prompts() -> Optional[np.ndarray]:
    """Cache the text embeddings for every prompt in the taxonomy."""
    global _prompt_matrix
    if _prompt_matrix is not None:
        return _prompt_matrix
    with _matrix_lock:
        if _prompt_matrix is not None:
            return _prompt_matrix
        try:
            _prompt_matrix = _emb.encode_texts_batch(_prompt_labels)
            logger.info("tag prompt matrix: %s", _prompt_matrix.shape)
            return _prompt_matrix
        except Exception as e:
            logger.warning("could not build tag prompt matrix (%s); tagging disabled", e)
            return None


def tag_embedding(video_emb: np.ndarray, *,
                  min_score: float = 0.22, top_k: int = 5) -> List[TagResult]:
    """Return the top tags for a single video's embedding.

    CLIP cosine values are usually low (~0.2-0.35 for clear matches with
    base model). Threshold 0.22 keeps noise out while surfacing anything
    the model is actually confident about."""
    prompts = _ensure_prompts()
    if prompts is None or video_emb is None:
        return []
    sims = prompts @ video_emb.astype(np.float32)
    order = np.argsort(-sims)
    out: List[TagResult] = []
    for idx in order[:max(top_k, 1)]:
        s = float(sims[idx])
        if s < min_score:
            break
        out.append(TagResult(tag=_prompt_labels[int(idx)], score=s))
    return out


def tag_matrix(embeddings: np.ndarray, *,
               min_score: float = 0.22, top_k: int = 5) -> List[List[TagResult]]:
    """Same as tag_embedding but vectorized over an (N, D) matrix."""
    prompts = _ensure_prompts()
    if prompts is None or embeddings.size == 0:
        return [[] for _ in range(len(embeddings))]
    sims = embeddings @ prompts.T  # (N, T)
    out: List[List[TagResult]] = []
    for row in sims:
        order = np.argsort(-row)
        tags: List[TagResult] = []
        for idx in order[:top_k]:
            s = float(row[idx])
            if s < min_score:
                break
            tags.append(TagResult(tag=_prompt_labels[int(idx)], score=s))
        out.append(tags)
    return out
