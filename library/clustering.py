"""K-means clustering over CLIP embeddings + auto-labeling.

The number of clusters defaults to sqrt(N) — enough to give the user a
useful bird's-eye view without every cluster being tiny. Each cluster
gets a human label by asking CLIP which of the taxonomy tags best
matches the cluster centroid.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import embeddings as _emb
from . import tagger as _tagger

logger = logging.getLogger("studiolite.library.clustering")


@dataclass
class ClusterLabel:
    id: int
    label: str
    member_ids: List[int]
    centroid: np.ndarray


def _auto_k(n_samples: int) -> int:
    if n_samples < 4:
        return max(1, n_samples)
    return max(2, min(24, int(math.sqrt(n_samples))))


def kmeans_cluster(video_ids: List[int], embeddings: np.ndarray, *,
                   k: Optional[int] = None) -> List[ClusterLabel]:
    """Return one ClusterLabel per cluster. `embeddings` must be L2-normalized
    (all our stored ones are), so k-means on the unit sphere approximates
    spherical k-means well enough."""
    if embeddings.size == 0 or len(video_ids) == 0:
        return []
    n = len(video_ids)
    k_use = k or _auto_k(n)
    k_use = max(1, min(k_use, n))

    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=k_use, n_init=4, random_state=42)
    labels = km.fit_predict(embeddings)
    centroids = km.cluster_centers_

    # Match every centroid against the CLIP tag taxonomy for a human label.
    prompt_matrix = _tagger._ensure_prompts()   # (T, D), L2-normalized
    tag_labels = _tagger._prompt_labels

    out: List[ClusterLabel] = []
    for cid in range(k_use):
        idx = np.where(labels == cid)[0].tolist()
        if not idx:
            continue
        centroid = centroids[cid]
        # Renormalize the centroid for cosine.
        norm = np.linalg.norm(centroid)
        if norm > 1e-8:
            centroid_norm = centroid / norm
        else:
            centroid_norm = centroid
        label = f"Cluster {cid + 1}"
        if prompt_matrix is not None:
            sims = prompt_matrix @ centroid_norm.astype(np.float32)
            best = int(np.argmax(sims))
            label = tag_labels[best]
        out.append(ClusterLabel(
            id=cid + 1,
            label=label,
            member_ids=[video_ids[i] for i in idx],
            centroid=centroid_norm.astype(np.float32),
        ))
    # Largest clusters first for a friendlier browse.
    out.sort(key=lambda c: len(c.member_ids), reverse=True)
    return out
