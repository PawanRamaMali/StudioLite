"""Search + cluster orchestration on top of the T2 index.

Kept small - the API router calls these functions instead of touching the
CLIP runtime or the store directly, so future backend swaps (e.g. SigLIP)
happen in one place."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from . import clustering, embeddings as _emb, tagger as _tagger
from .store import LibraryStore, Video

logger = logging.getLogger("studiolite.library.index")


@dataclass
class SearchHit:
    video: Dict[str, Any]
    score: float


def semantic_search(store: LibraryStore, query: str, *,
                    top_k: int = 24,
                    min_score: float = 0.15) -> List[SearchHit]:
    """Rank all indexed videos against a text query using CLIP cosine similarity.

    Raises `EmbeddingUnavailable` when CLIP can't load - the API layer
    turns that into a user-facing "run the embed job first" message."""
    if not query.strip():
        return []
    ids, matrix = store.load_all_embeddings(dim=_emb.EMBED_DIM)
    if matrix is None or matrix.size == 0:
        return []
    q = _emb.encode_text(query.strip())
    if q is None:
        return []
    sims = _emb.cosine_similarity(matrix, q)
    order = np.argsort(-sims)
    hits: List[SearchHit] = []
    for idx in order:
        s = float(sims[idx])
        if s < min_score:
            break
        v = store.get_video(int(ids[idx]))
        if v is None:
            continue
        hits.append(SearchHit(video=v.__dict__, score=s))
        if len(hits) >= top_k:
            break
    return hits


def build_clusters(store: LibraryStore, *,
                   k: Optional[int] = None) -> List[Dict[str, Any]]:
    """Cluster every indexed video, persist the assignments, and return a
    JSON-friendly summary with a preview member per cluster."""
    ids, matrix = store.load_all_embeddings(dim=_emb.EMBED_DIM)
    if matrix is None or matrix.size == 0:
        return []
    labels = clustering.kmeans_cluster(ids, matrix, k=k)
    if not labels:
        return []
    # Persist ids so the UI can browse "just this cluster".
    store.clear_cluster_ids()
    id_to_cluster: Dict[int, int] = {}
    for cl in labels:
        for vid in cl.member_ids:
            id_to_cluster[vid] = cl.id
    store.set_cluster_ids(id_to_cluster)

    out: List[Dict[str, Any]] = []
    for cl in labels:
        preview_id = cl.member_ids[0]
        preview = store.get_video(preview_id)
        out.append({
            "id": cl.id,
            "label": cl.label,
            "size": len(cl.member_ids),
            "preview": (preview.__dict__ if preview else None),
            "member_ids": cl.member_ids,
        })
    return out


def stats(store: LibraryStore) -> Dict[str, Any]:
    """Extended stats - how much of the library is content-indexed."""
    base = store.stats()
    ids, matrix = store.load_all_embeddings(dim=_emb.EMBED_DIM)
    base["embedded"] = 0 if matrix is None else int(matrix.shape[0])
    base["embed_model"] = _emb._MODEL_ID
    base["embed_dim"] = _emb.EMBED_DIM
    return base
