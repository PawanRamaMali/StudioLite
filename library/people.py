"""Person clustering + "forget" + export for the Library.

Hybrid strategy per the design doc:
  1. For every new face, try to assign to the nearest existing `persons.centroid`
     with cosine ≥ SFACE_COSINE_SAME_PERSON. If assigned, update the running-mean
     centroid incrementally.
  2. Faces that don't match any person stay in the "unassigned" pool and get
     clustered with DBSCAN (cosine distance) once per index-job, discovering
     new identities among them.
  3. A full "recluster" is user-triggered: wipe person_id on every face, re-run
     the full pipeline from scratch, preserve names by majority-vote mapping
     old→new ids.

Nothing in this module writes to disk itself — it only owns the SQL + the
cosine math. `library.jobs.FaceIndexJob` is the thing that calls
`assign_or_cluster()`.
"""
from __future__ import annotations

import logging
import os
import sqlite3
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from .faces import SFACE_COSINE_SAME_PERSON, EMBED_DIM, running_mean_add
from .store import LibraryStore

logger = logging.getLogger("studiolite.library.people")


# DBSCAN hyperparams for discovering NEW people inside the unassigned pool.
# eps is cosine distance (1 - cos); min_samples 2 is loose enough to catch
# twice-appearing faces in a short library and tight enough to leave
# singletons out of the person list.
_DBSCAN_EPS = 1.0 - SFACE_COSINE_SAME_PERSON   # 0.40
_DBSCAN_MIN = 2


def _blob_to_vec(blob: Optional[bytes]) -> Optional[np.ndarray]:
    if not blob: return None
    try:
        arr = np.frombuffer(blob, dtype=np.float32)
        if arr.size != EMBED_DIM:
            return None
        return arr.copy()
    except Exception:
        return None


def _vec_to_blob(v: np.ndarray) -> bytes:
    return np.asarray(v, dtype=np.float32).tobytes()


def _load_persons(c: sqlite3.Connection) -> List[Dict]:
    rows = c.execute(
        "SELECT id, centroid, face_count FROM persons WHERE hidden=0"
    ).fetchall()
    return [{
        "id": int(r["id"]),
        "centroid": _blob_to_vec(r["centroid"]),
        "face_count": int(r["face_count"] or 0),
    } for r in rows]


def assign_or_cluster(store: LibraryStore, new_face_ids: List[int]) -> Dict[str, int]:
    """Mutate `face_detections.person_id` + `persons.*` for the given face ids.

    1. Walk each new face in order, assign to the closest existing centroid
       within the SFace threshold; update that person's running mean.
    2. Collect the ids that didn't match anybody. If there are ≥ _DBSCAN_MIN,
       run DBSCAN and create a new person per cluster.
    3. Singletons stay unassigned (person_id NULL) — they show up in the UI
       as "ungrouped faces" and the next scan may absorb them when a second
       instance of the same face arrives.

    Returns {assigned, new_persons, unassigned}.
    """
    if not store._ok or not new_face_ids:
        return {"assigned": 0, "new_persons": 0, "unassigned": 0}

    import time as _time
    with store._lock:
        with store._connect() as c:
            # Load incoming faces + their embeddings.
            placeholders = ",".join("?" for _ in new_face_ids)
            rows = c.execute(
                f"SELECT id, embedding FROM face_detections WHERE id IN ({placeholders})",
                tuple(new_face_ids),
            ).fetchall()
            face_vecs: List[Tuple[int, np.ndarray]] = []
            for r in rows:
                v = _blob_to_vec(r["embedding"])
                if v is not None:
                    face_vecs.append((int(r["id"]), v))
            if not face_vecs:
                return {"assigned": 0, "new_persons": 0, "unassigned": 0}

            persons = _load_persons(c)
            persons_by_id = {p["id"]: p for p in persons}

            assigned = 0
            unmatched: List[Tuple[int, np.ndarray]] = []

            # --- Step 1: centroid-assign pass ---
            for face_id, emb in face_vecs:
                best_id, best_score = None, -1.0
                for p in persons:
                    cent = p["centroid"]
                    if cent is None: continue
                    s = float(np.dot(cent, emb))
                    if s > best_score:
                        best_score = s; best_id = p["id"]
                if best_id is not None and best_score >= SFACE_COSINE_SAME_PERSON:
                    p = persons_by_id[best_id]
                    new_cent, new_count = running_mean_add(
                        p["centroid"], p["face_count"], emb,
                    )
                    p["centroid"], p["face_count"] = new_cent, new_count
                    c.execute(
                        "UPDATE face_detections SET person_id=? WHERE id=?",
                        (best_id, face_id),
                    )
                    c.execute(
                        "UPDATE persons SET centroid=?, face_count=?, updated_at=? WHERE id=?",
                        (_vec_to_blob(new_cent), new_count, _time.time(), best_id),
                    )
                    assigned += 1
                else:
                    unmatched.append((face_id, emb))

            # --- Step 2: DBSCAN the leftovers ---
            new_persons = 0
            if len(unmatched) >= _DBSCAN_MIN:
                try:
                    from sklearn.cluster import DBSCAN
                    X = np.stack([v for _, v in unmatched])
                    # L2-normalized -> cosine distance == 1 - dot product.
                    labels = DBSCAN(eps=_DBSCAN_EPS, min_samples=_DBSCAN_MIN,
                                    metric="cosine").fit_predict(X)
                    # Build each cluster's running mean + create a person.
                    by_label: Dict[int, List[int]] = {}
                    for idx, lb in enumerate(labels):
                        if lb < 0: continue
                        by_label.setdefault(int(lb), []).append(idx)
                    for lb, idxs in by_label.items():
                        rows_cluster = [unmatched[i] for i in idxs]
                        cent, cnt = None, 0
                        for _fid, emb in rows_cluster:
                            cent, cnt = running_mean_add(cent, cnt, emb)
                        cover_face_id = rows_cluster[0][0]
                        cur = c.execute(
                            "INSERT INTO persons(centroid, face_count, cover_face_id, "
                            "created_at, updated_at) VALUES(?,?,?,?,?)",
                            (_vec_to_blob(cent), cnt, cover_face_id,
                             _time.time(), _time.time()),
                        )
                        pid = int(cur.lastrowid or 0)
                        if pid:
                            c.executemany(
                                "UPDATE face_detections SET person_id=? WHERE id=?",
                                [(pid, fid) for fid, _ in rows_cluster],
                            )
                            new_persons += 1
                except ImportError:
                    logger.warning("sklearn missing; cluster step skipped")
                except Exception as e:
                    logger.warning("DBSCAN step failed: %s", e)

            unassigned_now = len(unmatched) - (new_persons and 0)  # best-effort
            # Recompute real unassigned count by querying.
            unassigned_now = int(c.execute(
                "SELECT COUNT(*) AS n FROM face_detections WHERE person_id IS NULL "
                "AND id IN (" + placeholders + ")", tuple(new_face_ids),
            ).fetchone()["n"])
            return {"assigned": assigned, "new_persons": new_persons,
                    "unassigned": unassigned_now}


def recluster_all(store: LibraryStore) -> Dict[str, int]:
    """Wipe person_id across every face and rebuild the full taxonomy from
    scratch. Preserves `name` by majority-vote mapping old→new cluster ids.
    Returns {persons_before, persons_after, faces, kept_names}."""
    if not store._ok:
        return {"persons_before": 0, "persons_after": 0, "faces": 0, "kept_names": 0}

    import time as _time
    with store._lock:
        with store._connect() as c:
            # Snapshot old mapping + names BEFORE we touch anything.
            old_map: Dict[int, int] = {}
            for r in c.execute("SELECT id, person_id FROM face_detections "
                               "WHERE person_id IS NOT NULL"):
                old_map[int(r["id"])] = int(r["person_id"])
            old_names: Dict[int, str] = {}
            persons_before = 0
            for r in c.execute("SELECT id, name FROM persons"):
                persons_before += 1
                if r["name"]:
                    old_names[int(r["id"])] = str(r["name"])

            # Wipe. Keep face rows + their embeddings.
            c.execute("UPDATE face_detections SET person_id=NULL")
            c.execute("DELETE FROM persons")

            # Full DBSCAN run — pull every face embedding.
            face_rows = c.execute(
                "SELECT id, embedding FROM face_detections"
            ).fetchall()
            face_ids: List[int] = []
            vecs: List[np.ndarray] = []
            for r in face_rows:
                v = _blob_to_vec(r["embedding"])
                if v is None: continue
                face_ids.append(int(r["id"]))
                vecs.append(v)
            if len(vecs) < _DBSCAN_MIN:
                return {"persons_before": persons_before, "persons_after": 0,
                        "faces": len(vecs), "kept_names": 0}

            try:
                from sklearn.cluster import DBSCAN
                X = np.stack(vecs)
                labels = DBSCAN(eps=_DBSCAN_EPS, min_samples=_DBSCAN_MIN,
                                metric="cosine").fit_predict(X)
            except Exception as e:
                logger.warning("recluster_all failed: %s", e)
                return {"persons_before": persons_before, "persons_after": 0,
                        "faces": len(vecs), "kept_names": 0}

            by_label: Dict[int, List[int]] = {}
            for i, lb in enumerate(labels):
                if lb < 0: continue
                by_label.setdefault(int(lb), []).append(i)

            # Create a person per cluster. Give it the name from the
            # majority of its member faces' PREVIOUS person_id if any.
            kept_names = 0
            persons_after = 0
            for lb, idxs in by_label.items():
                cent, cnt = None, 0
                member_fids: List[int] = []
                vote: Dict[int, int] = {}  # old_person_id -> count
                for i in idxs:
                    emb = vecs[i]
                    cent, cnt = running_mean_add(cent, cnt, emb)
                    fid = face_ids[i]
                    member_fids.append(fid)
                    old_pid = old_map.get(fid)
                    if old_pid is not None:
                        vote[old_pid] = vote.get(old_pid, 0) + 1
                picked_name = None
                if vote:
                    winner_old = max(vote, key=lambda k: vote[k])
                    picked_name = old_names.get(winner_old)
                cur = c.execute(
                    "INSERT INTO persons(name, centroid, face_count, cover_face_id, "
                    "created_at, updated_at) VALUES(?,?,?,?,?,?)",
                    (picked_name, _vec_to_blob(cent), cnt, member_fids[0],
                     _time.time(), _time.time()),
                )
                pid = int(cur.lastrowid or 0)
                c.executemany(
                    "UPDATE face_detections SET person_id=? WHERE id=?",
                    [(pid, fid) for fid in member_fids],
                )
                persons_after += 1
                if picked_name: kept_names += 1
            return {"persons_before": persons_before, "persons_after": persons_after,
                    "faces": len(vecs), "kept_names": kept_names}


# ---------------------------------------------------------------------------
# Person CRUD-ish
# ---------------------------------------------------------------------------

def list_persons(store: LibraryStore) -> List[Dict]:
    if not store._ok: return []
    with store._connect() as c:
        rows = c.execute(
            "SELECT p.id, p.name, p.face_count, p.cover_face_id, p.created_at, "
            "p.updated_at, p.hidden, f.thumb_path AS cover_thumb "
            "FROM persons p LEFT JOIN face_detections f ON f.id = p.cover_face_id "
            "WHERE p.hidden=0 ORDER BY p.face_count DESC"
        ).fetchall()
        return [{
            "id": int(r["id"]), "name": r["name"] or None,
            "face_count": int(r["face_count"] or 0),
            "cover_face_id": r["cover_face_id"],
            "cover_thumb": r["cover_thumb"],
            "created_at": r["created_at"], "updated_at": r["updated_at"],
        } for r in rows]


def person_detail(store: LibraryStore, person_id: int, *,
                  limit_faces: int = 60) -> Optional[Dict]:
    if not store._ok: return None
    with store._connect() as c:
        p = c.execute("SELECT * FROM persons WHERE id=? AND hidden=0",
                      (person_id,)).fetchone()
        if not p: return None
        # Face rows + the video they came from.
        rows = c.execute(
            "SELECT f.id, f.video_id, f.t_sec, f.x, f.y, f.w, f.h, f.det_score, "
            "f.thumb_path, v.abs_path, v.duration_sec "
            "FROM face_detections f JOIN videos v ON v.id = f.video_id "
            "WHERE f.person_id=? ORDER BY v.added_at DESC, f.t_sec ASC "
            "LIMIT ?", (person_id, limit_faces),
        ).fetchall()
        return {
            "id": int(p["id"]),
            "name": p["name"], "face_count": int(p["face_count"] or 0),
            "cover_face_id": p["cover_face_id"],
            "created_at": p["created_at"], "updated_at": p["updated_at"],
            "faces": [dict(r) for r in rows],
        }


def rename_person(store: LibraryStore, person_id: int, name: Optional[str]) -> bool:
    if not store._ok: return False
    name = (name or "").strip() or None
    with store._lock:
        with store._connect() as c:
            cur = c.execute(
                "UPDATE persons SET name=?, updated_at=? WHERE id=?",
                (name, time.time(), person_id),
            )
            return cur.rowcount > 0


def forget_person(store: LibraryStore, person_id: int, *,
                  remember_as_forgotten: bool = False) -> Dict[str, int]:
    """Hard-delete every face + thumb for this person, drop the row.
    Optionally record the person's centroid in `forgotten_centroids` so a
    future scan skips re-detecting the same identity."""
    if not store._ok: return {"faces_removed": 0, "thumbs_removed": 0}
    import time as _time
    with store._lock:
        with store._connect() as c:
            p = c.execute(
                "SELECT centroid FROM persons WHERE id=?", (person_id,),
            ).fetchone()
            # Collect thumb paths so we can unlink the files.
            thumbs = [r["thumb_path"] for r in c.execute(
                "SELECT thumb_path FROM face_detections WHERE person_id=?",
                (person_id,),
            ).fetchall() if r["thumb_path"]]
            # Face count before we delete.
            faces_removed = int(c.execute(
                "SELECT COUNT(*) AS n FROM face_detections WHERE person_id=?",
                (person_id,),
            ).fetchone()["n"])
            c.execute("DELETE FROM face_detections WHERE person_id=?", (person_id,))
            c.execute("DELETE FROM persons WHERE id=?", (person_id,))
            if remember_as_forgotten and p and p["centroid"]:
                c.execute(
                    "INSERT INTO forgotten_centroids(centroid, reason, created_at) "
                    "VALUES(?,?,?)",
                    (p["centroid"], f"user forget person {person_id}", _time.time()),
                )
        # Unlink thumbs outside the transaction so OS errors don't abort the
        # row delete.
        thumbs_removed = 0
        for t in thumbs:
            if t and os.path.isfile(t):
                try: os.remove(t); thumbs_removed += 1
                except OSError: pass
    return {"faces_removed": faces_removed, "thumbs_removed": thumbs_removed}


def wipe_all(store: LibraryStore) -> Dict[str, int]:
    """Nuke every face + every person + every thumb — the privacy "wipe all"
    button. Does NOT touch `forgotten_centroids` because that's the opt-in
    recovery barrier."""
    if not store._ok: return {"faces_removed": 0, "thumbs_removed": 0, "persons_removed": 0}
    with store._lock:
        with store._connect() as c:
            thumbs = [r["thumb_path"] for r in c.execute(
                "SELECT thumb_path FROM face_detections"
            ).fetchall() if r["thumb_path"]]
            faces_removed = int(c.execute("SELECT COUNT(*) AS n FROM face_detections"
                                          ).fetchone()["n"])
            persons_removed = int(c.execute("SELECT COUNT(*) AS n FROM persons"
                                            ).fetchone()["n"])
            c.execute("DELETE FROM face_detections")
            c.execute("DELETE FROM persons")
            c.execute("UPDATE videos SET faces_indexed_at=NULL, faces_model=NULL")
        thumbs_removed = 0
        for t in thumbs:
            if t and os.path.isfile(t):
                try: os.remove(t); thumbs_removed += 1
                except OSError: pass
    return {"faces_removed": faces_removed, "thumbs_removed": thumbs_removed,
            "persons_removed": persons_removed}


# ---------------------------------------------------------------------------
# faces_enabled toggle (lives in library_settings)
# ---------------------------------------------------------------------------

_FACES_ENABLED_KEY = "faces_enabled"


def faces_enabled(store: LibraryStore) -> bool:
    """Default: OFF. Users have to opt in explicitly."""
    if not store._ok: return False
    try:
        with store._connect() as c:
            r = c.execute(
                "SELECT value FROM library_settings WHERE key=?",
                (_FACES_ENABLED_KEY,),
            ).fetchone()
    except Exception:
        return False
    if r is None: return False
    return str(r["value"]).lower() in ("1", "true", "yes", "on")


def set_faces_enabled(store: LibraryStore, enabled: bool) -> None:
    if not store._ok: return
    with store._lock:
        with store._connect() as c:
            c.execute(
                "INSERT INTO library_settings(key, value) VALUES(?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (_FACES_ENABLED_KEY, "1" if enabled else "0"),
            )
