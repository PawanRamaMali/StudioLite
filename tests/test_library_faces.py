"""Face-recognition store/people tests.

No real OpenCV calls — tests exercise:
  - the SQL migration adds persons / face_detections / library_settings
  - faces_enabled defaults to OFF; set_faces_enabled flips it
  - assign_or_cluster: existing centroid wins; new person on no-match
    (with ≥2 unmatched faces), singletons stay unassigned
  - forget_person removes rows + thumbs + optionally stashes centroid
  - wipe_all nukes everything
  - rename_person updates the row
"""
from __future__ import annotations

import os
import sqlite3
import tempfile
import time
from typing import List

_ = time  # make sure time is used (for the ns-stamped video paths)

import numpy as np
import pytest

from library import LibraryStore, people as _people
from library import faces as _faces


def _mkvec(seed: int, dim: int = _faces.EMBED_DIM) -> np.ndarray:
    """Deterministic L2-normalized vector so tests are reproducible."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return (v / np.linalg.norm(v)).astype(np.float32)


def _seed_video_and_faces(store: LibraryStore, faces: List[np.ndarray],
                           thumbs: bool = True) -> List[int]:
    """Insert a stub video row + one face_detection per embedding.
    Returns the new face ids."""
    now = 1_700_000_000.0
    ids: List[int] = []
    with sqlite3.connect(store.db_path) as raw:
        # Unique path per insertion so repeated calls in one test don't
        # trip the videos.abs_path UNIQUE constraint.
        unique_path = f"/v-{time.time_ns()}.mp4"
        raw.execute(
            "INSERT INTO videos(abs_path, rel_path, size_bytes, mtime, added_at, "
            "media_kind, duration_sec, width, height) "
            "VALUES(?,?,1000,?,?,'video',10,640,480)",
            (unique_path, os.path.basename(unique_path), now, now),
        )
        vid = raw.execute("SELECT last_insert_rowid()").fetchone()[0]
        for i, emb in enumerate(faces):
            thumb_path = None
            if thumbs:
                # Use a real temp file so forget_person's unlink has
                # something to delete.
                fh, thumb_path = tempfile.mkstemp(suffix=".jpg", prefix=f"face{i}_")
                os.close(fh)
                with open(thumb_path, "wb") as f: f.write(b"\xff\xd8\xff\xd9fake")
            cur = raw.execute(
                "INSERT INTO face_detections(video_id, t_sec, x, y, w, h, det_score, "
                "embedding, model, thumb_path, created_at) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (vid, float(i), 0.0, 0.0, 100.0, 100.0, 0.95,
                 emb.tobytes(), _faces.EMBED_MODEL, thumb_path, now),
            )
            ids.append(int(cur.lastrowid))
    return ids


def test_migration_creates_face_tables():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        store = LibraryStore(os.path.join(tmp, "lib.sqlite3"))
        with sqlite3.connect(store.db_path) as raw:
            tables = {r[0] for r in raw.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )}
        assert "persons" in tables
        assert "face_detections" in tables
        assert "forgotten_centroids" in tables
        assert "library_settings" in tables


def test_faces_enabled_defaults_off_and_toggle():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        store = LibraryStore(os.path.join(tmp, "lib.sqlite3"))
        assert _people.faces_enabled(store) is False
        _people.set_faces_enabled(store, True)
        assert _people.faces_enabled(store) is True
        _people.set_faces_enabled(store, False)
        assert _people.faces_enabled(store) is False


def test_assign_or_cluster_creates_new_person_on_duplicate_face():
    """Two unmatched faces with the SAME embedding should DBSCAN into one
    new person, and a third different face should stay unassigned."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        store = LibraryStore(os.path.join(tmp, "lib.sqlite3"))
        a1 = _mkvec(1)
        a2 = a1 + 0.001 * _mkvec(99); a2 = a2 / np.linalg.norm(a2)
        b1 = _mkvec(42)
        face_ids = _seed_video_and_faces(store, [a1, a2, b1])
        res = _people.assign_or_cluster(store, face_ids)
        # Two matched into one new person, one leftover singleton.
        assert res["assigned"] == 0, res   # no existing persons to assign to
        assert res["new_persons"] == 1, res
        assert res["unassigned"] == 1, res
        # Confirm persons table has one row with face_count=2.
        with sqlite3.connect(store.db_path) as raw:
            raw.row_factory = sqlite3.Row
            persons = raw.execute("SELECT id, face_count FROM persons").fetchall()
            assert len(persons) == 1
            assert int(persons[0]["face_count"]) == 2


def test_centroid_assigns_matching_face_without_new_person():
    """A face close to an existing centroid should ASSIGN, not spawn new."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        store = LibraryStore(os.path.join(tmp, "lib.sqlite3"))
        a1 = _mkvec(7); a2 = _mkvec(7) + 0.002 * _mkvec(8)
        a2 = a2 / np.linalg.norm(a2)
        a3 = _mkvec(7) + 0.003 * _mkvec(9); a3 = a3 / np.linalg.norm(a3)
        # First pass: a1 + a2 — same seed, should DBSCAN into one person.
        ids = _seed_video_and_faces(store, [a1, a2])
        _people.assign_or_cluster(store, ids)
        # Second pass: a3 — should assign to the SAME person via centroid.
        ids2 = _seed_video_and_faces(store, [a3])
        res = _people.assign_or_cluster(store, ids2)
        assert res["assigned"] == 1, res
        assert res["new_persons"] == 0, res
        with sqlite3.connect(store.db_path) as raw:
            raw.row_factory = sqlite3.Row
            p = raw.execute("SELECT face_count FROM persons").fetchone()
            assert int(p["face_count"]) == 3


def test_list_persons_and_rename():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        store = LibraryStore(os.path.join(tmp, "lib.sqlite3"))
        v = _mkvec(1)
        ids = _seed_video_and_faces(store, [v, v + 0.001])
        _people.assign_or_cluster(store, ids)
        persons = _people.list_persons(store)
        assert len(persons) == 1
        assert persons[0]["name"] is None
        _people.rename_person(store, persons[0]["id"], "Mom")
        persons = _people.list_persons(store)
        assert persons[0]["name"] == "Mom"
        # Blank name resets to NULL.
        _people.rename_person(store, persons[0]["id"], "   ")
        persons = _people.list_persons(store)
        assert persons[0]["name"] is None


def test_forget_person_removes_rows_and_thumbs():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        store = LibraryStore(os.path.join(tmp, "lib.sqlite3"))
        v = _mkvec(2)
        ids = _seed_video_and_faces(store, [v, v + 0.001])
        _people.assign_or_cluster(store, ids)
        persons = _people.list_persons(store)
        pid = persons[0]["id"]
        # Grab thumb paths before deletion.
        with sqlite3.connect(store.db_path) as raw:
            thumbs = [r[0] for r in raw.execute(
                "SELECT thumb_path FROM face_detections WHERE person_id=?",
                (pid,),
            )]
        res = _people.forget_person(store, pid, remember_as_forgotten=True)
        assert res["faces_removed"] == 2
        assert res["thumbs_removed"] == 2
        for t in thumbs:
            assert t is None or not os.path.isfile(t)
        # persons row is gone
        with sqlite3.connect(store.db_path) as raw:
            raw.row_factory = sqlite3.Row
            assert raw.execute("SELECT COUNT(*) FROM persons").fetchone()[0] == 0
            fc = raw.execute("SELECT COUNT(*) FROM forgotten_centroids").fetchone()[0]
            assert fc == 1, "remember_as_forgotten should stash a centroid"


def test_wipe_all_clears_everything():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        store = LibraryStore(os.path.join(tmp, "lib.sqlite3"))
        v1 = _mkvec(1); v2 = _mkvec(42)
        ids = _seed_video_and_faces(store, [v1, v1 + 0.001, v2])
        _people.assign_or_cluster(store, ids)
        res = _people.wipe_all(store)
        assert res["faces_removed"] == 3
        with sqlite3.connect(store.db_path) as raw:
            raw.row_factory = sqlite3.Row
            assert raw.execute("SELECT COUNT(*) FROM persons").fetchone()[0] == 0
            assert raw.execute("SELECT COUNT(*) FROM face_detections").fetchone()[0] == 0
            # faces_indexed_at cleared too so a re-run re-scans.
            cnt = raw.execute(
                "SELECT COUNT(*) FROM videos WHERE faces_indexed_at IS NOT NULL"
            ).fetchone()[0]
            assert cnt == 0


def test_running_mean_add_stays_unit_length():
    v1 = _mkvec(1); v2 = _mkvec(2)
    cent, cnt = _faces.running_mean_add(None, 0, v1)
    assert cnt == 1
    assert abs(float(np.linalg.norm(cent)) - 1.0) < 1e-5
    cent2, cnt2 = _faces.running_mean_add(cent, cnt, v2)
    assert cnt2 == 2
    assert abs(float(np.linalg.norm(cent2)) - 1.0) < 1e-5
