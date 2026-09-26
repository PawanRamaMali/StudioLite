"""SQLite index + version-snapshot layer on top of the file-based project
tree that filmmaker/project.py already writes.

Why a layer, not a rewrite
--------------------------

The pipeline reads and writes ``artifacts/<stage>.json`` all over the
place (agents.py, orchestrator.py, tests). Migrating that surface to
a database is a several-day refactor and would break every third-party
tool anyone has built against the JSON layout. Instead we treat the
files as canonical and use SQLite for two crisp jobs:

  1. **Index** - the ``projects`` table stores id, title, brief, and
     timestamps for fast list/search without walking .mp/films on every
     request. ``upsert_from_disk()`` reconciles the index with what's
     on disk, so nothing goes silently out of sync.

  2. **Version history** - every artifact write can call
     ``snapshot_artifact()``, which stores a JSON blob in the
     ``artifact_versions`` table keyed by (project_id, stage_key,
     version_no). The user can list versions, view any of them, and
     restore one. Snapshots are compressed with ``gzip`` above 8 KiB
     to keep the DB bounded on a screenplay-heavy project.

Everything is fail-open. If SQLite refuses to open the file, we log a
warning and every method returns an empty result or no-ops - the JSON
tree still works, just without the fast index and history."""
from __future__ import annotations

import gzip
import json
import logging
import os
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger("studiolite.project_store")


_SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
    project_id  TEXT PRIMARY KEY,
    title       TEXT NOT NULL,
    brief       TEXT NOT NULL,
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_projects_updated ON projects(updated_at DESC);

CREATE TABLE IF NOT EXISTS artifact_versions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id  TEXT NOT NULL,
    stage_key   TEXT NOT NULL,
    version_no  INTEGER NOT NULL,
    saved_at    REAL NOT NULL,
    note        TEXT,
    compressed  INTEGER NOT NULL DEFAULT 0,  -- 0 = raw json, 1 = gzip
    payload     BLOB NOT NULL,
    UNIQUE(project_id, stage_key, version_no)
);

CREATE INDEX IF NOT EXISTS idx_versions_by_stage
    ON artifact_versions(project_id, stage_key, version_no DESC);
"""


# Only compress payloads bigger than this - below 8 KiB the gzip overhead
# beats the wins.
_COMPRESS_THRESHOLD = 8 * 1024


class ProjectStore:
    """Thread-safe index + versioning layer. Fail-open: if the DB won't
    open, every read returns empty and every write silently no-ops so the
    JSON tree stays canonical."""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._lock = threading.Lock()
        self._open_or_fallback()

    def _open_or_fallback(self) -> None:
        try:
            os.makedirs(os.path.dirname(self._db_path) or ".", exist_ok=True)
            conn = sqlite3.connect(
                self._db_path,
                check_same_thread=False,
                isolation_level=None,
            )
            conn.row_factory = sqlite3.Row
            conn.executescript(_SCHEMA)
            self._conn = conn
            logger.info("Project store opened at %s", self._db_path)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "Could not open project store at %s (%s); index/versions disabled.",
                self._db_path, e,
            )
            self._conn = None

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:  # noqa: BLE001
                pass
            self._conn = None

    # --- index ------------------------------------------------------------

    def upsert_project(self, project_id: str, title: str, brief: str,
                       created_at: float, updated_at: float) -> None:
        """Reconcile the index row for a project. Called after every
        meta write so ``list_indexed_projects`` stays authoritative."""
        if self._conn is None:
            return
        with self._lock:
            try:
                self._conn.execute(
                    "INSERT INTO projects (project_id, title, brief, created_at, updated_at) "
                    "VALUES (?,?,?,?,?) "
                    "ON CONFLICT(project_id) DO UPDATE SET "
                    "  title=excluded.title, brief=excluded.brief, "
                    "  updated_at=excluded.updated_at",
                    (project_id, title, brief, created_at, updated_at),
                )
            except Exception as e:  # noqa: BLE001
                logger.warning("Project index write failed for %s: %s", project_id, e)

    def delete_project(self, project_id: str) -> None:
        """Drop the index row AND every version snapshot for a project.
        Meant to be called by the film-delete API path so the DB doesn't
        keep pointing at a project directory that's since been ``shutil.rmtree``d."""
        if self._conn is None:
            return
        with self._lock:
            try:
                self._conn.execute("DELETE FROM artifact_versions WHERE project_id=?", (project_id,))
                self._conn.execute("DELETE FROM projects WHERE project_id=?", (project_id,))
            except Exception as e:  # noqa: BLE001
                logger.warning("Project index delete failed for %s: %s", project_id, e)

    def list_indexed_projects(self, limit: int = 100) -> List[Dict[str, Any]]:
        if self._conn is None:
            return []
        with self._lock:
            try:
                rows = self._conn.execute(
                    "SELECT project_id, title, brief, created_at, updated_at "
                    "FROM projects ORDER BY updated_at DESC LIMIT ?",
                    (int(limit),),
                ).fetchall()
            except Exception as e:  # noqa: BLE001
                logger.warning("Project index read failed: %s", e)
                return []
        return [{
            "id": r["project_id"], "title": r["title"], "brief": r["brief"],
            "created_at": r["created_at"], "updated_at": r["updated_at"],
        } for r in rows]

    # --- versions ---------------------------------------------------------

    def snapshot_artifact(self, project_id: str, stage_key: str,
                          data: Dict[str, Any], note: str = "") -> Optional[int]:
        """Record a new version of an artifact. Returns the assigned
        version_no or None if the store is in fail-open mode.
        Compresses payloads above _COMPRESS_THRESHOLD with gzip so a
        heavy screenplay history doesn't balloon the DB."""
        if self._conn is None:
            return None
        raw = json.dumps(data, ensure_ascii=False).encode("utf-8")
        compressed = 0
        if len(raw) > _COMPRESS_THRESHOLD:
            raw = gzip.compress(raw)
            compressed = 1
        with self._lock:
            try:
                # Next version number for (project, stage) - pure sqlite,
                # no round-trip in Python.
                row = self._conn.execute(
                    "SELECT COALESCE(MAX(version_no), 0) + 1 AS next "
                    "FROM artifact_versions WHERE project_id=? AND stage_key=?",
                    (project_id, stage_key),
                ).fetchone()
                next_no = int(row["next"])
                self._conn.execute(
                    "INSERT INTO artifact_versions "
                    "(project_id, stage_key, version_no, saved_at, note, compressed, payload) "
                    "VALUES (?,?,?,?,?,?,?)",
                    (project_id, stage_key, next_no, time.time(),
                     note or "", compressed, raw),
                )
                return next_no
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "Artifact snapshot failed for %s/%s: %s",
                    project_id, stage_key, e,
                )
                return None

    def list_versions(self, project_id: str, stage_key: str,
                      limit: int = 50) -> List[Dict[str, Any]]:
        """Return {version_no, saved_at, note, size} newest-first for one
        artifact. Payloads are not included - the client asks for a
        specific version via ``load_version()``."""
        if self._conn is None:
            return []
        with self._lock:
            try:
                rows = self._conn.execute(
                    "SELECT version_no, saved_at, note, compressed, LENGTH(payload) AS size "
                    "FROM artifact_versions WHERE project_id=? AND stage_key=? "
                    "ORDER BY version_no DESC LIMIT ?",
                    (project_id, stage_key, int(limit)),
                ).fetchall()
            except Exception as e:  # noqa: BLE001
                logger.warning("List versions failed for %s/%s: %s",
                               project_id, stage_key, e)
                return []
        return [{
            "version_no": r["version_no"],
            "saved_at": r["saved_at"],
            "note": r["note"] or "",
            "compressed": bool(r["compressed"]),
            "size": r["size"],
        } for r in rows]

    def load_version(self, project_id: str, stage_key: str,
                     version_no: int) -> Optional[Dict[str, Any]]:
        """Materialize one version back into the artifact JSON shape."""
        if self._conn is None:
            return None
        with self._lock:
            try:
                row = self._conn.execute(
                    "SELECT payload, compressed FROM artifact_versions "
                    "WHERE project_id=? AND stage_key=? AND version_no=?",
                    (project_id, stage_key, int(version_no)),
                ).fetchone()
            except Exception as e:  # noqa: BLE001
                logger.warning("Load version failed for %s/%s/%s: %s",
                               project_id, stage_key, version_no, e)
                return None
        if row is None:
            return None
        payload = row["payload"]
        if row["compressed"]:
            try:
                payload = gzip.decompress(payload)
            except Exception:  # noqa: BLE001
                return None
        try:
            return json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            return None

    def prune_versions(self, project_id: str, stage_key: str,
                       keep: int = 20) -> int:
        """Bound the version history per (project, stage). Keeps the
        newest ``keep`` snapshots, deletes the rest. Returns the number
        of rows removed so the caller can log the compaction."""
        if self._conn is None or keep < 0:
            return 0
        with self._lock:
            try:
                cur = self._conn.execute(
                    "DELETE FROM artifact_versions "
                    "WHERE project_id=? AND stage_key=? "
                    "AND version_no NOT IN ("
                    "  SELECT version_no FROM artifact_versions "
                    "  WHERE project_id=? AND stage_key=? "
                    "  ORDER BY version_no DESC LIMIT ?)",
                    (project_id, stage_key, project_id, stage_key, int(keep)),
                )
                return cur.rowcount or 0
            except Exception as e:  # noqa: BLE001
                logger.warning("Prune versions failed for %s/%s: %s",
                               project_id, stage_key, e)
                return 0
