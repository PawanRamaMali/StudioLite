"""SQLite-backed persistent job store for the HTTP API.

Design goals
------------

The in-process ``jobs`` dict that api_server used to keep lives here
now. The important invariants:

  1. **API-compatible.** ``PersistentJobStore`` mimics enough of a
     ``dict[str, dict]`` interface that existing callers (which do
     ``jobs[job_id] = {...}``, ``jobs.get(...)``, ``jobs.items()``,
     ``len(jobs)``, ``in``) keep working. Every mutation is persisted
     synchronously; the in-memory copy is authoritative for reads
     within one process so the ~200 ms of typical SQLite churn per
     read is skipped.

  2. **Restart safe.** On import we run ``recover_interrupted_jobs()``
     which flips any job left in ``queued`` or ``running`` state at
     shutdown to ``interrupted``. That gives the user a clean signal
     to retry rather than a job that appears to still be running.

  3. **Fail open.** If SQLite can't be opened (locked disk, path
     doesn't exist), we fall back to a pure in-memory ``dict`` and
     log a warning. The API still works; only persistence is lost.

  4. **Boring schema.** One table. ``result`` and ``params`` are JSON
     blobs so the shape can evolve without a migration for every
     new job kind. Add a proper migrations tool the day a second
     table shows up.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple

logger = logging.getLogger("studiolite.job_store")


_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id       TEXT PRIMARY KEY,
    kind         TEXT NOT NULL,
    status       TEXT NOT NULL,
    progress     REAL NOT NULL DEFAULT 0.0,
    message      TEXT NOT NULL DEFAULT '',
    result       TEXT,               -- json, nullable
    error        TEXT,
    created_at   REAL NOT NULL,
    updated_at   REAL NOT NULL,
    params       TEXT NOT NULL DEFAULT '{}'   -- json
);

CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON jobs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
"""

# Columns in the order to_row / from_row expects. Keeping this near the
# schema definition makes drift obvious the day we add a column.
_COLUMNS: Tuple[str, ...] = (
    "job_id", "kind", "status", "progress", "message",
    "result", "error", "created_at", "updated_at", "params",
)


def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    """Turn a DB row into the shape the rest of api_server expects.

    ``result`` and ``params`` come back parsed; a corrupted JSON blob
    falls back to ``None`` / ``{}`` rather than crashing the whole read
    (a single bad row shouldn't take out the jobs list)."""
    d: Dict[str, Any] = {c: row[c] for c in _COLUMNS}
    for k in ("result", "params"):
        v = d.get(k)
        if v is None or v == "":
            d[k] = {} if k == "params" else None
            continue
        try:
            d[k] = json.loads(v)
        except (TypeError, ValueError):
            d[k] = {} if k == "params" else None
    return d


def _dict_to_row(job_id: str, job: Dict[str, Any]) -> Tuple[Any, ...]:
    """Squash a job dict into the tuple order sqlite expects. Every column
    is present so an INSERT-or-REPLACE is safe to reuse for updates."""
    return (
        job_id,
        str(job.get("kind", "")),
        str(job.get("status", "queued")),
        float(job.get("progress", 0.0) or 0.0),
        str(job.get("message", "") or ""),
        json.dumps(job.get("result")) if job.get("result") is not None else None,
        str(job.get("error", "") or "") or None,
        float(job.get("created_at", time.time())),
        float(job.get("updated_at", time.time())),
        json.dumps(job.get("params") or {}),
    )


class PersistentJobStore:
    """SQLite-backed job store that quacks like ``dict[str, dict]`` for the
    subset of dict methods the API surface uses.

    The store is **not** meant to be a thread-safe general-purpose dict
- it's meant to be a drop-in replacement for the specific access
    patterns already in api_server.py. Callers protect their own
    read-modify-write cycles with the module-level ``_jobs_lock``
    they had before this class existed."""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._mem: Dict[str, Dict[str, Any]] = {}
        self._open_or_fallback()

    # --- lifecycle --------------------------------------------------------

    def _open_or_fallback(self) -> None:
        try:
            os.makedirs(os.path.dirname(self._db_path) or ".", exist_ok=True)
            conn = sqlite3.connect(
                self._db_path,
                check_same_thread=False,
                isolation_level=None,  # autocommit; explicit BEGIN when we need one
            )
            conn.row_factory = sqlite3.Row
            conn.executescript(_SCHEMA)
            self._conn = conn
            # Load everything into the in-memory cache so reads are cheap.
            for row in conn.execute("SELECT * FROM jobs"):
                self._mem[row["job_id"]] = _row_to_dict(row)
            logger.info("Job store opened at %s (%d rows)", self._db_path, len(self._mem))
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "Could not open job store at %s (%s); running in-memory only.",
                self._db_path, e,
            )
            self._conn = None
            self._mem = {}

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:  # noqa: BLE001
                pass
            self._conn = None

    # --- dict-like interface ---------------------------------------------

    def __setitem__(self, job_id: str, job: Dict[str, Any]) -> None:
        job = dict(job)
        job.setdefault("created_at", time.time())
        job["updated_at"] = time.time()
        self._mem[job_id] = job
        if self._conn is not None:
            try:
                self._conn.execute(
                    "INSERT OR REPLACE INTO jobs (" + ",".join(_COLUMNS) + ") "
                    "VALUES (?,?,?,?,?,?,?,?,?,?)",
                    _dict_to_row(job_id, job),
                )
            except Exception as e:  # noqa: BLE001
                logger.warning("Job store write failed for %s: %s", job_id, e)

    def __getitem__(self, job_id: str) -> Dict[str, Any]:
        return self._mem[job_id]

    def __contains__(self, job_id: object) -> bool:
        return job_id in self._mem

    def __len__(self) -> int:
        return len(self._mem)

    def __iter__(self) -> Iterator[str]:
        return iter(self._mem)

    def get(self, job_id: str, default: Any = None) -> Any:
        return self._mem.get(job_id, default)

    def items(self):
        return self._mem.items()

    def values(self):
        return self._mem.values()

    def keys(self):
        return self._mem.keys()

    def update_fields(self, job_id: str, **fields: Any) -> None:
        """Partial update, used by _update_job. Falls back to __setitem__
        so the persistence path stays consistent."""
        job = self._mem.get(job_id)
        if job is None:
            return
        job = dict(job)
        job.update(fields)
        self[job_id] = job

    def clear(self) -> None:
        """Test-only reset. Not part of the API surface - the app never
        wipes its own job table at runtime."""
        self._mem.clear()
        if self._conn is not None:
            try:
                self._conn.execute("DELETE FROM jobs")
            except Exception:  # noqa: BLE001
                pass

    # --- housekeeping -----------------------------------------------------

    def recover_interrupted_jobs(self) -> List[str]:
        """Called at import time so a process crash / kill leaves the table
        honest. Anything that was ``running`` or ``queued`` at the last
        write is flipped to ``interrupted`` so the user can retry it
        instead of seeing a fake in-flight indicator."""
        recovered: List[str] = []
        for job_id, job in list(self._mem.items()):
            if job.get("status") in ("running", "queued", "cancelling"):
                job = dict(job)
                job["status"] = "interrupted"
                job["message"] = "Job was in flight when the process exited."
                self[job_id] = job
                recovered.append(job_id)
        if recovered:
            logger.info("Recovered %d interrupted jobs.", len(recovered))
        return recovered
