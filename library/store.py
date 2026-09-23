"""SQLite index for the video library.

Same fail-open pattern as ``filmmaker.project_store``: if we can't open the
DB (locked disk, missing path), every method returns an empty result or
no-ops. The user still gets a usable app; only the library index is lost.

Schema is small and boring on purpose. Three tables:

  library_roots   watched folders the user added
  videos          one row per discovered file, keyed by abs_path
  scans           one row per scan job — for progress + history

`videos` carries both the identity fields (sha256, phash_hex) and the
technical metadata (duration/w/h/codec) so a duplicate lookup or a
browser row read is a single query.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger("studiolite.library.store")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS library_roots (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    path         TEXT NOT NULL UNIQUE,
    include_glob TEXT DEFAULT '',
    exclude_glob TEXT DEFAULT '',
    added_at     REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS videos (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    root_id      INTEGER,
    abs_path     TEXT NOT NULL UNIQUE,
    rel_path     TEXT NOT NULL,
    size_bytes   INTEGER,
    mtime        REAL,
    sha256       TEXT,
    phash_hex    TEXT,
    duration_sec REAL,
    width        INTEGER,
    height       INTEGER,
    codec        TEXT,
    fps          REAL,
    added_at     REAL NOT NULL,
    scanned_at   REAL,
    missing      INTEGER DEFAULT 0,
    -- T2 content-index columns.
    embedding    BLOB,       -- float32[512], L2-normalized, mean-pooled CLIP
    embed_model  TEXT,       -- e.g. "openai/clip-vit-base-patch32"
    embedded_at  REAL,
    tags_json    TEXT,       -- JSON array of {tag, score}
    cluster_id   INTEGER,    -- populated by the "clusters" job; null until then
    -- Media kind — "video" (default, back-compat) or "image".
    media_kind   TEXT DEFAULT 'video',
    -- Speech-to-text index. `transcript_json` holds the raw segment list;
    -- the full-text-search index below joins on video_id for search.
    transcript_json  TEXT,
    transcript_model TEXT,
    transcribed_at   REAL,
    FOREIGN KEY (root_id) REFERENCES library_roots(id) ON DELETE SET NULL
);

-- FTS5 mirror of transcript text for fast "find the clip where X was said"
-- lookups. Kept in sync manually from set_transcript(); we don't wire
-- triggers because the source column is JSON that FTS can't index directly.
CREATE VIRTUAL TABLE IF NOT EXISTS transcripts_fts USING fts5(
    video_id UNINDEXED,
    text,
    tokenize='porter unicode61 remove_diacritics 2'
);

CREATE INDEX IF NOT EXISTS idx_videos_sha256 ON videos(sha256);
CREATE INDEX IF NOT EXISTS idx_videos_phash  ON videos(phash_hex);
CREATE INDEX IF NOT EXISTS idx_videos_root   ON videos(root_id);
CREATE INDEX IF NOT EXISTS idx_videos_size   ON videos(size_bytes);

CREATE TABLE IF NOT EXISTS scans (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at   REAL NOT NULL,
    finished_at  REAL,
    status       TEXT NOT NULL,       -- queued|running|done|failed|cancelled
    root_ids     TEXT NOT NULL,       -- json array
    discovered   INTEGER DEFAULT 0,
    hashed       INTEGER DEFAULT 0,
    phashed      INTEGER DEFAULT 0,
    skipped      INTEGER DEFAULT 0,
    error        TEXT
);
"""


@dataclass
class Video:
    id: int
    root_id: Optional[int]
    abs_path: str
    rel_path: str
    size_bytes: int
    mtime: float
    sha256: Optional[str]
    phash_hex: Optional[str]
    duration_sec: Optional[float]
    width: Optional[int]
    height: Optional[int]
    codec: Optional[str]
    fps: Optional[float]
    added_at: float
    scanned_at: Optional[float]
    missing: bool
    # T2 content-index fields — small, JSON-safe, no raw embedding bytes.
    tags: List[Dict[str, Any]] = None  # type: ignore[assignment]
    cluster_id: Optional[int] = None
    embedded: bool = False             # true when an embedding is stored
    # "video" or "image" — set by the scanner at discovery.
    kind: str = "video"
    # Speech index — True when a transcript is stored. Actual segments
    # live in a dedicated read via `get_transcript()` to keep list rows small.
    transcribed: bool = False
    transcript_language: Optional[str] = None


class LibraryStore:
    """Wraps the library SQLite DB. Thread-safe via a single write lock and
    connection-per-call for reads. Cheap to construct — one per process is fine."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._lock = threading.RLock()
        self._ok = True
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        try:
            with self._connect() as c:
                c.executescript(_SCHEMA)
                self._migrate(c)
                c.commit()
        except Exception as e:
            logger.warning("library store failed to initialize (%s); fail-open", e)
            self._ok = False

    def _migrate(self, c: sqlite3.Connection) -> None:
        """Idempotently apply any column additions the current schema needs.
        SQLite's `CREATE TABLE IF NOT EXISTS` never touches existing tables,
        so new columns land here."""
        try:
            cols = {r["name"] for r in c.execute("PRAGMA table_info(videos)").fetchall()}
        except Exception:
            return
        needs = [
            ("embedding",        "BLOB"),
            ("embed_model",      "TEXT"),
            ("embedded_at",      "REAL"),
            ("tags_json",        "TEXT"),
            ("cluster_id",       "INTEGER"),
            ("media_kind",       "TEXT DEFAULT 'video'"),
            ("transcript_json",  "TEXT"),
            ("transcript_model", "TEXT"),
            ("transcribed_at",   "REAL"),
        ]
        for name, coltype in needs:
            if name not in cols:
                try:
                    c.execute(f"ALTER TABLE videos ADD COLUMN {name} {coltype}")
                except Exception as e:
                    logger.warning("could not add column %s (%s)", name, e)

    # ---- connection --------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=15.0, isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.row_factory = sqlite3.Row
        return conn

    # ---- roots -------------------------------------------------------------

    def add_root(self, path: str, include_glob: str = "", exclude_glob: str = "") -> Optional[int]:
        if not self._ok: return None
        abs_path = os.path.abspath(path)
        with self._lock:
            try:
                with self._connect() as c:
                    cur = c.execute(
                        "INSERT OR IGNORE INTO library_roots(path, include_glob, exclude_glob, added_at) "
                        "VALUES(?,?,?,?)",
                        (abs_path, include_glob or "", exclude_glob or "", time.time()),
                    )
                    if cur.lastrowid:
                        return int(cur.lastrowid)
                    # Already existed — fetch id
                    row = c.execute("SELECT id FROM library_roots WHERE path=?", (abs_path,)).fetchone()
                    return int(row["id"]) if row else None
            except Exception as e:
                logger.warning("add_root(%s) failed: %s", path, e)
                return None

    def remove_root(self, root_id: int) -> int:
        if not self._ok: return 0
        with self._lock:
            try:
                with self._connect() as c:
                    cur = c.execute("DELETE FROM library_roots WHERE id=?", (root_id,))
                    return int(cur.rowcount)
            except Exception as e:
                logger.warning("remove_root failed: %s", e)
                return 0

    def list_roots(self) -> List[Dict[str, Any]]:
        if not self._ok: return []
        try:
            with self._connect() as c:
                rows = c.execute(
                    "SELECT id, path, include_glob, exclude_glob, added_at FROM library_roots ORDER BY added_at ASC"
                ).fetchall()
        except Exception:
            return []
        # Attach a video-count for each so the UI can render at-a-glance stats.
        out = []
        with self._connect() as c:
            for r in rows:
                n = c.execute("SELECT COUNT(*) as n FROM videos WHERE root_id=? AND missing=0",
                              (r["id"],)).fetchone()["n"]
                out.append({
                    "id": r["id"], "path": r["path"],
                    "include_glob": r["include_glob"], "exclude_glob": r["exclude_glob"],
                    "added_at": r["added_at"], "video_count": n,
                })
        return out

    def get_root(self, root_id: int) -> Optional[Dict[str, Any]]:
        if not self._ok: return None
        try:
            with self._connect() as c:
                r = c.execute("SELECT id, path, include_glob, exclude_glob FROM library_roots WHERE id=?",
                              (root_id,)).fetchone()
                return dict(r) if r else None
        except Exception:
            return None

    # ---- videos ------------------------------------------------------------

    def upsert_video_discovered(self, root_id: Optional[int], abs_path: str,
                                rel_path: str, size_bytes: int, mtime: float,
                                kind: str = "video") -> Optional[int]:
        """Called by the scanner when it FIRST sees a file. Only touches
        identity/inode-shape fields — sha256/phash/probe come later.
        `kind` is either "video" or "image"; on re-scan we lock it in so
        two different scanners agree."""
        if not self._ok: return None
        now = time.time()
        with self._lock:
            try:
                with self._connect() as c:
                    row = c.execute("SELECT id, mtime, size_bytes FROM videos WHERE abs_path=?",
                                    (abs_path,)).fetchone()
                    if row is None:
                        cur = c.execute(
                            "INSERT INTO videos(root_id, abs_path, rel_path, size_bytes, mtime, added_at, missing, media_kind) "
                            "VALUES(?,?,?,?,?,?,0,?)",
                            (root_id, abs_path, rel_path, size_bytes, mtime, now, kind),
                        )
                        return int(cur.lastrowid)
                    # Existing row — refresh mtime/size/missing but keep hashes if
                    # the file wasn't rewritten (mtime + size both unchanged).
                    unchanged = row["mtime"] == mtime and row["size_bytes"] == size_bytes
                    if not unchanged:
                        c.execute(
                            "UPDATE videos SET size_bytes=?, mtime=?, sha256=NULL, phash_hex=NULL, missing=0, media_kind=? WHERE id=?",
                            (size_bytes, mtime, kind, row["id"]),
                        )
                    else:
                        c.execute("UPDATE videos SET missing=0, media_kind=? WHERE id=?",
                                  (kind, row["id"]))
                    return int(row["id"])
            except Exception as e:
                logger.warning("upsert_video_discovered(%s) failed: %s", abs_path, e)
                return None

    def update_video_metadata(self, video_id: int, **fields: Any) -> None:
        if not self._ok or not fields: return
        allowed = {"sha256", "phash_hex", "duration_sec", "width", "height",
                   "codec", "fps", "scanned_at", "missing"}
        cols, vals = [], []
        for k, v in fields.items():
            if k in allowed:
                cols.append(f"{k}=?")
                vals.append(v)
        if not cols: return
        vals.append(video_id)
        with self._lock:
            try:
                with self._connect() as c:
                    c.execute(f"UPDATE videos SET {', '.join(cols)} WHERE id=?", vals)
            except Exception as e:
                logger.warning("update_video_metadata failed: %s", e)

    def mark_missing(self, video_id: int) -> None:
        self.update_video_metadata(video_id, missing=1)

    def delete_video(self, video_id: int) -> bool:
        if not self._ok: return False
        with self._lock:
            try:
                with self._connect() as c:
                    cur = c.execute("DELETE FROM videos WHERE id=?", (video_id,))
                    return cur.rowcount > 0
            except Exception:
                return False

    def get_video(self, video_id: int) -> Optional[Video]:
        if not self._ok: return None
        try:
            with self._connect() as c:
                r = c.execute("SELECT * FROM videos WHERE id=?", (video_id,)).fetchone()
                return _video_from_row(r) if r else None
        except Exception:
            return None

    def get_video_by_path(self, abs_path: str) -> Optional[Video]:
        if not self._ok: return None
        try:
            with self._connect() as c:
                r = c.execute("SELECT * FROM videos WHERE abs_path=?", (abs_path,)).fetchone()
                return _video_from_row(r) if r else None
        except Exception:
            return None

    def list_videos(self, *, root_id: Optional[int] = None,
                    limit: int = 200, offset: int = 0,
                    include_missing: bool = False,
                    query: Optional[str] = None,
                    kind: Optional[str] = None) -> Tuple[List[Video], int]:
        if not self._ok: return ([], 0)
        clauses, vals = [], []
        if not include_missing:
            clauses.append("missing=0")
        if root_id is not None:
            clauses.append("root_id=?"); vals.append(root_id)
        if kind in ("video", "image"):
            clauses.append("media_kind=?"); vals.append(kind)
        if query:
            clauses.append("(abs_path LIKE ? OR rel_path LIKE ?)")
            like = f"%{query}%"
            vals.extend([like, like])
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        try:
            with self._connect() as c:
                total = c.execute(f"SELECT COUNT(*) as n FROM videos {where}", vals).fetchone()["n"]
                rows = c.execute(
                    f"SELECT * FROM videos {where} ORDER BY added_at DESC LIMIT ? OFFSET ?",
                    (*vals, limit, offset),
                ).fetchall()
                return ([_video_from_row(r) for r in rows], int(total))
        except Exception:
            return ([], 0)

    def iter_videos_needing_hash(self, limit: int = 500) -> List[Video]:
        if not self._ok: return []
        try:
            with self._connect() as c:
                rows = c.execute(
                    "SELECT * FROM videos WHERE sha256 IS NULL AND missing=0 ORDER BY size_bytes ASC LIMIT ?",
                    (limit,),
                ).fetchall()
                return [_video_from_row(r) for r in rows]
        except Exception:
            return []

    def iter_videos_needing_phash(self, limit: int = 500) -> List[Video]:
        if not self._ok: return []
        try:
            with self._connect() as c:
                rows = c.execute(
                    "SELECT * FROM videos WHERE phash_hex IS NULL AND missing=0 ORDER BY size_bytes ASC LIMIT ?",
                    (limit,),
                ).fetchall()
                return [_video_from_row(r) for r in rows]
        except Exception:
            return []

    def iter_videos_needing_embedding(self, limit: int = 500,
                                       model_id: Optional[str] = None) -> List[Video]:
        if not self._ok: return []
        try:
            with self._connect() as c:
                if model_id:
                    rows = c.execute(
                        "SELECT * FROM videos WHERE missing=0 AND "
                        "(embedding IS NULL OR embed_model IS NULL OR embed_model != ?) "
                        "ORDER BY size_bytes ASC LIMIT ?",
                        (model_id, limit),
                    ).fetchall()
                else:
                    rows = c.execute(
                        "SELECT * FROM videos WHERE embedding IS NULL AND missing=0 "
                        "ORDER BY size_bytes ASC LIMIT ?", (limit,),
                    ).fetchall()
                return [_video_from_row(r) for r in rows]
        except Exception:
            return []

    def load_all_embeddings(self, dim: int = 512):
        """Return (video_ids, matrix) for every video that has an embedding.
        `matrix` is (N, dim) float32. Rows preserve the row-order of ids."""
        if not self._ok: return ([], None)
        import numpy as np
        ids: List[int] = []
        blobs: List[bytes] = []
        try:
            with self._connect() as c:
                for r in c.execute(
                    "SELECT id, embedding FROM videos WHERE embedding IS NOT NULL AND missing=0"
                ).fetchall():
                    ids.append(int(r["id"]))
                    blobs.append(r["embedding"])
        except Exception:
            return ([], None)
        if not blobs:
            return (ids, np.zeros((0, dim), dtype=np.float32))
        try:
            matrix = np.stack([np.frombuffer(b, dtype=np.float32)[:dim] for b in blobs])
        except Exception:
            return ([], None)
        return (ids, matrix.astype(np.float32))

    def set_embedding(self, video_id: int, embedding_blob: bytes, model_id: str,
                       tags_json: Optional[str] = None) -> None:
        if not self._ok: return
        import time as _time
        with self._lock:
            try:
                with self._connect() as c:
                    fields = ["embedding=?", "embed_model=?", "embedded_at=?"]
                    vals: list = [embedding_blob, model_id, _time.time()]
                    if tags_json is not None:
                        fields.append("tags_json=?"); vals.append(tags_json)
                    vals.append(video_id)
                    c.execute(f"UPDATE videos SET {', '.join(fields)} WHERE id=?", vals)
            except Exception as e:
                logger.warning("set_embedding failed: %s", e)

    def clear_cluster_ids(self) -> None:
        if not self._ok: return
        with self._lock:
            try:
                with self._connect() as c:
                    c.execute("UPDATE videos SET cluster_id=NULL WHERE missing=0")
            except Exception:
                pass

    def set_cluster_ids(self, mapping: Dict[int, int]) -> None:
        """Bulk assign cluster ids by video id."""
        if not self._ok or not mapping: return
        with self._lock:
            try:
                with self._connect() as c:
                    c.executemany(
                        "UPDATE videos SET cluster_id=? WHERE id=?",
                        [(cid, vid) for vid, cid in mapping.items()],
                    )
            except Exception as e:
                logger.warning("set_cluster_ids failed: %s", e)

    # ---- transcripts ------------------------------------------------------

    def iter_videos_needing_transcript(self, limit: int = 500,
                                        model_id: Optional[str] = None) -> List[Video]:
        """Videos without a transcript (or transcribed by a different model
        than `model_id`). Images always skip — no audio to transcribe."""
        if not self._ok: return []
        try:
            with self._connect() as c:
                if model_id:
                    rows = c.execute(
                        "SELECT * FROM videos WHERE missing=0 AND media_kind='video' AND "
                        "(transcript_json IS NULL OR transcript_model IS NULL OR transcript_model != ?) "
                        "ORDER BY size_bytes ASC LIMIT ?",
                        (model_id, limit),
                    ).fetchall()
                else:
                    rows = c.execute(
                        "SELECT * FROM videos WHERE transcript_json IS NULL AND missing=0 "
                        "AND media_kind='video' ORDER BY size_bytes ASC LIMIT ?", (limit,),
                    ).fetchall()
                return [_video_from_row(r) for r in rows]
        except Exception:
            return []

    def set_transcript(self, video_id: int, transcript: Dict[str, Any],
                       model_id: str) -> None:
        """Write the transcript blob AND refresh the FTS mirror in one txn.
        `transcript` shape: {language, text, segments:[{start,end,text}]}."""
        if not self._ok: return
        import time as _time
        text_all = str(transcript.get("text") or "").strip()
        if not text_all:
            segs = transcript.get("segments") or []
            if isinstance(segs, list):
                text_all = " ".join(str(s.get("text", "")).strip()
                                    for s in segs if isinstance(s, dict))
        payload = json.dumps(transcript, ensure_ascii=False)
        with self._lock:
            try:
                with self._connect() as c:
                    c.execute("BEGIN")
                    c.execute(
                        "UPDATE videos SET transcript_json=?, transcript_model=?, transcribed_at=? WHERE id=?",
                        (payload, model_id, _time.time(), video_id),
                    )
                    # Refresh FTS row.
                    c.execute("DELETE FROM transcripts_fts WHERE video_id=?", (video_id,))
                    if text_all:
                        c.execute(
                            "INSERT INTO transcripts_fts(video_id, text) VALUES(?, ?)",
                            (video_id, text_all),
                        )
                    c.execute("COMMIT")
            except Exception as e:
                logger.warning("set_transcript failed: %s", e)

    def get_transcript(self, video_id: int) -> Optional[Dict[str, Any]]:
        if not self._ok: return None
        try:
            with self._connect() as c:
                r = c.execute("SELECT transcript_json, transcript_model, transcribed_at "
                              "FROM videos WHERE id=?", (video_id,)).fetchone()
                if not r or not r["transcript_json"]:
                    return None
                try:
                    payload = json.loads(r["transcript_json"])
                except Exception:
                    return None
                payload["_model"] = r["transcript_model"]
                payload["_at"] = r["transcribed_at"]
                return payload
        except Exception:
            return None

    def clear_transcript(self, video_id: int) -> None:
        if not self._ok: return
        with self._lock:
            try:
                with self._connect() as c:
                    c.execute("UPDATE videos SET transcript_json=NULL, transcript_model=NULL, "
                              "transcribed_at=NULL WHERE id=?", (video_id,))
                    c.execute("DELETE FROM transcripts_fts WHERE video_id=?", (video_id,))
            except Exception:
                pass

    def search_transcripts(self, query: str, *, limit: int = 24) -> List[Dict[str, Any]]:
        """FTS5 search across every stored transcript. Returns hits with a
        short snippet and the video row so the caller can also pull the
        segment timestamps from get_transcript() for click-to-seek."""
        if not self._ok or not query.strip():
            return []
        try:
            with self._connect() as c:
                # `snippet` returns text with the match wrapped in ⟨…⟩ so the
                # UI can highlight without re-searching client-side.
                rows = c.execute(
                    "SELECT v.*, "
                    "snippet(transcripts_fts, 1, '⟪', '⟫', '…', 12) AS snippet, "
                    "bm25(transcripts_fts) AS rank "
                    "FROM transcripts_fts JOIN videos v ON v.id = transcripts_fts.video_id "
                    "WHERE transcripts_fts MATCH ? AND v.missing=0 "
                    "ORDER BY rank LIMIT ?",
                    (query, limit),
                ).fetchall()
                out = []
                for r in rows:
                    v = _video_from_row(r)
                    out.append({"video": v.__dict__, "snippet": r["snippet"],
                                "rank": float(r["rank"])})
                return out
        except Exception as e:
            logger.warning("search_transcripts failed: %s", e)
            return []

    def list_legacy_videos(self, limit: int = 1000) -> List[Video]:
        """Videos whose codec is in the legacy set defined by reencode.py.
        Only returns rows the probe pass has actually filled in — a null
        codec doesn't count as legacy, it counts as un-probed."""
        if not self._ok: return []
        from .reencode import LEGACY_CODECS
        try:
            with self._connect() as c:
                # Case-insensitive match; ffprobe returns lowercase but the
                # store isn't strict about what other tools put in there.
                placeholders = ",".join("?" for _ in LEGACY_CODECS)
                rows = c.execute(
                    f"SELECT * FROM videos WHERE missing=0 AND media_kind='video' "
                    f"AND codec IS NOT NULL AND LOWER(codec) IN ({placeholders}) "
                    f"ORDER BY size_bytes DESC LIMIT ?",
                    (*[c.lower() for c in LEGACY_CODECS], limit),
                ).fetchall()
                return [_video_from_row(r) for r in rows]
        except Exception as e:
            logger.warning("list_legacy_videos failed: %s", e)
            return []

    def videos_in_cluster(self, cluster_id: int) -> List[Video]:
        if not self._ok: return []
        try:
            with self._connect() as c:
                rows = c.execute(
                    "SELECT * FROM videos WHERE cluster_id=? AND missing=0 "
                    "ORDER BY added_at DESC", (cluster_id,),
                ).fetchall()
                return [_video_from_row(r) for r in rows]
        except Exception:
            return []

    def iter_videos_needing_probe(self, limit: int = 500) -> List[Video]:
        if not self._ok: return []
        try:
            with self._connect() as c:
                rows = c.execute(
                    "SELECT * FROM videos WHERE duration_sec IS NULL AND missing=0 ORDER BY size_bytes ASC LIMIT ?",
                    (limit,),
                ).fetchall()
                return [_video_from_row(r) for r in rows]
        except Exception:
            return []

    # ---- duplicates --------------------------------------------------------

    def exact_duplicate_clusters(self, *, min_group: int = 2) -> List[Dict[str, Any]]:
        """Group videos by SHA-256. Returns clusters with 2+ members."""
        if not self._ok: return []
        try:
            with self._connect() as c:
                hashes = c.execute(
                    "SELECT sha256, COUNT(*) as n FROM videos "
                    "WHERE sha256 IS NOT NULL AND missing=0 GROUP BY sha256 HAVING n>=?",
                    (min_group,),
                ).fetchall()
                clusters = []
                for h in hashes:
                    members = c.execute("SELECT * FROM videos WHERE sha256=? AND missing=0", (h["sha256"],)).fetchall()
                    clusters.append({
                        "kind": "exact",
                        "key": h["sha256"],
                        "members": [_video_from_row(r).__dict__ for r in members],
                    })
                # Sort largest clusters first
                clusters.sort(key=lambda c: len(c["members"]), reverse=True)
                return clusters
        except Exception:
            return []

    def phash_candidate_pairs(self) -> List[Tuple[Video, Video]]:
        """Return every pair of videos that share the same phash prefix. Cheap
        pre-filter that limits the O(n^2) Hamming-distance work to a handful
        of buckets."""
        if not self._ok: return []
        buckets: Dict[str, List[Video]] = {}
        try:
            with self._connect() as c:
                for r in c.execute(
                    "SELECT * FROM videos WHERE phash_hex IS NOT NULL AND missing=0"
                ).fetchall():
                    v = _video_from_row(r)
                    if not v.phash_hex: continue
                    prefix = v.phash_hex[:4]  # 16 bits — plenty of pre-filter
                    buckets.setdefault(prefix, []).append(v)
        except Exception:
            return []
        pairs: List[Tuple[Video, Video]] = []
        for bucket in buckets.values():
            if len(bucket) < 2: continue
            for i in range(len(bucket)):
                for j in range(i + 1, len(bucket)):
                    pairs.append((bucket[i], bucket[j]))
        return pairs

    # ---- scans -------------------------------------------------------------

    def create_scan(self, root_ids: List[int]) -> Optional[int]:
        if not self._ok: return None
        with self._lock:
            try:
                with self._connect() as c:
                    cur = c.execute(
                        "INSERT INTO scans(started_at, status, root_ids) VALUES(?,?,?)",
                        (time.time(), "queued", json.dumps(root_ids)),
                    )
                    return int(cur.lastrowid)
            except Exception:
                return None

    def update_scan(self, scan_id: int, **fields: Any) -> None:
        if not self._ok or not fields: return
        allowed = {"status", "finished_at", "discovered", "hashed", "phashed", "skipped", "error"}
        cols, vals = [], []
        for k, v in fields.items():
            if k in allowed:
                cols.append(f"{k}=?"); vals.append(v)
        if not cols: return
        vals.append(scan_id)
        with self._lock:
            try:
                with self._connect() as c:
                    c.execute(f"UPDATE scans SET {', '.join(cols)} WHERE id=?", vals)
            except Exception:
                pass

    def get_scan(self, scan_id: int) -> Optional[Dict[str, Any]]:
        if not self._ok: return None
        try:
            with self._connect() as c:
                r = c.execute("SELECT * FROM scans WHERE id=?", (scan_id,)).fetchone()
                if not r: return None
                d = dict(r)
                try:
                    d["root_ids"] = json.loads(d["root_ids"] or "[]")
                except Exception:
                    d["root_ids"] = []
                return d
        except Exception:
            return None

    def list_scans(self, limit: int = 20) -> List[Dict[str, Any]]:
        if not self._ok: return []
        try:
            with self._connect() as c:
                rows = c.execute(
                    "SELECT * FROM scans ORDER BY started_at DESC LIMIT ?", (limit,)
                ).fetchall()
                out = []
                for r in rows:
                    d = dict(r)
                    try:
                        d["root_ids"] = json.loads(d["root_ids"] or "[]")
                    except Exception:
                        d["root_ids"] = []
                    out.append(d)
                return out
        except Exception:
            return []

    # ---- summary -----------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        if not self._ok:
            return {"videos": 0, "roots": 0, "total_bytes": 0, "hashed": 0, "phashed": 0,
                    "transcribed": 0, "video_count": 0, "image_count": 0}
        try:
            with self._connect() as c:
                v = c.execute(
                    "SELECT COUNT(*) as n, IFNULL(SUM(size_bytes),0) as bytes, "
                    "SUM(CASE WHEN sha256 IS NOT NULL THEN 1 ELSE 0 END) as hashed, "
                    "SUM(CASE WHEN phash_hex IS NOT NULL THEN 1 ELSE 0 END) as phashed, "
                    "SUM(CASE WHEN transcript_json IS NOT NULL THEN 1 ELSE 0 END) as transcribed, "
                    "SUM(CASE WHEN media_kind='video' THEN 1 ELSE 0 END) as vids, "
                    "SUM(CASE WHEN media_kind='image' THEN 1 ELSE 0 END) as imgs "
                    "FROM videos WHERE missing=0"
                ).fetchone()
                r = c.execute("SELECT COUNT(*) as n FROM library_roots").fetchone()
                return {
                    "videos": int(v["n"]), "roots": int(r["n"]),
                    "total_bytes": int(v["bytes"] or 0),
                    "hashed": int(v["hashed"] or 0),
                    "phashed": int(v["phashed"] or 0),
                    "transcribed": int(v["transcribed"] or 0),
                    "video_count": int(v["vids"] or 0),
                    "image_count": int(v["imgs"] or 0),
                }
        except Exception:
            return {"videos": 0, "roots": 0, "total_bytes": 0, "hashed": 0, "phashed": 0,
                    "transcribed": 0, "video_count": 0, "image_count": 0}


def _video_from_row(r) -> Video:
    tags: List[Dict[str, Any]] = []
    tags_raw = None
    cluster_id = None
    has_embedding = False
    kind = "video"
    transcribed = False
    transcript_language: Optional[str] = None
    # SQLite Row supports both index and key access; guard when a query
    # doesn't project the T2 columns (older SELECT * calls).
    try:
        tags_raw = r["tags_json"]
    except Exception:
        pass
    try:
        cluster_id = r["cluster_id"]
    except Exception:
        pass
    try:
        has_embedding = r["embedding"] is not None
    except Exception:
        pass
    try:
        kind = r["media_kind"] or "video"
    except Exception:
        pass
    try:
        tj = r["transcript_json"]
        if tj:
            transcribed = True
            try:
                parsed = json.loads(tj)
                if isinstance(parsed, dict):
                    transcript_language = parsed.get("language")
            except Exception:
                pass
    except Exception:
        pass
    if tags_raw:
        try:
            parsed = json.loads(tags_raw)
            if isinstance(parsed, list):
                tags = [t for t in parsed if isinstance(t, dict)]
        except Exception:
            tags = []
    return Video(
        id=int(r["id"]), root_id=(int(r["root_id"]) if r["root_id"] is not None else None),
        abs_path=r["abs_path"], rel_path=r["rel_path"],
        size_bytes=int(r["size_bytes"] or 0), mtime=float(r["mtime"] or 0),
        sha256=r["sha256"], phash_hex=r["phash_hex"],
        duration_sec=(float(r["duration_sec"]) if r["duration_sec"] is not None else None),
        width=(int(r["width"]) if r["width"] is not None else None),
        height=(int(r["height"]) if r["height"] is not None else None),
        codec=r["codec"],
        fps=(float(r["fps"]) if r["fps"] is not None else None),
        added_at=float(r["added_at"]), scanned_at=(float(r["scanned_at"]) if r["scanned_at"] is not None else None),
        missing=bool(r["missing"] or 0),
        tags=tags,
        cluster_id=(int(cluster_id) if cluster_id is not None else None),
        embedded=has_embedding,
        kind=kind,
        transcribed=transcribed,
        transcript_language=transcript_language,
    )
