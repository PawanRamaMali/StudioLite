"""SHA-256 fingerprint for exact-duplicate detection.

For huge files we do a two-stage identity: a fast fingerprint of the
first + last 4 MiB (used as a cheap pre-filter) and, on hash collision
or when the caller asks for it, a full-file SHA-256.

In practice full-file SHA-256 on modern SSDs is roughly disk speed. The
partial fingerprint exists to short-circuit obvious mismatches when two
files of the same size happen to sit in the same phash bucket during
near-duplicate work; we don't need it for the primary dedupe path.
"""
from __future__ import annotations

import hashlib
import os
from typing import Callable, Optional

CHUNK = 1 << 20  # 1 MiB


def sha256_file(path: str, *, cancel: Optional[Callable[[], bool]] = None) -> Optional[str]:
    """Full-file SHA-256 as hex, or None if the file can't be read.

    `cancel` is a cheap poll called every chunk — the scanner uses it so a
    long hash on a massive file yields to a cancel request."""
    h = hashlib.sha256()
    try:
        with open(path, "rb", buffering=0) as f:
            while True:
                if cancel and cancel():
                    return None
                buf = f.read(CHUNK)
                if not buf:
                    break
                h.update(buf)
    except OSError:
        return None
    return h.hexdigest()


def head_tail_fingerprint(path: str, edge_bytes: int = 4 * 1024 * 1024) -> Optional[str]:
    """A SHA-256 over (first edge_bytes) + (last edge_bytes) + size.
    Cheap way to disqualify almost-identical files without paying for a
    full-file hash. Not used for exact-duplicate clustering — that always
    uses the full sha256."""
    try:
        size = os.path.getsize(path)
        h = hashlib.sha256()
        h.update(str(size).encode())
        with open(path, "rb", buffering=0) as f:
            head = f.read(min(edge_bytes, size))
            h.update(head)
            if size > edge_bytes:
                f.seek(max(0, size - edge_bytes))
                h.update(f.read(edge_bytes))
        return h.hexdigest()
    except OSError:
        return None
