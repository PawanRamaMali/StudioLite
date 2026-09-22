"""Directory walker for the library.

Two responsibilities: iterate videos under a root, and reconcile the store
with what's on disk (mark rows missing when a file disappears). Everything
IO — no hashing or probing here."""
from __future__ import annotations

import fnmatch
import logging
import os
from dataclasses import dataclass
from typing import Callable, Iterable, Iterator, List, Optional

from .store import LibraryStore

logger = logging.getLogger("studiolite.library.scanner")


# Every video container ffmpeg understands. Comprehensive on purpose —
# a library "organize my videos" tool shouldn't silently skip a format.
# If the user has a .rm or .mxf lying around, they want it visible so
# they can act on it (dedupe, delete, re-encode).
VIDEO_EXTS = frozenset({
    # ISO/BMFF family
    ".mp4", ".m4v", ".m4p", ".mov", ".qt",
    # Matroska / WebM
    ".mkv", ".mk3d", ".webm",
    # AVI + Windows Media
    ".avi", ".divx", ".wmv", ".asf",
    # MPEG program/transport streams
    ".mpg", ".mpeg", ".mp2", ".mpe", ".mpv", ".m2v",
    ".ts", ".m2ts", ".mts", ".m2t", ".tp", ".trp",
    # DVD / Blu-ray raw
    ".vob", ".evo",
    # Real, Flash, others
    ".rm", ".rmvb", ".flv", ".f4v", ".f4p", ".swf",
    # Ogg family
    ".ogv", ".ogm",
    # Mobile
    ".3gp", ".3g2",
    # Professional / broadcast
    ".mxf", ".mts", ".dv", ".dvr-ms",
    # Miscellaneous but real
    ".amv", ".drc", ".yuv", ".mng", ".roq", ".nsv",
    ".svi", ".mtv", ".gxf",
    # Newer / codec-specific containers
    ".av1", ".ivf", ".h264", ".h265", ".hevc", ".265", ".264",
})


@dataclass
class Discovered:
    root_id: Optional[int]
    abs_path: str
    rel_path: str
    size_bytes: int
    mtime: float


def _matches(path: str, include_glob: str, exclude_glob: str) -> bool:
    name = os.path.basename(path)
    if include_glob:
        includes = [g.strip() for g in include_glob.split(",") if g.strip()]
        if includes and not any(fnmatch.fnmatch(name, g) for g in includes):
            return False
    if exclude_glob:
        excludes = [g.strip() for g in exclude_glob.split(",") if g.strip()]
        if any(fnmatch.fnmatch(name, g) for g in excludes):
            return False
    return True


def walk_root(root_path: str, include_glob: str = "", exclude_glob: str = "",
              max_files: int = 200_000) -> Iterator[Discovered]:
    """Yield videos under `root_path`. Ignores permission errors on
    subtrees so a Windows Recycle Bin or a driver-restricted folder
    inside the tree doesn't nuke the whole scan."""
    if not os.path.isdir(root_path):
        return
    root_abs = os.path.abspath(root_path)
    count = 0
    for dirpath, dirnames, filenames in os.walk(root_abs, topdown=True, followlinks=False):
        # Skip a couple of well-known noisy Windows folders.
        dirnames[:] = [d for d in dirnames if d not in {
            "$RECYCLE.BIN", "System Volume Information", ".git",
        } and not d.startswith(".")]
        for name in filenames:
            ext = os.path.splitext(name)[1].lower()
            if ext not in VIDEO_EXTS:
                continue
            abs_path = os.path.join(dirpath, name)
            if not _matches(abs_path, include_glob, exclude_glob):
                continue
            try:
                st = os.stat(abs_path)
            except OSError:
                continue
            rel_path = os.path.relpath(abs_path, root_abs).replace("\\", "/")
            yield Discovered(root_id=None, abs_path=abs_path, rel_path=rel_path,
                             size_bytes=st.st_size, mtime=st.st_mtime)
            count += 1
            if count >= max_files:
                logger.warning("walk_root %s hit max_files=%d; stopping", root_abs, max_files)
                return


def discover(store: LibraryStore, root_ids: List[int],
             progress: Optional[Callable[[str, int], None]] = None,
             cancel: Optional[Callable[[], bool]] = None) -> int:
    """Walk each root, upsert every discovered file. Returns the count of
    files touched (both new and re-seen). Calls `progress("discover", n)`
    every 25 files."""
    total = 0
    for root_id in root_ids:
        root = store.get_root(root_id)
        if root is None:
            continue
        for d in walk_root(root["path"], root.get("include_glob") or "", root.get("exclude_glob") or ""):
            if cancel and cancel():
                return total
            store.upsert_video_discovered(
                root_id=root_id,
                abs_path=d.abs_path,
                rel_path=d.rel_path,
                size_bytes=d.size_bytes,
                mtime=d.mtime,
            )
            total += 1
            if progress and total % 25 == 0:
                progress("discover", total)
    if progress:
        progress("discover", total)
    return total
