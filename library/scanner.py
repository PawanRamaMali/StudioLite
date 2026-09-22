"""Directory walker for the library.

Two responsibilities: iterate videos under a root, and reconcile the store
with what's on disk (mark rows missing when a file disappears). Everything
IO — no hashing or probing here."""
from __future__ import annotations

import fnmatch
import logging
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional

from .store import LibraryStore

logger = logging.getLogger("studiolite.library.scanner")


# Image formats Pillow can open, plus a few extras that fall through to
# a header probe. Same rule as videos: cast a wide net so we don't
# silently skip a file the user cares about.
IMAGE_EXTS = frozenset({
    ".jpg", ".jpeg", ".jpe", ".jfif",
    ".png", ".apng",
    ".webp", ".avif", ".heic", ".heif",
    ".gif",
    ".bmp", ".dib",
    ".tif", ".tiff",
    ".ico", ".cur",
    ".ppm", ".pgm", ".pbm", ".pnm",
    ".tga", ".icns",
    # Raw camera formats — Pillow reads header at minimum
    ".arw", ".cr2", ".cr3", ".nef", ".nrw", ".orf", ".pef",
    ".raf", ".rw2", ".dng", ".srw",
    # Legacy / niche
    ".jp2", ".j2k", ".jpx",
    ".psd", ".xcf",
})


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


MEDIA_EXTS = VIDEO_EXTS | IMAGE_EXTS


def kind_for(path: str) -> Optional[str]:
    ext = os.path.splitext(path)[1].lower()
    if ext in VIDEO_EXTS: return "video"
    if ext in IMAGE_EXTS: return "image"
    return None


@dataclass
class Discovered:
    root_id: Optional[int]
    abs_path: str
    rel_path: str
    size_bytes: int
    mtime: float
    kind: str = "video"


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
            if ext not in MEDIA_EXTS:
                continue
            kind = "image" if ext in IMAGE_EXTS else "video"
            abs_path = os.path.join(dirpath, name)
            if not _matches(abs_path, include_glob, exclude_glob):
                continue
            try:
                st = os.stat(abs_path)
            except OSError:
                continue
            rel_path = os.path.relpath(abs_path, root_abs).replace("\\", "/")
            yield Discovered(root_id=None, abs_path=abs_path, rel_path=rel_path,
                             size_bytes=st.st_size, mtime=st.st_mtime, kind=kind)
            count += 1
            if count >= max_files:
                logger.warning("walk_root %s hit max_files=%d; stopping", root_abs, max_files)
                return


def is_directory_effectively_empty(dirpath: str) -> bool:
    """True when the directory has no files and no non-empty subdirectories.
    A folder full of Thumbs.db, .DS_Store, and desktop.ini is considered
    empty — those junk files shouldn't stop cleanup."""
    JUNK = {"thumbs.db", ".ds_store", "desktop.ini", ".directory"}
    try:
        for entry in os.scandir(dirpath):
            if entry.is_file(follow_symlinks=False):
                if entry.name.lower() in JUNK:
                    continue
                return False
            elif entry.is_dir(follow_symlinks=False):
                if not is_directory_effectively_empty(entry.path):
                    return False
    except OSError:
        return False
    return True


def cleanup_empty_folders(root_paths: List[str], *, dry_run: bool = False,
                          protect_root: bool = True) -> Dict[str, Any]:
    """Walk each root bottom-up and remove effectively-empty directories.

    - `protect_root` (default True) never deletes the root path itself,
      even if it becomes empty. The user asked to organize inside those
      folders, not to delete the containers they specified.
    - Junk sentinels (Thumbs.db, .DS_Store, desktop.ini, .directory)
      count as "empty" for the purpose of cleanup, and get removed
      alongside the folder.
    """
    removed: List[str] = []
    remaining_empty: List[str] = []
    JUNK = {"thumbs.db", ".ds_store", "desktop.ini", ".directory"}
    for root in root_paths:
        root_abs = os.path.abspath(root)
        if not os.path.isdir(root_abs):
            continue
        # Bottom-up so parents can become empty as we remove children.
        for dirpath, _dirs, _files in os.walk(root_abs, topdown=False, followlinks=False):
            if protect_root and os.path.abspath(dirpath) == root_abs:
                continue
            if not is_directory_effectively_empty(dirpath):
                continue
            if dry_run:
                remaining_empty.append(dirpath)
                continue
            # Clear the junk sentinels first, then rmdir.
            try:
                for entry in os.scandir(dirpath):
                    if entry.is_file(follow_symlinks=False) and entry.name.lower() in JUNK:
                        try: os.remove(entry.path)
                        except OSError: pass
                os.rmdir(dirpath)
                removed.append(dirpath)
            except OSError as e:
                logger.info("could not remove empty %s: %s", dirpath, e)
                remaining_empty.append(dirpath)
    return {
        "removed": removed,
        "remaining_empty": remaining_empty,
        "count_removed": len(removed),
        "dry_run": dry_run,
    }


def cleanup_empty_parents(paths_deleted: List[str], root_paths: List[str],
                          *, dry_run: bool = False) -> Dict[str, Any]:
    """Given a list of files just deleted, walk each unique parent folder
    upward — removing it if empty — and stop at the first non-empty
    ancestor or when we cross a library root boundary.
    """
    if not paths_deleted:
        return {"removed": [], "count_removed": 0, "dry_run": dry_run}
    roots_abs = {os.path.abspath(r).rstrip("/\\") for r in root_paths}
    seen: set = set()
    parents: List[str] = []
    for p in paths_deleted:
        parent = os.path.dirname(os.path.abspath(p))
        if parent not in seen:
            seen.add(parent)
            parents.append(parent)
    removed: List[str] = []
    JUNK = {"thumbs.db", ".ds_store", "desktop.ini", ".directory"}
    for parent in parents:
        cur = parent
        while cur and os.path.abspath(cur).rstrip("/\\") not in roots_abs:
            if not os.path.isdir(cur):
                break
            if not is_directory_effectively_empty(cur):
                break
            if dry_run:
                removed.append(cur)
                cur = os.path.dirname(cur)
                continue
            try:
                for entry in os.scandir(cur):
                    if entry.is_file(follow_symlinks=False) and entry.name.lower() in JUNK:
                        try: os.remove(entry.path)
                        except OSError: pass
                os.rmdir(cur)
                removed.append(cur)
            except OSError:
                break
            cur = os.path.dirname(cur)
    return {"removed": removed, "count_removed": len(removed), "dry_run": dry_run}


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
                kind=d.kind,
            )
            total += 1
            if progress and total % 25 == 0:
                progress("discover", total)
    if progress:
        progress("discover", total)
    return total
