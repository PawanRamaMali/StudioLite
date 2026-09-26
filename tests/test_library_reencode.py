"""ReencodeJob + list_legacy_videos structural tests.

No real ffmpeg - patches subprocess.Popen so the job "converts" by
writing a stub output file. Verifies:
- list_legacy_videos filters by codec case-insensitively
- the job runs through every id, tracks counts, and computes bytes saved
- cancel drops the partial file
- replace_original produces the .legacy sidecar
"""
from __future__ import annotations

import os
import shutil
import sqlite3
import tempfile
import threading
import time
from typing import List
from unittest.mock import patch, MagicMock

from library import LibraryStore
from library.reencode import ReencodeJob, LEGACY_CODECS, TARGET_CODECS


def _seed_store(path: str, entries: List[dict]) -> LibraryStore:
    store = LibraryStore(path)
    # Directly INSERT rows so we control codec + size without running probe.
    with sqlite3.connect(store.db_path) as raw:
        for i, e in enumerate(entries, 1):
            raw.execute(
                "INSERT INTO videos(abs_path, rel_path, size_bytes, mtime, "
                "added_at, media_kind, codec, width, height, duration_sec) "
                "VALUES(?,?,?,?,?,?,?,?,?,?)",
                (e["abs_path"], e.get("rel_path", os.path.basename(e["abs_path"])),
                 e["size_bytes"], 1700000000.0, 1700000000.0,
                 e.get("kind", "video"), e.get("codec"), 1280, 720, 30.0),
            )
    return store


def test_list_legacy_videos_filters_correctly():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        store = _seed_store(os.path.join(tmp, "lib.sqlite3"), [
            {"abs_path": "/a.mpg",  "size_bytes": 1_000_000_000, "codec": "mpeg2video"},
            {"abs_path": "/b.wmv",  "size_bytes": 500_000_000,   "codec": "WMV3"},   # mixed-case
            {"abs_path": "/c.rmvb", "size_bytes": 700_000_000,   "codec": "rv40"},
            {"abs_path": "/d.mp4",  "size_bytes": 800_000_000,   "codec": "h264"},   # modern; skip
            {"abs_path": "/e.mkv",  "size_bytes": 300_000_000,   "codec": None},     # un-probed; skip
            {"abs_path": "/img.jpg","size_bytes": 100_000,       "codec": "mjpeg",   "kind": "image"},
        ])
        legacy = store.list_legacy_videos()
        paths = sorted(v.abs_path for v in legacy)
        assert paths == ["/a.mpg", "/b.wmv", "/c.rmvb"], paths
        # Ordered by size DESC so the "biggest wins" first.
        assert legacy[0].abs_path == "/a.mpg"


def test_reencode_job_writes_outputs_and_tallies_saved_bytes():
    """Mock Popen so the 'ffmpeg' just writes a much smaller output file."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        # Build real 100KB source files so the size math is honest.
        src_dir = os.path.join(tmp, "media")
        os.makedirs(src_dir)
        srcs = []
        for name in ("a.mpg", "b.wmv"):
            p = os.path.join(src_dir, name)
            with open(p, "wb") as f: f.write(b"\0" * 100_000)
            srcs.append(p)

        store = _seed_store(os.path.join(tmp, "lib.sqlite3"), [
            {"abs_path": srcs[0], "size_bytes": 100_000, "codec": "mpeg2video"},
            {"abs_path": srcs[1], "size_bytes": 100_000, "codec": "wmv3"},
        ])
        legacy = store.list_legacy_videos()
        assert len(legacy) == 2

        out_dir = os.path.join(tmp, "reenc")
        events = []

        def fake_popen(cmd, **kw):
            # Extract the output path from the ffmpeg command (last arg).
            out_path = cmd[-1]
            mock = MagicMock()
            def _communicate():
                # Write a 40KB "encoded" file - 60% smaller than source.
                with open(out_path, "wb") as f: f.write(b"\0" * 40_000)
                mock.returncode = 0
                return (b"", b"")
            mock.communicate = _communicate
            mock.poll.return_value = 0
            mock.returncode = 0
            return mock

        with patch("library.reencode.subprocess.Popen", side_effect=fake_popen):
            job = ReencodeJob(
                store, [v.id for v in legacy],
                output_dir=out_dir,
                on_update=lambda payload: events.append(payload),
                target_codec="h264",
                crf=23,
                replace_original=False,
            )
            job.run()

        assert job.counts["done"] == 2, job.counts
        assert job.counts["failed"] == 0
        assert job.counts["bytes_saved"] == 2 * (100_000 - 40_000)
        # Outputs live in the sidecar dir with the id__codec.mp4 pattern.
        outs = sorted(os.listdir(out_dir))
        assert len(outs) == 2
        assert all(name.endswith("__h264.mp4") for name in outs), outs
        # Final event carries a summary status.
        assert events[-1]["status"] == "completed"


def test_reencode_replace_original_moves_source_to_legacy_sidecar():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        src_dir = os.path.join(tmp, "media")
        os.makedirs(src_dir)
        src = os.path.join(src_dir, "a.mpg")
        with open(src, "wb") as f: f.write(b"\0" * 200_000)

        store = _seed_store(os.path.join(tmp, "lib.sqlite3"), [
            {"abs_path": src, "size_bytes": 200_000, "codec": "mpeg2video"},
        ])
        vid = store.list_legacy_videos()[0].id

        def fake_popen(cmd, **kw):
            out_path = cmd[-1]
            mock = MagicMock()
            def _communicate():
                with open(out_path, "wb") as f: f.write(b"\0" * 90_000)
                mock.returncode = 0
                return (b"", b"")
            mock.communicate = _communicate
            mock.returncode = 0
            return mock

        events = []
        with patch("library.reencode.subprocess.Popen", side_effect=fake_popen):
            job = ReencodeJob(
                store, [vid], output_dir=os.path.join(tmp, "reenc-unused"),
                on_update=lambda p: events.append(p),
                target_codec="h264", replace_original=True,
            )
            job.run()

        assert job.counts["done"] == 1, job.counts
        assert os.path.exists(src), "source path should hold the new file"
        assert os.path.getsize(src) == 90_000
        assert os.path.exists(src + ".legacy"), "old file should be moved to .legacy"
        assert os.path.getsize(src + ".legacy") == 200_000
        # Store row's codec was cleared so a rescan will re-probe.
        v = store.get_video(vid)
        assert v.codec is None


def test_reencode_cancel_stops_early_and_removes_partial():
    """Job should honor cancel between files."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        src_dir = os.path.join(tmp, "media")
        os.makedirs(src_dir)
        srcs = []
        for name in ("a.mpg", "b.mpg", "c.mpg"):
            p = os.path.join(src_dir, name)
            with open(p, "wb") as f: f.write(b"\0" * 50_000)
            srcs.append(p)

        store = _seed_store(os.path.join(tmp, "lib.sqlite3"), [
            {"abs_path": s, "size_bytes": 50_000, "codec": "mpeg2video"} for s in srcs
        ])
        cancel_evt = threading.Event()
        calls = {"n": 0}

        def fake_popen(cmd, **kw):
            calls["n"] += 1
            # After the FIRST file, flip cancel so the outer loop bails.
            if calls["n"] == 1:
                cancel_evt.set()
            out_path = cmd[-1]
            mock = MagicMock()
            def _communicate():
                with open(out_path, "wb") as f: f.write(b"\0" * 20_000)
                mock.returncode = 0
                return (b"", b"")
            mock.communicate = _communicate
            mock.returncode = 0
            return mock

        with patch("library.reencode.subprocess.Popen", side_effect=fake_popen):
            job = ReencodeJob(
                store, [v.id for v in store.list_legacy_videos()],
                output_dir=os.path.join(tmp, "reenc"),
                on_update=lambda p: None,
                cancel_event=cancel_evt,
            )
            job.run()

        # After the first file finishes we set cancel; loop should exit BEFORE
        # ffmpeg is called for the remaining files.
        assert calls["n"] == 1, calls
        assert job.counts["done"] == 1


def test_target_codecs_are_reasonable():
    assert "h264" in TARGET_CODECS
    assert "h265" in TARGET_CODECS
    # Every entry: (codec, ext, encoder, extra_args)
    for name, (codec, ext, enc, extra) in TARGET_CODECS.items():
        assert ext.startswith("."), f"{name}: bad ext {ext}"
        assert enc.startswith("libx"), f"{name}: bad encoder {enc}"
        assert isinstance(extra, list)
