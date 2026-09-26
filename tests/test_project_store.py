"""Coverage for the SQLite index + artifact-version-history store."""
from __future__ import annotations

import gzip
import json
import os
import time

import pytest

from filmmaker.project_store import ProjectStore, _COMPRESS_THRESHOLD


def _make_store(tmp_path):
    return ProjectStore(str(tmp_path / "projects.sqlite3"))


class TestIndex:
    def test_upsert_and_list(self, tmp_path):
        store = _make_store(tmp_path)
        store.upsert_project("p1", "First", "brief 1", 1.0, 1.5)
        store.upsert_project("p2", "Second", "brief 2", 2.0, 2.5)
        rows = store.list_indexed_projects()
        # Newest updated_at first
        ids = [r["id"] for r in rows]
        assert ids == ["p2", "p1"]

    def test_upsert_conflict_updates_title(self, tmp_path):
        store = _make_store(tmp_path)
        store.upsert_project("p1", "Old", "brief", 1.0, 1.5)
        store.upsert_project("p1", "New", "brief 2", 1.0, 3.0)
        rows = store.list_indexed_projects()
        assert len(rows) == 1
        assert rows[0]["title"] == "New"
        assert rows[0]["brief"] == "brief 2"
        assert rows[0]["updated_at"] == 3.0

    def test_delete_removes_index_and_versions(self, tmp_path):
        store = _make_store(tmp_path)
        store.upsert_project("p1", "T", "b", 0.0, 0.0)
        store.snapshot_artifact("p1", "producer", {"loglines": ["a"]})
        assert len(store.list_versions("p1", "producer")) == 1
        store.delete_project("p1")
        assert store.list_indexed_projects() == []
        assert store.list_versions("p1", "producer") == []


class TestVersions:
    def test_snapshot_assigns_incrementing_numbers(self, tmp_path):
        store = _make_store(tmp_path)
        v1 = store.snapshot_artifact("p1", "producer", {"take": 1})
        v2 = store.snapshot_artifact("p1", "producer", {"take": 2})
        v3 = store.snapshot_artifact("p1", "producer", {"take": 3})
        assert (v1, v2, v3) == (1, 2, 3)

    def test_versions_isolated_per_stage(self, tmp_path):
        store = _make_store(tmp_path)
        store.snapshot_artifact("p1", "producer", {"take": 1})
        store.snapshot_artifact("p1", "screenwriter", {"draft": 1})
        prod = store.list_versions("p1", "producer")
        script = store.list_versions("p1", "screenwriter")
        assert [v["version_no"] for v in prod] == [1]
        assert [v["version_no"] for v in script] == [1]

    def test_list_versions_returns_newest_first(self, tmp_path):
        store = _make_store(tmp_path)
        for i in range(1, 6):
            store.snapshot_artifact("p1", "producer", {"take": i})
        rows = store.list_versions("p1", "producer")
        assert [r["version_no"] for r in rows] == [5, 4, 3, 2, 1]

    def test_load_version_roundtrips(self, tmp_path):
        store = _make_store(tmp_path)
        payload = {"loglines": [{"title": "A"}], "chosen": 0}
        v = store.snapshot_artifact("p1", "producer", payload)
        assert store.load_version("p1", "producer", v) == payload

    def test_load_missing_version_returns_none(self, tmp_path):
        store = _make_store(tmp_path)
        assert store.load_version("p1", "producer", 42) is None

    def test_large_payload_is_compressed(self, tmp_path):
        store = _make_store(tmp_path)
        big = {"blob": "x" * (_COMPRESS_THRESHOLD * 2)}
        v = store.snapshot_artifact("p1", "producer", big)
        rows = store.list_versions("p1", "producer")
        assert rows[0]["compressed"] is True
        # Round-trip still yields the same dict
        assert store.load_version("p1", "producer", v) == big

    def test_small_payload_not_compressed(self, tmp_path):
        store = _make_store(tmp_path)
        store.snapshot_artifact("p1", "producer", {"tiny": 1})
        rows = store.list_versions("p1", "producer")
        assert rows[0]["compressed"] is False

    def test_prune_keeps_newest(self, tmp_path):
        store = _make_store(tmp_path)
        for i in range(1, 11):
            store.snapshot_artifact("p1", "producer", {"take": i})
        removed = store.prune_versions("p1", "producer", keep=3)
        assert removed == 7
        rows = store.list_versions("p1", "producer")
        nums = [r["version_no"] for r in rows]
        assert nums == [10, 9, 8]


class TestIntegrationWithProject:
    def test_write_artifact_creates_snapshot(self, tmp_path, monkeypatch):
        from filmmaker.project import Project, ProjectConfig

        # Point the class-level store at our tmp DB so we don't touch a
        # real ~/.mp/projects.sqlite3.
        store = ProjectStore(str(tmp_path / "projects.sqlite3"))
        Project.set_store(store)
        try:
            p = Project.create(str(tmp_path), brief="b", title="T",
                                config=ProjectConfig())
            p.write_artifact("producer", {"loglines": ["a"]})
            p.write_artifact("producer", {"loglines": ["b"]})
            versions = p.list_versions("producer")
            assert [v["version_no"] for v in versions] == [2, 1]
        finally:
            Project.set_store(None)

    def test_restore_version(self, tmp_path):
        from filmmaker.project import Project, ProjectConfig

        store = ProjectStore(str(tmp_path / "projects.sqlite3"))
        Project.set_store(store)
        try:
            p = Project.create(str(tmp_path), brief="b", title="T",
                                config=ProjectConfig())
            p.write_artifact("producer", {"loglines": ["a"]})
            p.write_artifact("producer", {"loglines": ["b"]})
            # Restore v1
            ok = p.restore_version("producer", 1)
            assert ok is True
            assert p.read_artifact("producer") == {"loglines": ["a"]}
            # Pre-restore snapshot means we now have three versions - 
            # v1, v2, v3 (pre-restore of v1).
            versions = p.list_versions("producer")
            assert len(versions) >= 3
        finally:
            Project.set_store(None)

    def test_meta_save_updates_index(self, tmp_path):
        from filmmaker.project import Project, ProjectConfig

        store = ProjectStore(str(tmp_path / "projects.sqlite3"))
        Project.set_store(store)
        try:
            p = Project.create(str(tmp_path), brief="brief", title="Original",
                                config=ProjectConfig())
            rows = store.list_indexed_projects()
            assert rows[0]["title"] == "Original"

            meta = p.meta
            meta.title = "Renamed"
            p.save_meta(meta)
            rows = store.list_indexed_projects()
            assert rows[0]["title"] == "Renamed"
        finally:
            Project.set_store(None)


class TestFallback:
    def test_bad_path_returns_empty_reads_silently(self, tmp_path):
        # Point at a directory to force sqlite to refuse to open
        bad = str(tmp_path / "dir")
        os.makedirs(bad, exist_ok=True)
        store = ProjectStore(bad)
        assert store._conn is None
        # Every read path returns the empty answer without raising
        assert store.list_indexed_projects() == []
        assert store.list_versions("p", "s") == []
        assert store.load_version("p", "s", 1) is None
        # Every write path is a silent no-op
        store.upsert_project("p", "t", "b", 0.0, 0.0)
        store.snapshot_artifact("p", "s", {"x": 1})
