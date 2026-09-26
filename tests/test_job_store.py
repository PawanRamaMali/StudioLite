"""Coverage for the SQLite-backed job store: dict-compatibility, persistence
round-trip, and interrupted-job recovery."""
from __future__ import annotations

import os

import pytest

from filmmaker.job_store import PersistentJobStore


def _sample_job(kind="test", status="queued", **extra):
    base = {
        "kind": kind,
        "status": status,
        "progress": 0.0,
        "message": "",
        "result": None,
        "error": None,
        "created_at": 1000.0,
        "params": {},
    }
    base.update(extra)
    return base


class TestDictInterface:
    def test_set_and_get(self, tmp_path):
        store = PersistentJobStore(str(tmp_path / "jobs.sqlite3"))
        store["j1"] = _sample_job()
        assert "j1" in store
        assert store["j1"]["status"] == "queued"
        assert len(store) == 1

    def test_iter_and_items(self, tmp_path):
        store = PersistentJobStore(str(tmp_path / "jobs.sqlite3"))
        store["j1"] = _sample_job()
        store["j2"] = _sample_job(kind="other")
        assert set(store) == {"j1", "j2"}
        kinds = {jid: job["kind"] for jid, job in store.items()}
        assert kinds == {"j1": "test", "j2": "other"}

    def test_get_returns_default(self, tmp_path):
        store = PersistentJobStore(str(tmp_path / "jobs.sqlite3"))
        assert store.get("no", "fallback") == "fallback"

    def test_update_fields(self, tmp_path):
        store = PersistentJobStore(str(tmp_path / "jobs.sqlite3"))
        store["j1"] = _sample_job()
        store.update_fields("j1", status="running", progress=0.3)
        assert store["j1"]["status"] == "running"
        assert store["j1"]["progress"] == 0.3


class TestPersistence:
    def test_reopen_roundtrip(self, tmp_path):
        path = str(tmp_path / "jobs.sqlite3")
        store = PersistentJobStore(path)
        store["j1"] = _sample_job(status="completed", progress=1.0,
                                  result={"path": "video.mp4"})
        store.close()

        store2 = PersistentJobStore(path)
        assert "j1" in store2
        job = store2["j1"]
        assert job["status"] == "completed"
        assert job["progress"] == 1.0
        # JSON blob survives the round-trip
        assert job["result"] == {"path": "video.mp4"}

    def test_reopen_preserves_params(self, tmp_path):
        path = str(tmp_path / "jobs.sqlite3")
        store = PersistentJobStore(path)
        store["j1"] = _sample_job(params={"seed": 42, "prompt": "hi"})
        store.close()

        store2 = PersistentJobStore(path)
        assert store2["j1"]["params"] == {"seed": 42, "prompt": "hi"}


class TestRecovery:
    def test_recovers_running_and_queued_jobs(self, tmp_path):
        path = str(tmp_path / "jobs.sqlite3")
        store = PersistentJobStore(path)
        store["j1"] = _sample_job(status="running")
        store["j2"] = _sample_job(status="queued")
        store["j3"] = _sample_job(status="completed")
        store["j4"] = _sample_job(status="cancelling")
        store.close()

        store2 = PersistentJobStore(path)
        recovered = store2.recover_interrupted_jobs()
        assert set(recovered) == {"j1", "j2", "j4"}
        assert store2["j1"]["status"] == "interrupted"
        assert store2["j2"]["status"] == "interrupted"
        assert store2["j4"]["status"] == "interrupted"
        # Completed jobs stay completed - recovery must not touch them.
        assert store2["j3"]["status"] == "completed"

    def test_recovery_is_idempotent(self, tmp_path):
        path = str(tmp_path / "jobs.sqlite3")
        store = PersistentJobStore(path)
        store["j1"] = _sample_job(status="running")
        store.close()

        store2 = PersistentJobStore(path)
        first = store2.recover_interrupted_jobs()
        second = store2.recover_interrupted_jobs()
        assert first == ["j1"]
        assert second == []  # already flipped; nothing left to recover


class TestFallbackMode:
    def test_bad_path_still_works_in_memory(self, tmp_path):
        """A path we can't open must not crash the process - the store
        falls back to in-memory-only and warns."""
        bad_path = os.path.join(str(tmp_path), "nonexistent-dir", "jobs.sqlite3")
        # We deliberately don't create the parent directory. The store
        # will try, succeed (makedirs is exist_ok=True) and open. So
        # to actually exercise the fallback we point it at something
        # that sqlite refuses to open - a directory path.
        os.makedirs(bad_path, exist_ok=True)
        store = PersistentJobStore(bad_path)
        # Fallback path: no connection, but the dict interface still works.
        assert store._conn is None
        store["j1"] = _sample_job()
        assert "j1" in store
