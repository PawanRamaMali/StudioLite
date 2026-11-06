"""Tests for character_library.py (JSON-backed character persistence)."""
from __future__ import annotations

import json
import os
import pytest


@pytest.fixture(autouse=True)
def _isolate_library(tmp_path, monkeypatch):
    """Redirect the library file to a fresh path per test."""
    lib = tmp_path / "characters.json"
    monkeypatch.setenv("STUDIOLITE_CHAR_LIBRARY", str(lib))
    yield


def test_save_and_get(monkeypatch):
    import character_library
    entry = character_library.save_character(
        name="Alex Chen",
        appearance="30s man, short dark hair, round glasses, grey hoodie",
        description="software engineer",
    )
    assert entry is not None
    assert entry["id"] == "alex_chen"
    assert "glasses" in entry["appearance"]
    got = character_library.get_character("Alex Chen")
    assert got is not None and got["id"] == "alex_chen"


def test_get_case_insensitive_display_name():
    import character_library
    character_library.save_character(name="Zara", appearance="tall woman")
    assert character_library.get_character("ZARA") is not None
    assert character_library.get_character("zara") is not None


def test_get_missing_returns_none():
    import character_library
    assert character_library.get_character("ghost") is None


def test_save_rejects_unusable_name():
    import character_library
    assert character_library.save_character(name="", appearance="x") is None
    assert character_library.save_character(name="   ", appearance="x") is None
    assert character_library.save_character(name="!!!!", appearance="x") is None


def test_delete_removes_entry():
    import character_library
    character_library.save_character(name="Bob", appearance="old man")
    assert character_library.delete_character("Bob") is True
    assert character_library.get_character("Bob") is None


def test_delete_missing_returns_false():
    import character_library
    assert character_library.delete_character("nope") is False


def test_lookup_appearances_bulk():
    import character_library
    character_library.save_character(name="A", appearance="look-a")
    character_library.save_character(name="B", appearance="look-b")
    found = character_library.lookup_appearances(["A", "B", "ghost"])
    assert found == {"A": "look-a", "B": "look-b"}


def test_lookup_appearances_skips_empty_text():
    import character_library
    # save with empty appearance - lookup should skip it (nothing useful to inject)
    character_library.save_character(name="Hollow", appearance="")
    found = character_library.lookup_appearances(["Hollow"])
    assert "Hollow" not in found


def test_list_newest_first(monkeypatch):
    import character_library
    import time
    character_library.save_character(name="Old", appearance="first")
    # Force a time skip without actually sleeping
    orig_time = character_library.time.time
    monkeypatch.setattr(character_library.time, "time", lambda: orig_time() + 100)
    character_library.save_character(name="New", appearance="second")
    rows = character_library.list_characters()
    assert [r["id"] for r in rows[:2]] == ["new", "old"]


def test_file_corruption_recovers_gracefully(tmp_path, monkeypatch):
    import character_library
    lib = tmp_path / "corrupt.json"
    lib.write_text("{not json at all")
    monkeypatch.setenv("STUDIOLITE_CHAR_LIBRARY", str(lib))
    # Load should return {} rather than raising
    assert character_library.list_characters() == []
    # And a subsequent save should overwrite the corrupt file cleanly
    assert character_library.save_character(name="Ok", appearance="x") is not None
    with open(lib) as f:
        data = json.load(f)
    assert "ok" in data


def test_name_with_special_chars_is_sanitized():
    import character_library
    entry = character_library.save_character(name="Mx. Rōs-9!", appearance="x")
    assert entry is not None
    # Unicode strips to ASCII-safe id
    assert "_" in entry["id"] or "-" in entry["id"]
