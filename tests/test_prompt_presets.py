"""Tests for prompt_presets.py and the /api/v1/presets endpoints.

Covers:
- Built-in library: at least 8 styles + 5 genres, each has a positive prompt.
- id validation (safe characters, length) and path-traversal guard.
- list_user_presets ignores malformed JSON, picks up valid entries.
- save_user_preset slug-generates when id missing, clamps fields, writes
  the JSON file in PRESETS_DIR.
- delete_user_preset only removes files under PRESETS_DIR.
- get_preset returns built-in hits by id.
- Endpoints respond with correct shapes.
"""
from __future__ import annotations

import json
import os

import pytest


def _reload_with_dir(tmp_path, monkeypatch):
    """Reload prompt_presets with PRESETS_DIR pointing at a temp folder."""
    import importlib
    import sys as _sys
    monkeypatch.setenv("STUDIOLITE_PRESETS_DIR", str(tmp_path))
    if "prompt_presets" in _sys.modules:
        del _sys.modules["prompt_presets"]
    import prompt_presets
    importlib.reload(prompt_presets)
    return prompt_presets


# ---------------------------------------------------------------------------
# Built-in library
# ---------------------------------------------------------------------------


def test_builtin_library_meets_acceptance_counts():
    from prompt_presets import list_builtin_presets
    presets = list_builtin_presets()
    styles = [p for p in presets if p["category"] == "style"]
    genres = [p for p in presets if p["category"] == "genre"]
    assert len(styles) >= 8  # acceptance says "at least 8 built-in style presets"
    assert len(genres) >= 5  # "at least 5 genre prompt templates"
    for p in presets:
        assert p["positive_prompt"]
        assert p["builtin"] is True


def test_builtin_categories_lists_all_distinct():
    from prompt_presets import list_builtin_categories
    cats = list_builtin_categories()
    assert "style" in cats
    assert "genre" in cats
    assert "shot" in cats
    assert "lighting" in cats


def test_get_preset_returns_builtin():
    from prompt_presets import get_preset
    p = get_preset("style_cinematic")
    assert p is not None
    assert p["name"] == "Cinematic Film"


def test_get_preset_unknown_returns_none(tmp_path, monkeypatch):
    pp = _reload_with_dir(tmp_path, monkeypatch)
    assert pp.get_preset("nope_xxx") is None


# ---------------------------------------------------------------------------
# id guard / path traversal
# ---------------------------------------------------------------------------


def test_preset_path_rejects_slashes(tmp_path, monkeypatch):
    pp = _reload_with_dir(tmp_path, monkeypatch)
    # Traversal attempts return None instead of a resolvable path.
    assert pp._preset_path("../etc/passwd") is None
    assert pp._preset_path("..") is None
    assert pp._preset_path("a/b") is None
    assert pp._preset_path("") is None


def test_preset_path_accepts_safe_id(tmp_path, monkeypatch):
    pp = _reload_with_dir(tmp_path, monkeypatch)
    p = pp._preset_path("custom_abc-123")
    assert p is not None
    assert p.endswith("custom_abc-123.json")


# ---------------------------------------------------------------------------
# save / list / delete
# ---------------------------------------------------------------------------


def test_save_user_preset_writes_json(tmp_path, monkeypatch):
    pp = _reload_with_dir(tmp_path, monkeypatch)
    data = pp.save_user_preset({
        "name": "My Look",
        "positive_prompt": "warm, grainy",
        "category": "custom",
    })
    assert data["name"] == "My Look"
    assert data["id"].startswith("custom_my_look_")
    path = os.path.join(str(tmp_path), f"{data['id']}.json")
    assert os.path.isfile(path)
    with open(path, encoding="utf-8") as f:
        on_disk = json.load(f)
    assert on_disk["positive_prompt"] == "warm, grainy"


def test_save_user_preset_requires_name(tmp_path, monkeypatch):
    pp = _reload_with_dir(tmp_path, monkeypatch)
    with pytest.raises(ValueError):
        pp.save_user_preset({"name": ""})


def test_save_user_preset_rejects_unsafe_id(tmp_path, monkeypatch):
    pp = _reload_with_dir(tmp_path, monkeypatch)
    with pytest.raises(ValueError):
        pp.save_user_preset({"name": "ok", "id": "../etc/passwd"})


def test_save_user_preset_clamps_field_lengths(tmp_path, monkeypatch):
    pp = _reload_with_dir(tmp_path, monkeypatch)
    huge = "x" * 10000
    data = pp.save_user_preset({
        "name": huge, "positive_prompt": huge, "negative_prompt": huge,
    })
    assert len(data["name"]) <= 120
    assert len(data["positive_prompt"]) <= 4000
    assert len(data["negative_prompt"]) <= 4000


def test_save_user_preset_coerces_bad_recommended_settings(tmp_path, monkeypatch):
    pp = _reload_with_dir(tmp_path, monkeypatch)
    data = pp.save_user_preset({
        "name": "foo",
        "recommended_settings": "not a dict",
    })
    assert data["recommended_settings"] == {}


def test_list_user_presets_skips_malformed(tmp_path, monkeypatch):
    pp = _reload_with_dir(tmp_path, monkeypatch)
    # One valid
    pp.save_user_preset({"name": "Good", "positive_prompt": "p"})
    # One malformed
    bad = tmp_path / "broken.json"
    bad.write_text("{{ not valid json")
    # One with wrong shape
    noshape = tmp_path / "noshape.json"
    noshape.write_text(json.dumps({"hello": "world"}))

    rows = pp.list_user_presets()
    assert len(rows) == 1
    assert rows[0]["name"] == "Good"


def test_delete_user_preset(tmp_path, monkeypatch):
    pp = _reload_with_dir(tmp_path, monkeypatch)
    data = pp.save_user_preset({"name": "throwaway"})
    assert pp.delete_user_preset(data["id"]) is True
    assert pp.delete_user_preset(data["id"]) is False  # idempotent-ish
    assert pp.delete_user_preset("..escape") is False


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


def _fresh_api(monkeypatch, tmp_path):
    import importlib
    import sys as _sys
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("STUDIOLITE_AUTH", "off")
    presets_dir = tmp_path / "presets_work"
    presets_dir.mkdir()
    monkeypatch.setenv("STUDIOLITE_PRESETS_DIR", str(presets_dir))
    for name in list(_sys.modules):
        if name in ("api_server", "prompt_presets") or name.startswith("api_server."):
            del _sys.modules[name]
    api = importlib.import_module("api_server")
    from fastapi.testclient import TestClient
    return api, TestClient(api.app)


def test_presets_endpoint_lists_builtins(tmp_path, monkeypatch):
    _api, client = _fresh_api(monkeypatch, tmp_path)
    r = client.get("/api/v1/presets")
    assert r.status_code == 200
    data = r.json()
    assert "builtin" in data
    assert "user" in data
    assert len(data["builtin"]) >= 13
    ids = {p["id"] for p in data["builtin"]}
    assert "style_cinematic" in ids


def test_presets_endpoint_save_and_delete(tmp_path, monkeypatch):
    _api, client = _fresh_api(monkeypatch, tmp_path)
    r = client.post("/api/v1/presets", json={
        "name": "Test look",
        "positive_prompt": "neon, grain",
        "category": "custom",
    })
    assert r.status_code == 200
    body = r.json()
    pid = body["id"]

    listing = client.get("/api/v1/presets").json()
    user_ids = {p["id"] for p in listing["user"]}
    assert pid in user_ids

    r = client.delete(f"/api/v1/presets/{pid}")
    assert r.status_code == 200
    assert r.json().get("deleted") is True


def test_presets_endpoint_rejects_bad_name(tmp_path, monkeypatch):
    _api, client = _fresh_api(monkeypatch, tmp_path)
    r = client.post("/api/v1/presets", json={"name": ""})
    assert r.status_code == 422
