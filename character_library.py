"""Persistent character library.

A JSON-backed store that maps a character name to their appearance text, a
portrait URL, and an IP-Adapter id. Pure CRUD; the Story Mode runner auto-pulls
appearance text from here when a scene references a character by name but no
``CharacterDef`` was passed in the request.

File format (characters.json):
    {
      "<safe_name>": {
        "name": "<display name>",
        "appearance": "<detailed look text>",
        "description": "<short traits>",
        "style": "photorealistic",
        "portrait_url": "/static/portraits/<file>.png",
        "ip_adapter_id": "<safe_name>",
        "updated_at": <unix ts>
      },
      ...
    }
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Optional

_SAFE_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]{1,40}$")


def _library_path() -> str:
    override = os.environ.get("STUDIOLITE_CHAR_LIBRARY")
    if override:
        return override
    root = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(root, ".mp", "characters.json")


def _safe_id(name: str) -> Optional[str]:
    """Return a storage-safe id or None if the name resolves to nothing usable."""
    if not name:
        return None
    s = re.sub(r"[^a-zA-Z0-9_-]", "_", str(name).strip().lower())[:40]
    s = s.strip("_-")
    if not s or not _SAFE_NAME_RE.match(s):
        return None
    return s


def _load_all() -> dict:
    path = _library_path()
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _save_all(data: dict) -> None:
    path = _library_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def list_characters() -> list[dict]:
    """Return every stored character as a list, newest first."""
    data = _load_all()
    out = list(data.values())
    out.sort(key=lambda d: d.get("updated_at", 0), reverse=True)
    return out


def get_character(name: str) -> Optional[dict]:
    """Look up by display name OR safe id (case-insensitive)."""
    sid = _safe_id(name)
    if sid is None:
        return None
    data = _load_all()
    if sid in data:
        return data[sid]
    # Fallback: scan by display-name match (case-insensitive) for user ergonomics
    target = str(name).strip().lower()
    for entry in data.values():
        if str(entry.get("name", "")).strip().lower() == target:
            return entry
    return None


def save_character(name: str, appearance: str, description: str = "",
                   style: str = "photorealistic", portrait_url: str = "",
                   ip_adapter_id: Optional[str] = None) -> Optional[dict]:
    """Insert-or-replace a character record. Returns the stored entry, or None if the name is unusable."""
    sid = _safe_id(name)
    if sid is None:
        return None
    entry = {
        "id": sid,
        "name": str(name).strip()[:80],
        "appearance": str(appearance).strip()[:2000],
        "description": str(description).strip()[:500],
        "style": str(style).strip()[:40] or "photorealistic",
        "portrait_url": str(portrait_url).strip()[:500],
        "ip_adapter_id": str(ip_adapter_id or sid)[:40],
        "updated_at": int(time.time()),
    }
    data = _load_all()
    data[sid] = entry
    _save_all(data)
    return entry


def delete_character(name: str) -> bool:
    """Remove ``name`` from the library. Returns True iff an entry was removed."""
    sid = _safe_id(name)
    if sid is None:
        return False
    data = _load_all()
    if sid not in data:
        return False
    del data[sid]
    _save_all(data)
    return True


def lookup_appearances(names: list[str]) -> dict[str, str]:
    """Bulk lookup: given a list of scene-character names, return {name: appearance}.

    Names that aren't in the library are skipped (not returned) so the caller
    can tell which ones need prompt-fallback. Preserves the input casing in the
    returned keys so callers can plug it into prompts directly.
    """
    out: dict[str, str] = {}
    for n in names or []:
        entry = get_character(n)
        if entry and entry.get("appearance"):
            out[str(n)] = entry["appearance"]
    return out
