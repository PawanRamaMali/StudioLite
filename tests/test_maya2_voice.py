"""Tests for the Maya2-native voice backend wiring.

Maya2-native is a multilingual (12 Indian languages + English) autoregressive
TTS from maya-research. Two activation paths:

  * Native Python model (``maya3_tts.Maya3TTS``) - needs Linux + CUDA 12.8
    and the local weights at ``~/models/maya2-native``.
  * HTTP server - OpenAI-compatible ``POST /v1/audio/speech`` endpoint,
    activated by setting ``MAYA2_URL=http://host:port``.

These tests cover: loader branching (non-maya2 backend returns None, HTTP
path picks up ``MAYA2_URL``, native path returns None when weights are
missing), the voice/language cascade through ``voice_cast``, and the HTTP
request shape against a stubbed ``requests.post``.
"""
from __future__ import annotations

import os
from unittest.mock import patch, MagicMock

import pytest

from filmmaker import agents


class _StubProject:
    """Minimum Project stand-in for the voice loader. The wrap only reads
    ``voice_cast`` and ``artifacts_dir`` so we hand-roll both."""

    def __init__(self, tmp_path, cast=None):
        self.dir = str(tmp_path)
        self.artifacts_dir = str(tmp_path / "artifacts")
        os.makedirs(self.artifacts_dir, exist_ok=True)
        self._artifacts = {"voice_cast": {"cast": cast or {}}}

        class _Cfg:
            voice_backend = "maya2"
            per_stage: dict = {}
        class _Meta:
            config = _Cfg()
        self.meta = _Meta()

    def read_artifact(self, key):
        return self._artifacts.get(key)


# ---------------------------------------------------------------------------
# Loader branching
# ---------------------------------------------------------------------------

def test_loader_returns_none_when_backend_is_not_maya2(tmp_path):
    assert agents._load_maya2_if_requested("piper", _StubProject(tmp_path)) is None
    assert agents._load_maya2_if_requested("xtts", _StubProject(tmp_path)) is None
    assert agents._load_maya2_if_requested("", _StubProject(tmp_path)) is None


def test_loader_picks_http_mode_when_maya2_url_env_set(tmp_path, monkeypatch):
    monkeypatch.setenv("MAYA2_URL", "http://example.local:9000")
    wrap = agents._load_maya2_if_requested("maya2", _StubProject(tmp_path))
    assert wrap is not None
    assert wrap._mode == "http"
    # Trailing slash must be stripped so endpoint URL concat stays clean.
    assert wrap._http_url == "http://example.local:9000"


def test_loader_strips_trailing_slash_from_url(tmp_path, monkeypatch):
    monkeypatch.setenv("MAYA2_URL", "http://host/")
    wrap = agents._load_maya2_if_requested("maya2", _StubProject(tmp_path))
    assert wrap is not None
    assert wrap._http_url == "http://host"


def test_loader_falls_back_when_package_missing_and_no_url(tmp_path, monkeypatch):
    """Native path requires the ``maya3_tts`` package, which isn't installed
    on Windows. With no MAYA2_URL set and no package importable, the loader
    must return None so the voice stage falls through to IndexTTS/XTTS/Piper."""
    monkeypatch.delenv("MAYA2_URL", raising=False)
    # Force the import to fail even if the package were somehow present.
    import builtins
    real_import = builtins.__import__

    def _blocked(name, *a, **kw):
        if name == "maya3_tts":
            raise ImportError("blocked for test")
        return real_import(name, *a, **kw)

    with patch.object(builtins, "__import__", side_effect=_blocked):
        wrap = agents._load_maya2_if_requested("maya2", _StubProject(tmp_path))
    assert wrap is None


# ---------------------------------------------------------------------------
# Voice & language cascade
# ---------------------------------------------------------------------------

def test_voice_cascade_falls_back_to_defaults_when_cast_empty(tmp_path):
    wrap = agents._Maya2Wrap(_StubProject(tmp_path), mode="http",
                              http_url="http://x")
    v, l = wrap._voice_and_lang_for("MARGO")
    assert v == agents._MAYA2_DEFAULT_VOICE  # "Ahana"
    assert l == agents._MAYA2_DEFAULT_LANG   # "en"


def test_voice_cascade_prefers_exact_cast_match(tmp_path):
    cast = {"MARGO": {"maya2_voice": "Priya", "maya2_lang": "hi"},
            "KAI":   {"maya2_voice": "Rohan", "maya2_lang": "en"}}
    wrap = agents._Maya2Wrap(_StubProject(tmp_path, cast=cast),
                              mode="http", http_url="http://x")
    assert wrap._voice_and_lang_for("MARGO") == ("Priya", "hi")
    assert wrap._voice_and_lang_for("KAI") == ("Rohan", "en")


def test_voice_cascade_matches_first_token_when_speaker_is_fuller_name(tmp_path):
    """breakdown often carries a fuller name ('Margo Chen') than the
    screenplay cue ('MARGO'); first-token fallback catches this."""
    cast = {"Margo Chen": {"maya2_voice": "Priya", "maya2_lang": "hi"}}
    wrap = agents._Maya2Wrap(_StubProject(tmp_path, cast=cast),
                              mode="http", http_url="http://x")
    assert wrap._voice_and_lang_for("MARGO") == ("Priya", "hi")


def test_voice_cascade_accepts_voice_field_when_maya2_voice_missing(tmp_path):
    """A project built pre-Maya2 only has a generic ``voice`` field. Use it
    rather than silently falling back to the global default."""
    cast = {"MARGO": {"voice": "Priya"}}
    wrap = agents._Maya2Wrap(_StubProject(tmp_path, cast=cast),
                              mode="http", http_url="http://x")
    v, l = wrap._voice_and_lang_for("MARGO")
    assert v == "Priya"
    # No maya2_lang → default lang.
    assert l == agents._MAYA2_DEFAULT_LANG


# ---------------------------------------------------------------------------
# HTTP request shape
# ---------------------------------------------------------------------------

def _resp_bytes(status: int, content: bytes = b"RIFF" + b"\x00" * 100) -> MagicMock:
    r = MagicMock()
    r.status_code = status
    r.content = content
    r.text = ""
    return r


def test_http_synthesize_posts_to_speech_endpoint(tmp_path):
    """The HTTP mode must hit ``/v1/audio/speech`` with an OpenAI-compatible
    payload plus Maya2's language + response_format extensions."""
    cast = {"MARGO": {"maya2_voice": "Priya", "maya2_lang": "hi"}}
    wrap = agents._Maya2Wrap(_StubProject(tmp_path, cast=cast),
                              mode="http", http_url="http://host:8000")
    out = tmp_path / "out.wav"
    with patch("requests.post", return_value=_resp_bytes(200)) as m:
        wrap.synthesize("नमस्ते", str(out), speaker="MARGO")

    args, kwargs = m.call_args
    assert args[0] == "http://host:8000/v1/audio/speech"
    body = kwargs["json"]
    assert body["input"] == "नमस्ते"
    assert body["voice"] == "Priya"
    assert body["language"] == "hi"
    assert body["response_format"] == "wav"
    assert body["model"] == "maya2-native"
    # Server returned bytes should land on disk unchanged.
    assert out.exists() and out.read_bytes().startswith(b"RIFF")


def test_http_synthesize_raises_on_non_200(tmp_path):
    wrap = agents._Maya2Wrap(_StubProject(tmp_path), mode="http",
                              http_url="http://host")
    out = tmp_path / "out.wav"
    err = MagicMock()
    err.status_code = 500
    err.content = b""
    err.text = "boom"
    with patch("requests.post", return_value=err):
        with pytest.raises(RuntimeError) as exc:
            wrap.synthesize("hi", str(out), speaker="MARGO")
    msg = str(exc.value)
    assert "500" in msg
    # The error must include the server's message so operator sees why.
    assert "boom" in msg
    # And must NOT have written a partial file to disk.
    assert not out.exists()
