"""Tests for the shared HF Hub download wrapper (hf_download.py).

Covers:
- hf_token() resolves from each supported env var in priority order and
  falls back to the on-disk CLI cache when no env var is set.
- hf_download / hf_snapshot pass the resolved token through to
  huggingface_hub by default, and respect an explicit token kwarg.
- Rate-limit / auth style failures (HTTP 429 / 401 / 403, or messages
  that mention "rate limit" / "429" / "Too Many Requests") get re-raised
  as HfRateLimitError with the HF_TOKEN hint.
- Other failures pass through unchanged - we never swallow surprises.
"""
from __future__ import annotations

import os
import pytest

import hf_download
from hf_download import HfRateLimitError, _is_rate_limit, hf_token


# ---------------------------------------------------------------------------
# hf_token()
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_token_env(monkeypatch, tmp_path):
    """Unset every token env var and point HOME at a clean temp dir so the
    on-disk CLI cache cannot leak this user's real token into a test."""
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        monkeypatch.delenv(key, raising=False)
    # Point the home dir at an empty tmp dir so ~/.cache/huggingface/token
    # does not exist, even if the real one does.
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    yield


def test_hf_token_returns_none_when_nothing_set():
    assert hf_token() is None


@pytest.mark.parametrize("key", [
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "HUGGINGFACE_HUB_TOKEN",
])
def test_hf_token_reads_each_env_var(monkeypatch, key):
    monkeypatch.setenv(key, "tok-abc")
    assert hf_token() == "tok-abc"


def test_hf_token_env_wins_over_cache(monkeypatch, tmp_path):
    cache = tmp_path / ".cache" / "huggingface"
    cache.mkdir(parents=True)
    (cache / "token").write_text("from-disk", encoding="utf-8")
    monkeypatch.setenv("HF_TOKEN", "from-env")
    assert hf_token() == "from-env"


def test_hf_token_falls_back_to_cache_file(monkeypatch, tmp_path):
    cache = tmp_path / ".cache" / "huggingface"
    cache.mkdir(parents=True)
    (cache / "token").write_text("from-disk\n", encoding="utf-8")
    assert hf_token() == "from-disk"


def test_hf_token_strips_whitespace_only_env(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "   ")
    assert hf_token() is None


# ---------------------------------------------------------------------------
# _is_rate_limit()
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, status_code):
        self.status_code = status_code


class _FakeHttpError(Exception):
    def __init__(self, status_code, message=""):
        super().__init__(message or f"HTTP {status_code}")
        self.response = _FakeResponse(status_code)


@pytest.mark.parametrize("code", [401, 403, 429])
def test_is_rate_limit_on_known_status_codes(code):
    assert _is_rate_limit(_FakeHttpError(code))


def test_is_rate_limit_on_200_is_false():
    assert not _is_rate_limit(_FakeHttpError(200))


@pytest.mark.parametrize("msg", [
    "429 Client Error: Too Many Requests",
    "You have hit the rate limit",
    "Unauthorized",
    "FORBIDDEN access to repo",
])
def test_is_rate_limit_on_message_patterns(msg):
    assert _is_rate_limit(RuntimeError(msg))


def test_is_rate_limit_on_unrelated_message_is_false():
    assert not _is_rate_limit(RuntimeError("disk full"))


# ---------------------------------------------------------------------------
# hf_download / hf_snapshot behavior
# ---------------------------------------------------------------------------


def test_hf_download_passes_token_from_env(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "the-token")
    seen = {}

    def fake(*, repo_id, filename, **kwargs):
        seen.update(kwargs, repo_id=repo_id, filename=filename)
        return "/tmp/fake.bin"

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake, raising=True)

    out = hf_download.hf_download("owner/repo", "weights.bin")
    assert out == "/tmp/fake.bin"
    assert seen["token"] == "the-token"
    assert seen["repo_id"] == "owner/repo"
    assert seen["filename"] == "weights.bin"


def test_hf_download_explicit_token_wins_over_env(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "from-env")
    seen = {}

    def fake(*, repo_id, filename, **kwargs):
        seen.update(kwargs)
        return "/tmp/fake.bin"

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake, raising=True)
    hf_download.hf_download("owner/repo", "w.bin", token="caller-wins")
    assert seen["token"] == "caller-wins"


def test_hf_download_rate_limit_wraps_as_hfratelimiterror(monkeypatch):
    def fake(*, repo_id, filename, **kwargs):
        raise _FakeHttpError(429, "Too Many Requests")

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake, raising=True)

    with pytest.raises(HfRateLimitError) as exc:
        hf_download.hf_download("owner/repo", "w.bin")
    msg = str(exc.value)
    assert "HF_TOKEN" in msg
    assert "owner/repo" in msg
    assert "w.bin" in msg
    assert isinstance(exc.value.__cause__, _FakeHttpError)


def test_hf_download_message_only_429_still_wraps(monkeypatch):
    def fake(*, repo_id, filename, **kwargs):
        raise RuntimeError("429 Client Error: Too Many Requests for url: ...")

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake, raising=True)
    with pytest.raises(HfRateLimitError):
        hf_download.hf_download("owner/repo", "w.bin")


def test_hf_download_other_errors_pass_through(monkeypatch):
    def fake(*, repo_id, filename, **kwargs):
        raise ValueError("bad kwargs")

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake, raising=True)
    with pytest.raises(ValueError, match="bad kwargs"):
        hf_download.hf_download("owner/repo", "w.bin")


def test_hf_snapshot_passes_token(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "snap-token")
    seen = {}

    def fake(*, repo_id, **kwargs):
        seen.update(kwargs, repo_id=repo_id)
        return "/tmp/snap"

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake, raising=True)
    assert hf_download.hf_snapshot("owner/repo") == "/tmp/snap"
    assert seen["token"] == "snap-token"
    assert seen["repo_id"] == "owner/repo"


def test_hf_snapshot_rate_limit_wraps(monkeypatch):
    def fake(*, repo_id, **kwargs):
        raise _FakeHttpError(403, "Forbidden")

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake, raising=True)
    with pytest.raises(HfRateLimitError) as exc:
        hf_download.hf_snapshot("owner/repo")
    assert "HF_TOKEN" in str(exc.value)
    assert "owner/repo" in str(exc.value)
