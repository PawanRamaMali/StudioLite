"""Thin wrapper around huggingface_hub downloads.

Reads HF_TOKEN once from the environment (or the cached CLI token on disk),
passes it to hf_hub_download / snapshot_download, and converts HF Hub rate
limit / auth errors into HfRateLimitError with a clear remediation message.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

_TOKEN_ENV_KEYS = (
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "HUGGINGFACE_HUB_TOKEN",
)

_RATE_LIMIT_HINT = (
    "HuggingFace Hub is rate-limiting or refusing this download. "
    "Set HF_TOKEN in the environment to a free HuggingFace account token "
    "or run `huggingface-cli login` once, then retry."
)


class HfRateLimitError(RuntimeError):
    """Raised when an HF Hub download fails because of a 429 / 401 / 403."""


def hf_token() -> Optional[str]:
    """Return the HF token from the environment or the on-disk CLI cache.

    Priority: HF_TOKEN, HUGGING_FACE_HUB_TOKEN, HUGGINGFACE_HUB_TOKEN,
    then ~/.cache/huggingface/token (written by `huggingface-cli login`).
    Empty / whitespace-only values are treated as absent.
    """
    for key in _TOKEN_ENV_KEYS:
        raw = os.environ.get(key)
        if raw:
            tok = raw.strip()
            if tok:
                return tok
    cached = Path.home() / ".cache" / "huggingface" / "token"
    try:
        if cached.is_file():
            txt = cached.read_text(encoding="utf-8").strip()
            return txt or None
    except OSError:
        return None
    return None


def _is_rate_limit(err: BaseException) -> bool:
    """True when the failure looks like a HF Hub 429 / 401 / 403."""
    resp = getattr(err, "response", None)
    code = getattr(resp, "status_code", None)
    if code in (401, 403, 429):
        return True
    msg = str(err).lower()
    needles = (
        "429",
        "401",
        "403",
        "too many requests",
        "rate limit",
        "rate-limit",
        "unauthorized",
        "forbidden",
    )
    return any(n in msg for n in needles)


def hf_download(repo_id: str, filename: str, **kwargs: Any) -> str:
    """`huggingface_hub.hf_hub_download` with token propagation.

    Any `token` already in kwargs wins; otherwise the resolved
    `hf_token()` is passed. 429 / 401 / 403 failures are re-raised as
    `HfRateLimitError` with a clear hint instead of the bare message.
    """
    from huggingface_hub import hf_hub_download
    if "token" not in kwargs:
        kwargs["token"] = hf_token()
    try:
        return hf_hub_download(repo_id=repo_id, filename=filename, **kwargs)
    except Exception as e:
        if _is_rate_limit(e):
            raise HfRateLimitError(
                f"{repo_id}/{filename}: {_RATE_LIMIT_HINT}"
            ) from e
        raise


def hf_snapshot(repo_id: str, **kwargs: Any) -> str:
    """`huggingface_hub.snapshot_download` with the same wrapping."""
    from huggingface_hub import snapshot_download
    if "token" not in kwargs:
        kwargs["token"] = hf_token()
    try:
        return snapshot_download(repo_id=repo_id, **kwargs)
    except Exception as e:
        if _is_rate_limit(e):
            raise HfRateLimitError(
                f"{repo_id}: {_RATE_LIMIT_HINT}"
            ) from e
        raise
