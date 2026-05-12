#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM post-processor for transcripts.

Runs raw text (Screen Transcribe output, but works for anything text-shaped)
through a local Ollama model with a style-specific system prompt and returns
clean Markdown.

Defaults to Ollama on localhost:11434; configurable via OLLAMA_HOST env var.
"""

from __future__ import annotations

import logging
import os
import re
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_MODEL = os.environ.get("OLLAMA_DEFAULT_MODEL", "llama3.2")


# ---------------------------------------------------------------------------
# Style presets
# ---------------------------------------------------------------------------

_BASE_FIXUP = """The input is raw OCR text captured from a screen recording. It has these typical defects:
- Missing or merged spaces ("IdentityManagement" should be "Identity Management").
- Character drift ("Editot" -> "Editor", "0" -> "O", "l" -> "1").
- Duplicate fragments from scrolling.
- Residual UI chrome (browser/Office menu labels, page numbers, footers) that survived earlier filters.

When fixing, use context. Do NOT invent content - work only from what's in the text.
"""

STYLE_PROMPTS: Dict[str, Dict[str, str]] = {
    "cleanup": {
        "label": "Cleanup",
        "description": "Fix OCR errors and restore structure; preserve original ordering.",
        "system": _BASE_FIXUP + """
Your task:
1. Repair OCR errors and missing word boundaries.
2. Drop residual UI chrome and any clearly-duplicated fragments.
3. Preserve the source document's structure: headings, lists, paragraphs - in the original order.
4. Output clean Markdown only. No preamble, no commentary, no code fence.
""",
    },
    "meeting": {
        "label": "Meeting Notes",
        "description": "Reorganize as meeting minutes with sections.",
        "system": _BASE_FIXUP + """
Your task: turn the input into structured meeting notes in Markdown. Use these sections, in this order, but ONLY include sections that have content in the source:

# Meeting Notes

## Attendees
- (names mentioned as participants)

## Topics Discussed
- (each topic as a bullet, with brief sub-bullets for key points)

## Decisions
- (decisions made during the meeting)

## Action Items
- (action items / follow-ups, including owner if named)

Drop everything that doesn't fit one of those buckets. Output Markdown only.
""",
    },
    "outline": {
        "label": "Outline",
        "description": "Hierarchical outline with headings and bullets.",
        "system": _BASE_FIXUP + """
Your task: restructure the input as a hierarchical Markdown outline.
- `#` for top-level sections
- `##` for subsections
- `###` for sub-subsections
- `-` for bullets within a section

Drop UI chrome and duplicates. Keep it tight. Output Markdown only.
""",
    },
    "summary": {
        "label": "Summary",
        "description": "Short summary (~250 words) preserving key facts and decisions.",
        "system": _BASE_FIXUP + """
Your task: produce a concise summary in Markdown, at most 250 words. Preserve:
- The main topics
- Decisions made
- Action items / commitments
- Notable facts and numbers

Use brief Markdown headings. Output Markdown only.
""",
    },
    "custom": {
        "label": "Custom",
        "description": "Use your own prompt (the user supplies a system prompt at run time).",
        "system": "",
    },
}


def list_styles() -> List[Dict[str, str]]:
    return [{"id": k, "label": v["label"], "description": v["description"]} for k, v in STYLE_PROMPTS.items()]


# ---------------------------------------------------------------------------
# Ollama plumbing
# ---------------------------------------------------------------------------

def _resolve_host(host: Optional[str]) -> str:
    return (host or OLLAMA_HOST).strip().rstrip("/")


def _client(host: Optional[str] = None):
    import ollama
    return ollama.Client(host=_resolve_host(host))


def is_ollama_reachable(host: Optional[str] = None, timeout: float = 1.5) -> bool:
    """Quick TCP probe so the UI can render a helpful state when Ollama is offline."""
    import socket
    from urllib.parse import urlparse
    u = urlparse(_resolve_host(host))
    hostname = u.hostname or "127.0.0.1"
    port = u.port or 11434
    try:
        with socket.create_connection((hostname, port), timeout=timeout):
            return True
    except OSError:
        return False


def list_models(host: Optional[str] = None) -> List[Dict[str, Any]]:
    """Return models available on the given Ollama host. Empty list if unreachable."""
    if not is_ollama_reachable(host):
        return []
    try:
        resp = _client(host).list()
    except Exception as e:
        logger.warning(f"ollama list failed: {e}")
        return []
    out = []
    for m in resp.get("models", []) or []:
        name = m.get("model") or m.get("name")
        if not name:
            continue
        details = m.get("details") or {}
        out.append({
            "name": name,
            "size_gb": round((m.get("size") or 0) / 1e9, 2),
            "family": details.get("family"),
            "parameter_size": details.get("parameter_size"),
            "quantization": details.get("quantization_level"),
        })
    return out


# ---------------------------------------------------------------------------
# Input pre-processing
# ---------------------------------------------------------------------------

_TIMESTAMP_RE = re.compile(r"^\[\d{2}:\d{2}:\d{2}\]\s*")


def strip_timestamps(text: str) -> str:
    """Remove leading [HH:MM:SS] tags that Screen Transcribe writes per line."""
    return "\n".join(_TIMESTAMP_RE.sub("", line) for line in text.splitlines())


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def restructure(
    text: str,
    model: str = DEFAULT_MODEL,
    style: str = "cleanup",
    custom_prompt: Optional[str] = None,
    temperature: float = 0.2,
    timeout: int = 300,
    host: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the input through an Ollama model with the chosen style prompt.

    Returns:
        {
            "markdown": str,            # the cleaned, structured output
            "model":   str,
            "style":   str,
            "elapsed": float,           # seconds
            "input_chars":  int,
            "output_chars": int,
        }
    """
    resolved_host = _resolve_host(host)
    if not is_ollama_reachable(resolved_host):
        raise RuntimeError(
            f"Ollama is not reachable at {resolved_host}. "
            "Start it with `ollama serve` or open the Ollama app."
        )
    if style == "custom":
        if not custom_prompt or not custom_prompt.strip():
            raise ValueError("Custom style requires a non-empty custom_prompt.")
        system_prompt = custom_prompt.strip()
    else:
        preset = STYLE_PROMPTS.get(style)
        if not preset:
            raise ValueError(f"Unknown style: {style!r}. Available: {list(STYLE_PROMPTS)}")
        system_prompt = preset["system"]

    cleaned_input = strip_timestamps(text).strip()
    if not cleaned_input:
        raise ValueError("Input text is empty after stripping timestamps.")

    client = _client(resolved_host)
    started = time.time()
    try:
        resp = client.chat(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": cleaned_input},
            ],
            options={
                "temperature": temperature,
                "num_predict": 4096,
            },
            stream=False,
            keep_alive="5m",
        )
    except Exception as e:
        # ollama errors here usually mean: model not pulled, or model failed.
        raise RuntimeError(f"Ollama request failed: {e}") from e

    md = (resp.get("message", {}) or {}).get("content", "") or ""
    md = _strip_code_fence(md.strip())

    return {
        "markdown": md,
        "model": model,
        "style": style,
        "elapsed": round(time.time() - started, 2),
        "input_chars": len(cleaned_input),
        "output_chars": len(md),
    }


_FENCE_RE = re.compile(r"^```(?:markdown|md)?\s*\n(.*)\n```\s*$", re.S | re.I)


def _strip_code_fence(text: str) -> str:
    """LLMs sometimes wrap their entire reply in a ```markdown fence. Unwrap."""
    m = _FENCE_RE.match(text)
    if m:
        return m.group(1).strip()
    return text
