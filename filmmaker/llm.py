"""
LLM helper for stage agents.

T1 wraps Ollama (reusing llm_filter's connection logic). Gemini / Groq /
HF Inference are named in the switch but raise NotImplementedError until
T2 wires their SDKs — this keeps the config surface stable now so the
frontend can already offer per-agent backend picks.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

from llm_filter import _client as _ollama_client, _resolve_host, is_ollama_reachable


class LLMError(RuntimeError):
    pass


def chat(
    system: str,
    user: str,
    *,
    backend: str = "ollama",
    model: str = "llama3.2",
    host: Optional[str] = None,
    temperature: float = 0.5,
    max_tokens: int = 4096,
    timeout: int = 300,
    want_json: bool = False,
) -> str:
    """Single blocking chat completion. Returns raw string content.

    `want_json` hints the local model to emit JSON only. We also try to
    strip common wrappers on the way out; the caller can still `parse_json`
    to get a dict.
    """
    backend = (backend or "ollama").lower()
    if backend == "ollama":
        return _ollama_chat(system, user, model=model, host=host,
                            temperature=temperature, max_tokens=max_tokens,
                            want_json=want_json)
    if backend in {"gemini", "groq", "hf"}:
        raise LLMError(
            f"Backend {backend!r} is not wired in T1 — pick 'ollama' or wait for T2."
        )
    raise LLMError(f"Unknown LLM backend: {backend!r}")


def parse_json(text: str) -> Dict[str, Any]:
    """Robust-ish JSON extractor for messy LLM output. Raises LLMError on failure."""
    stripped = _strip_fence(text).strip()
    # 1. Direct parse.
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    # 2. Grab the first {...} block by brace matching.
    obj = _extract_first_json_object(stripped)
    if obj is not None:
        try:
            return json.loads(obj)
        except json.JSONDecodeError as e:
            raise LLMError(f"Could not parse JSON block from LLM: {e}\nSnippet:\n{obj[:400]}")
    raise LLMError(f"No JSON object found in LLM output. Snippet:\n{stripped[:400]}")


# ---------------------------------------------------------------------------
# internals
# ---------------------------------------------------------------------------

def _ollama_chat(system: str, user: str, *, model: str, host: Optional[str],
                 temperature: float, max_tokens: int, want_json: bool) -> str:
    resolved = _resolve_host(host)
    if not is_ollama_reachable(resolved):
        raise LLMError(
            f"Ollama is not reachable at {resolved}. Start it with `ollama serve` "
            "or open the Ollama app, then retry."
        )
    client = _ollama_client(resolved)
    options: Dict[str, Any] = {"temperature": temperature, "num_predict": max_tokens}
    if want_json:
        options["format"] = "json"
        # A polite reminder at the end of `system` costs nothing and helps small models.
        system = system.rstrip() + "\n\nRespond with valid JSON only. No prose, no code fences."
    try:
        resp = client.chat(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            options=options,
            stream=False,
            keep_alive="5m",
        )
    except Exception as e:
        raise LLMError(f"Ollama request failed: {e}") from e
    return (resp.get("message", {}) or {}).get("content", "") or ""


_FENCE_RE = re.compile(r"^```(?:json|markdown|md)?\s*\n(.*)\n```\s*$", re.S | re.I)


def _strip_fence(text: str) -> str:
    m = _FENCE_RE.match(text.strip())
    return m.group(1).strip() if m else text


def _extract_first_json_object(text: str) -> Optional[str]:
    """Find the outermost {...} span using a brace-matching scan.
    Ignores braces inside string literals so we don't split on them."""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None
