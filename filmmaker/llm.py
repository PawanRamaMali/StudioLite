"""
LLM helper for stage agents.

Four backends are wired: local Ollama, Google Gemini (REST), Groq (OpenAI-
compatible REST), and Hugging Face Inference (OpenAI-compatible router).
Every non-Ollama backend requires an env-var API key:

    GEMINI_API_KEY   for Gemini
    GROQ_API_KEY     for Groq
    HF_TOKEN         for Hugging Face

Missing keys are surfaced as `LLMError` with a message telling the user
which env var to set. Keys are never logged.

Every backend understands `want_json=True` and enables the provider's
native JSON-mode where available:

    Ollama : options.format = "json"
    Gemini : generationConfig.response_mime_type = "application/json"
    Groq   : response_format = {"type": "json_object"}
    HF     : response_format = {"type": "json_object"} on providers that
             expose it; on those that don't we still add a system-prompt
             reminder and rely on `parse_json` to survive.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

import requests

from llm_filter import _client as _ollama_client, _resolve_host, is_ollama_reachable

logger = logging.getLogger("studiolite.filmmaker.llm")


class LLMError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

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

    `want_json` hints the model to emit JSON only. Providers with a native
    JSON mode use it; the caller should still `parse_json()` the result
    to survive the occasional truncation."""
    backend = (backend or "ollama").lower().strip()
    if backend == "ollama":
        return _ollama_chat(system, user, model=model, host=host,
                            temperature=temperature, max_tokens=max_tokens,
                            want_json=want_json)
    if backend == "gemini":
        return _gemini_chat(system, user, model=model,
                            temperature=temperature, max_tokens=max_tokens,
                            timeout=timeout, want_json=want_json)
    if backend == "groq":
        return _groq_chat(system, user, model=model,
                          temperature=temperature, max_tokens=max_tokens,
                          timeout=timeout, want_json=want_json)
    if backend == "hf":
        return _hf_chat(system, user, model=model,
                        temperature=temperature, max_tokens=max_tokens,
                        timeout=timeout, want_json=want_json)
    raise LLMError(f"Unknown LLM backend: {backend!r}")


def backend_status() -> Dict[str, Dict[str, Any]]:
    """Report which backends are ready to use. Read-only - never contacts
    an external service, just inspects env vars + local Ollama reachability.
    Used by the /system/llm-backends probe so the UI can gate its picker."""
    status: Dict[str, Dict[str, Any]] = {}
    resolved = _resolve_host(None)
    status["ollama"] = {
        "configured": True,   # always available in principle
        "reachable": bool(is_ollama_reachable(resolved)),
        "host": resolved,
        "default_model": os.environ.get("OLLAMA_DEFAULT_MODEL", "llama3.2"),
        "env_var": None,
    }
    for name, env_var, default_model in [
        ("gemini", "GEMINI_API_KEY", os.environ.get("GEMINI_DEFAULT_MODEL", "gemini-2.5-flash")),
        ("groq",   "GROQ_API_KEY",   os.environ.get("GROQ_DEFAULT_MODEL",   "llama-3.3-70b-versatile")),
        ("hf",     "HF_TOKEN",       os.environ.get("HF_DEFAULT_MODEL",     "meta-llama/Meta-Llama-3-70B-Instruct")),
    ]:
        has = bool(os.environ.get(env_var))
        status[name] = {
            "configured": has,
            "reachable": has,      # cloud - trust the key exists; live probe would spend a request
            "host": None,
            "default_model": default_model,
            "env_var": env_var,
        }
    return status


# parse_json is defined below with the rest of the JSON helpers.


# ---------------------------------------------------------------------------
# Ollama
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


# ---------------------------------------------------------------------------
# Gemini (Google Generative Language API)
# ---------------------------------------------------------------------------

_GEMINI_BASE = os.environ.get(
    "GEMINI_API_BASE",
    "https://generativelanguage.googleapis.com/v1beta",
)


def _gemini_chat(system: str, user: str, *, model: str,
                 temperature: float, max_tokens: int, timeout: int,
                 want_json: bool) -> str:
    key = _require_env("GEMINI_API_KEY")
    model = (model or "gemini-2.5-flash").strip()
    # Gemini supports both system_instruction and inline `system` role; the
    # cleanest is system_instruction, which frees us from encoding it as a
    # user turn and getting the polarity wrong.
    body: Dict[str, Any] = {
        "system_instruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig": {
            "temperature": float(temperature),
            "maxOutputTokens": int(max_tokens),
        },
    }
    if want_json:
        body["generationConfig"]["response_mime_type"] = "application/json"

    url = f"{_GEMINI_BASE}/models/{model}:generateContent?key={key}"
    try:
        r = requests.post(url, json=body, timeout=timeout,
                          headers={"Content-Type": "application/json"})
    except requests.RequestException as e:
        raise LLMError(f"Gemini network error: {e}") from e
    if r.status_code != 200:
        raise LLMError(_short_http_error("Gemini", r))
    data = _safe_json(r)
    # Standard path: candidates[0].content.parts[*].text (concatenate).
    try:
        cands = data.get("candidates") or []
        if not cands:
            # Sometimes the request is blocked with a promptFeedback field.
            fb = data.get("promptFeedback")
            raise LLMError(f"Gemini returned no candidates ({fb})")
        parts = (cands[0].get("content") or {}).get("parts") or []
        return "".join(p.get("text", "") for p in parts)
    except LLMError:
        raise
    except Exception as e:
        raise LLMError(f"Gemini response parse failed: {e}; snippet={str(data)[:400]}") from e


# ---------------------------------------------------------------------------
# Groq (OpenAI-compatible REST)
# ---------------------------------------------------------------------------

_GROQ_BASE = os.environ.get("GROQ_API_BASE", "https://api.groq.com/openai/v1")


def _groq_chat(system: str, user: str, *, model: str,
               temperature: float, max_tokens: int, timeout: int,
               want_json: bool) -> str:
    key = _require_env("GROQ_API_KEY")
    model = (model or "llama-3.3-70b-versatile").strip()
    body: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }
    if want_json:
        body["response_format"] = {"type": "json_object"}
        # OpenAI-compat JSON mode still requires the system prompt to
        # mention the word "json" somewhere.
        if "json" not in (system or "").lower():
            body["messages"][0]["content"] = (system.rstrip()
                + "\n\nRespond with valid JSON only.")
    return _openai_compat_call("Groq", _GROQ_BASE, key, body, timeout)


# ---------------------------------------------------------------------------
# Hugging Face Inference (OpenAI-compatible router)
# ---------------------------------------------------------------------------

# The router.huggingface.co surface exposes an OpenAI-compat /chat/completions
# for every model that the paired provider supports. `HF_INFERENCE_BASE` can
# point at a self-hosted TGI endpoint or a different provider prefix
# (e.g. router.huggingface.co/together, router.huggingface.co/fireworks).
_HF_BASE = os.environ.get("HF_INFERENCE_BASE",
                          "https://router.huggingface.co/hf-inference/v1")


def _hf_chat(system: str, user: str, *, model: str,
             temperature: float, max_tokens: int, timeout: int,
             want_json: bool) -> str:
    key = _require_env("HF_TOKEN")
    model = (model or "meta-llama/Meta-Llama-3-70B-Instruct").strip()
    body: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }
    if want_json:
        # HF's router forwards this to providers that honor OpenAI JSON mode;
        # for providers that ignore it, the system-prompt reminder still helps.
        body["response_format"] = {"type": "json_object"}
        if "json" not in (system or "").lower():
            body["messages"][0]["content"] = (system.rstrip()
                + "\n\nRespond with valid JSON only.")
    return _openai_compat_call("HuggingFace", _HF_BASE, key, body, timeout)


# ---------------------------------------------------------------------------
# Shared HTTP helpers
# ---------------------------------------------------------------------------

def _openai_compat_call(label: str, base_url: str, api_key: str,
                        body: Dict[str, Any], timeout: int) -> str:
    """POST to `{base_url}/chat/completions` and extract choices[0].message.content."""
    url = base_url.rstrip("/") + "/chat/completions"
    try:
        r = requests.post(url, json=body, timeout=timeout, headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        })
    except requests.RequestException as e:
        raise LLMError(f"{label} network error: {e}") from e
    if r.status_code != 200:
        raise LLMError(_short_http_error(label, r))
    data = _safe_json(r)
    try:
        choices = data.get("choices") or []
        if not choices:
            raise LLMError(f"{label} returned no choices: {str(data)[:300]}")
        msg = choices[0].get("message") or {}
        return msg.get("content", "") or ""
    except LLMError:
        raise
    except Exception as e:
        raise LLMError(f"{label} response parse failed: {e}; snippet={str(data)[:400]}") from e


def _short_http_error(label: str, r: "requests.Response") -> str:
    """Turn a non-200 into a compact message. Never leaks the API key even
    if the provider echoes it (which some do in error bodies)."""
    body_txt = ""
    try:
        body_txt = r.text
    except Exception:
        pass
    # Redact anything that looks like the key just in case.
    for env_var in ("GEMINI_API_KEY", "GROQ_API_KEY", "HF_TOKEN"):
        key = os.environ.get(env_var)
        if key and key in body_txt:
            body_txt = body_txt.replace(key, "***")
    if len(body_txt) > 400:
        body_txt = body_txt[:400] + "…"
    return f"{label} HTTP {r.status_code}: {body_txt or '(empty body)'}"


def _safe_json(r: "requests.Response") -> Dict[str, Any]:
    try:
        return r.json()
    except Exception as e:
        raise LLMError(f"non-JSON response: {e}; body={r.text[:300]!r}") from e


def _require_env(name: str) -> str:
    v = os.environ.get(name, "").strip()
    if not v:
        raise LLMError(
            f"{name} is not set. Add it to your environment (or .env next to "
            "api_server.py) before using this backend."
        )
    return v


# ---------------------------------------------------------------------------
# JSON extraction internals (shared by parse_json above)
# ---------------------------------------------------------------------------

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


def _repair_truncated_json(text: str) -> Optional[str]:
    """Best-effort completion of a JSON snippet that was cut off mid-way.

    Handles the shapes providers actually produce when truncated:
- Unclosed string (append `"`)
- Trailing `,` or `:` with no value after (strip; insert `null` when
      mid-key-value)
- Unclosed `{` / `[` (append matching closers in reverse)

    Returns a JSON string that MIGHT parse, or None if the input has no
    JSON-shaped content at all. The caller should still `json.loads` the
    result to confirm."""
    if not text:
        return None
    candidates = [i for i in (text.find("{"), text.find("[")) if i >= 0]
    if not candidates:
        return None
    text = text[min(candidates):]

    stack: list = []
    in_str = False
    escape = False
    for c in text:
        if in_str:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c in "{[":
                stack.append(c)
            elif c in "}]":
                if stack:
                    stack.pop()

    out = text
    if in_str:
        if escape:
            out = out.rstrip("\\")
        out += '"'
    trimmed = out.rstrip()
    while trimmed and trimmed[-1] in ",:":
        if trimmed[-1] == ":":
            trimmed = trimmed + " null"
            break
        trimmed = trimmed[:-1].rstrip()
    closers = []
    while stack:
        open_c = stack.pop()
        closers.append("}" if open_c == "{" else "]")
    return trimmed + "".join(closers)


def parse_json(text: str) -> Dict[str, Any]:
    """Robust-ish JSON extractor for messy LLM output. Raises LLMError on failure.

    Attempts, in order: direct parse, first {..} block by brace matching,
    a truncation repair (close hanging strings + brackets). Ollama occasionally
    stops mid-object when the model runs long; the repair lets us salvage the
    complete portion instead of nuking the whole stage."""
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
        except json.JSONDecodeError:
            pass
    # 3. Try to repair a truncated response.
    repaired = _repair_truncated_json(stripped)
    if repaired:
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            pass
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


def _repair_truncated_json(text: str) -> Optional[str]:
    """Best-effort completion of a JSON snippet that was cut off mid-way.

    Handles the shapes Ollama actually produces when its response is
    truncated by num_predict:
- Unclosed string (append `"`)
- Trailing `,` or `:` with no value after (strip the dangling separator;
      insert `null` when the caller was mid-key-value)
- Unclosed `{` / `[` (append matching closers in reverse)

    Returns a JSON string that MIGHT parse, or None if the input has no
    JSON-shaped content at all. The caller should still `json.loads` the
    result to confirm."""
    if not text:
        return None
    # Trim to the first `{` or `[`.
    candidates = [i for i in (text.find("{"), text.find("[")) if i >= 0]
    if not candidates:
        return None
    text = text[min(candidates):]

    stack: list = []      # each entry is '{' or '['
    in_str = False
    escape = False
    for c in text:
        if in_str:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c in "{[":
                stack.append(c)
            elif c in "}]":
                if stack:
                    stack.pop()

    out = text
    # If we ended mid-string, close it. A dangling `\` at the end is an
    # unfinished escape sequence - drop it before closing, otherwise the
    # appended `"` becomes an escaped quote and the string never closes.
    if in_str:
        if escape:
            out = out.rstrip("\\")
        out += '"'
    # Drop dangling separators that would create empty values.
    trimmed = out.rstrip()
    while trimmed and trimmed[-1] in ",:":
        # `:` means we truncated after a key - insert `null` before closing
        # so the JSON becomes `{ "k": null }` instead of `{ "k": }`.
        if trimmed[-1] == ":":
            trimmed = trimmed + " null"
            break
        trimmed = trimmed[:-1].rstrip()
    # Close open brackets in reverse.
    closers = []
    while stack:
        open_c = stack.pop()
        closers.append("}" if open_c == "{" else "]")
    return trimmed + "".join(closers)
