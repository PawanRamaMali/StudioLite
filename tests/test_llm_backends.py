"""LLM backend request/response shape tests.

Never hits the network - patches `requests.post` to a MagicMock and asserts
the request URL, headers, and body match what each provider expects, plus
that the response-shape parser pulls the right text back."""
from __future__ import annotations

import json
import os
from unittest.mock import patch, MagicMock

import pytest

from filmmaker import llm


def _resp(status: int, payload) -> MagicMock:
    r = MagicMock()
    r.status_code = status
    r.json.return_value = payload
    r.text = json.dumps(payload)
    return r


# ---------- Gemini ----------------------------------------------------------

def test_gemini_happy_path():
    os.environ["GEMINI_API_KEY"] = "test-gem-key"
    try:
        payload = {"candidates": [{"content": {"parts": [{"text": "hello world"}]}}]}
        with patch("filmmaker.llm.requests.post", return_value=_resp(200, payload)) as m:
            out = llm.chat("sys msg", "user msg", backend="gemini", model="gemini-2.5-flash")
        assert out == "hello world"
        # URL + body shape
        args, kwargs = m.call_args
        url = args[0]
        assert "generativelanguage.googleapis.com" in url
        assert "gemini-2.5-flash:generateContent" in url
        assert "test-gem-key" in url        # Gemini takes the key as a query param
        body = kwargs["json"]
        assert body["system_instruction"]["parts"][0]["text"] == "sys msg"
        assert body["contents"][0]["parts"][0]["text"] == "user msg"
    finally:
        del os.environ["GEMINI_API_KEY"]


def test_gemini_json_mode_sets_response_mime_type():
    os.environ["GEMINI_API_KEY"] = "test-gem-key"
    try:
        payload = {"candidates": [{"content": {"parts": [{"text": '{"ok":true}'}]}}]}
        with patch("filmmaker.llm.requests.post", return_value=_resp(200, payload)) as m:
            llm.chat("sys", "user", backend="gemini", model="m", want_json=True)
        body = m.call_args.kwargs["json"]
        assert body["generationConfig"]["response_mime_type"] == "application/json"
    finally:
        del os.environ["GEMINI_API_KEY"]


def test_gemini_http_error_hides_key():
    os.environ["GEMINI_API_KEY"] = "secret-key-should-not-leak"
    try:
        err = MagicMock()
        err.status_code = 400
        err.text = "invalid api key: secret-key-should-not-leak"
        with patch("filmmaker.llm.requests.post", return_value=err):
            with pytest.raises(llm.LLMError) as exc_info:
                llm.chat("s", "u", backend="gemini", model="m")
        msg = str(exc_info.value)
        assert "***" in msg
        assert "secret-key-should-not-leak" not in msg
    finally:
        del os.environ["GEMINI_API_KEY"]


# ---------- Groq ------------------------------------------------------------

def test_groq_openai_compat_shape():
    os.environ["GROQ_API_KEY"] = "test-groq-key"
    try:
        payload = {"choices": [{"message": {"role": "assistant", "content": "hi from groq"}}]}
        with patch("filmmaker.llm.requests.post", return_value=_resp(200, payload)) as m:
            out = llm.chat("sys", "user", backend="groq", model="llama-3.3-70b-versatile",
                           temperature=0.2, max_tokens=1000)
        assert out == "hi from groq"
        args, kwargs = m.call_args
        assert args[0] == "https://api.groq.com/openai/v1/chat/completions"
        assert kwargs["headers"]["Authorization"] == "Bearer test-groq-key"
        body = kwargs["json"]
        assert body["model"] == "llama-3.3-70b-versatile"
        assert body["messages"] == [
            {"role": "system", "content": "sys"},
            {"role": "user",   "content": "user"},
        ]
        assert body["temperature"] == 0.2
        assert body["max_tokens"] == 1000
    finally:
        del os.environ["GROQ_API_KEY"]


def test_groq_json_mode_injects_word_json_in_system():
    os.environ["GROQ_API_KEY"] = "test-groq-key"
    try:
        payload = {"choices": [{"message": {"content": "{}"}}]}
        with patch("filmmaker.llm.requests.post", return_value=_resp(200, payload)) as m:
            llm.chat("plain system", "user", backend="groq", model="m", want_json=True)
        body = m.call_args.kwargs["json"]
        assert body["response_format"] == {"type": "json_object"}
        # OpenAI JSON mode requires the string "json" somewhere in messages.
        assert "json" in body["messages"][0]["content"].lower()
    finally:
        del os.environ["GROQ_API_KEY"]


# ---------- HF Router -------------------------------------------------------

def test_hf_router_shape():
    os.environ["HF_TOKEN"] = "hf_test_token"
    try:
        payload = {"choices": [{"message": {"content": "hi from hf"}}]}
        with patch("filmmaker.llm.requests.post", return_value=_resp(200, payload)) as m:
            out = llm.chat("sys", "user", backend="hf",
                           model="meta-llama/Meta-Llama-3-70B-Instruct")
        assert out == "hi from hf"
        args, kwargs = m.call_args
        assert "router.huggingface.co" in args[0]
        assert args[0].endswith("/chat/completions")
        assert kwargs["headers"]["Authorization"] == "Bearer hf_test_token"
    finally:
        del os.environ["HF_TOKEN"]


# ---------- backend_status --------------------------------------------------

def test_backend_status_reports_missing_keys():
    for e in ("GEMINI_API_KEY", "GROQ_API_KEY", "HF_TOKEN"):
        os.environ.pop(e, None)
    s = llm.backend_status()
    assert s["gemini"]["configured"] is False
    assert s["groq"]["configured"] is False
    assert s["hf"]["configured"] is False
    assert s["ollama"]["configured"] is True
    for name, env_var in [("gemini", "GEMINI_API_KEY"),
                          ("groq",   "GROQ_API_KEY"),
                          ("hf",     "HF_TOKEN")]:
        assert s[name]["env_var"] == env_var
        assert s[name]["default_model"]


def test_backend_status_flips_when_key_set():
    os.environ["GROQ_API_KEY"] = "any-nonempty"
    try:
        s = llm.backend_status()
        assert s["groq"]["configured"] is True
    finally:
        del os.environ["GROQ_API_KEY"]


def test_missing_key_raises_specific_llmerror():
    for e in ("GEMINI_API_KEY", "GROQ_API_KEY", "HF_TOKEN"):
        os.environ.pop(e, None)
    for backend, env in [("gemini", "GEMINI_API_KEY"),
                         ("groq",   "GROQ_API_KEY"),
                         ("hf",     "HF_TOKEN")]:
        with pytest.raises(llm.LLMError) as exc_info:
            llm.chat("s", "u", backend=backend, model="m")
        assert env in str(exc_info.value)


def test_unknown_backend_raises():
    with pytest.raises(llm.LLMError):
        llm.chat("s", "u", backend="does-not-exist", model="m")


# ---------- parse_json survives real-provider messiness --------------------

def test_parse_json_survives_fenced_output():
    assert llm.parse_json("```json\n{\"a\": 1}\n```") == {"a": 1}


def test_parse_json_repairs_truncated_output():
    assert llm.parse_json('{"a": 1, "b": [1, 2') == {"a": 1, "b": [1, 2]}


def test_parse_json_rejects_pure_prose():
    """A coder or task-tuned model that ignores the JSON instruction and
    writes free-form prose should raise, not silently return {}."""
    with pytest.raises(llm.LLMError):
        llm.parse_json("The provided text is a repetitive pattern of colors.")


# ---------- agents._chat strict-JSON retry ---------------------------------

def test_agents_chat_retries_with_strict_prompt_when_prose_returned():
    """When want_json=True and the model returns prose, _chat should
    retry with a stricter system prompt rather than propagate the error."""
    from filmmaker import agents
    from filmmaker.project import Project, ProjectConfig, ProjectMeta

    class _StubProject:
        def __init__(self):
            self.meta = ProjectMeta(id="p", title="t", brief="b",
                                     config=ProjectConfig(llm_backend="ollama",
                                                          llm_model="stub"),
                                     created_at=0.0, updated_at=0.0)

    calls = []

    def fake_chat(*, system, user, backend, model, host,
                   temperature, max_tokens, want_json):
        calls.append({"system": system, "temperature": temperature,
                       "max_tokens": max_tokens})
        # First call: prose. Retry call (has strict marker): JSON.
        if "CRITICAL" in system:
            return '{"ok": true}'
        return "here is some prose, no json at all"

    with patch("filmmaker.agents.llm.chat", side_effect=fake_chat):
        out = agents._chat(_StubProject(), "producer",
                            system="you are the producer",
                            user="do the thing",
                            want_json=True, temperature=0.7)

    assert out == '{"ok": true}'
    assert len(calls) == 2
    assert "CRITICAL" in calls[1]["system"]
    assert calls[1]["temperature"] < calls[0]["temperature"]
