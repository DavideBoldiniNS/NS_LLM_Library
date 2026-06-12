import json

import pytest
from openai.resources.chat.completions import Completions as ChatCompletions
from ns_llm.structured.providers.openai_provider import call_openai_structured

API_PARAMS = {
    "model": "gpt-4o",
    "max_output_tokens": 100,
    "temperature": 0.5,
    "system_prompt": "Sei un assistente.",
    "user_prompt": "test",
    "api_key": "fake-key",
    "schema": {"type": "object", "properties": {"ok": {"type": "boolean"}}},
    "reasoning": False,
}

EXPECTED_JSON = '{"ok": true}'


def test_native_path(monkeypatch):
    class FakeMessage:
        content = EXPECTED_JSON
    class FakeChoice:
        message = FakeMessage()
    class FakeResponse:
        choices = [FakeChoice()]

    captured = {}
    def fake_create(*args, **kwargs):
        captured.update(kwargs)
        return FakeResponse()

    monkeypatch.setattr(
        ChatCompletions,
        "create",
        fake_create,
    )

    result = call_openai_structured(**API_PARAMS)
    assert result == EXPECTED_JSON
    assert captured.get("response_format") == {
        "type": "json_schema",
        "json_schema": {"name": "response", "strict": True,
                         "schema": API_PARAMS["schema"]},
    }


def test_native_fails_fallback_to_prompt(monkeypatch):
    called_with = {}
    def fake_generate(**kwargs):
        called_with.update(kwargs)
        return {"text": EXPECTED_JSON, "input_tokens": 10, "output_tokens": 5}

    monkeypatch.setattr(
        ChatCompletions,
        "create",
        lambda **kw: (_ for _ in ()).throw(Exception("API error")),
    )
    monkeypatch.setattr(
        "ns_llm.structured.providers.openai_provider.generate_response",
        fake_generate,
    )

    result = call_openai_structured(**API_PARAMS)
    assert result == EXPECTED_JSON
    assert "Rispondi SOLO con un JSON valido" in called_with.get("system_prompt", "")
