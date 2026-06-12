import json
import pytest
from anthropic.resources.messages import Messages
from ns_llm.structured.providers.anthropic_provider import call_anthropic_structured

API_PARAMS = {
    "model": "claude-haiku-4-5",
    "max_output_tokens": 100,
    "temperature": 0.5,
    "system_prompt": "Sei un assistente.",
    "user_prompt": "test",
    "api_key": "fake-key",
    "schema": {"type": "object", "properties": {"ok": {"type": "boolean"}}},
    "reasoning": False,
}

EXPECTED_JSON = '{"ok": true}'


def test_native_tool_use_path(monkeypatch):
    class FakeBlock:
        input = {"ok": True}
        type = "tool_use"
        name = "output"
    class FakeResponse:
        content = [FakeBlock()]
        class FakeUsage:
            input_tokens = 10
            output_tokens = 5
        usage = FakeUsage()

    captured = {}
    def fake_create(self, **kwargs):
        captured.update(kwargs)
        return FakeResponse()

    monkeypatch.setattr(
        Messages,
        "create",
        fake_create,
    )

    result = call_anthropic_structured(**API_PARAMS)
    assert result == EXPECTED_JSON
    assert "tools" in captured
    assert captured["tool_choice"] == {"type": "tool", "name": "output"}


def test_native_fails_fallback(monkeypatch):
    called_with = {}
    def fake_generate(**kwargs):
        called_with.update(kwargs)
        return {"text": EXPECTED_JSON, "input_tokens": 10, "output_tokens": 5}

    monkeypatch.setattr(
        Messages,
        "create",
        lambda self, **kw: (_ for _ in ()).throw(Exception("API error")),
    )
    monkeypatch.setattr(
        "ns_llm.structured.providers.anthropic_provider.generate_response",
        fake_generate,
    )

    result = call_anthropic_structured(**API_PARAMS)
    assert result == EXPECTED_JSON
    assert "JSON valido" in called_with.get("system_prompt", "")
