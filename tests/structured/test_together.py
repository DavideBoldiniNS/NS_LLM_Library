import pytest
from ns_llm.structured.providers.together_provider import call_together_structured

API_PARAMS = {
    "model": "meta-llama/Llama-3-70b",
    "max_output_tokens": 100,
    "temperature": 0.5,
    "system_prompt": "Sei un assistente.",
    "user_prompt": "test",
    "api_key": "fake-key",
    "schema": {"type": "object", "properties": {"ok": {"type": "boolean"}}},
    "reasoning": False,
}

EXPECTED_JSON = '{"ok": true}'


def test_fallback_prompt_injection(monkeypatch):
    from ns_llm import generate_response

    called_with = {}
    def fake_generate(**kwargs):
        called_with.update(kwargs)
        return {"text": EXPECTED_JSON, "input_tokens": 10, "output_tokens": 5}

    monkeypatch.setattr(
        "ns_llm.structured.providers.together_provider.generate_response",
        fake_generate,
    )

    result = call_together_structured(**API_PARAMS)
    assert result == EXPECTED_JSON
    assert "Rispondi SOLO" in called_with.get("system_prompt", "")
