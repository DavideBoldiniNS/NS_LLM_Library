import pytest
from pydantic import BaseModel
from ns_llm.structured.client import generate_structured_response


class TestPersona(BaseModel):
    name: str
    age: int


def test_dict_return_type():
    def fake_handler(**kw):
        return '{"name": "Marco", "age": 30}'
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            "ns_llm.structured.client.PROVIDER_HANDLERS",
            {"openai": fake_handler},
        )
        result = generate_structured_response(
            provider="openai",
            model="x",
            schema={"type": "object"},
            max_output_tokens=100,
            temperature=0.5,
            system_prompt="",
            user_prompt="test",
            api_key="fake-key",
        )
        assert isinstance(result, dict)
        assert result["name"] == "Marco"
        assert result["age"] == 30


def test_pydantic_return_type():
    def fake_handler(**kw):
        return '{"name": "Marco", "age": 30}'
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            "ns_llm.structured.client.PROVIDER_HANDLERS",
            {"openai": fake_handler},
        )
        result = generate_structured_response(
            provider="openai",
            model="x",
            schema=TestPersona,
            max_output_tokens=100,
            temperature=0.5,
            system_prompt="",
            user_prompt="test",
            api_key="fake-key",
        )
        assert isinstance(result, TestPersona)
        assert result.name == "Marco"
        assert result.age == 30


def test_provider_non_valido():
    with pytest.raises(ValueError, match="Provider.*non supportato"):
        generate_structured_response(
            provider="invalido",
            model="x",
            schema={"type": "object"},
            max_output_tokens=100,
            temperature=0.5,
            system_prompt="",
            user_prompt="test",
            api_key="fake-key",
        )


def test_api_key_vuota():
    with pytest.raises(ValueError, match="api_key"):
        generate_structured_response(
            provider="openai",
            model="x",
            schema={"type": "object"},
            max_output_tokens=100,
            temperature=0.5,
            system_prompt="",
            user_prompt="test",
            api_key="",
        )


def test_schema_non_valido():
    with pytest.raises(TypeError):
        generate_structured_response(
            provider="openai",
            model="x",
            schema=42,
            max_output_tokens=100,
            temperature=0.5,
            system_prompt="",
            user_prompt="test",
            api_key="fake-key",
        )


def test_max_retries_negativo():
    with pytest.raises(ValueError, match="max_retries"):
        generate_structured_response(
            provider="openai",
            model="x",
            schema={"type": "object"},
            max_output_tokens=100,
            temperature=0.5,
            system_prompt="",
            user_prompt="test",
            api_key="fake-key",
            max_retries=-1,
        )


def test_retry_success(monkeypatch):
    calls = [0]
    def fake_handler(**kw):
        calls[0] += 1
        if calls[0] == 1:
            return "{broken json"
        return '{"ok": true}'
    monkeypatch.setattr(
        "ns_llm.structured.client.PROVIDER_HANDLERS",
        {"openai": fake_handler},
    )
    result = generate_structured_response(
        provider="openai",
        model="x",
        schema={"type": "object"},
        max_output_tokens=100,
        temperature=0.5,
        system_prompt="",
        user_prompt="test",
        api_key="fake-key",
        max_retries=2,
    )
    assert result == {"ok": True}
    assert calls[0] == 2


def test_retry_exhausted(monkeypatch):
    monkeypatch.setattr(
        "ns_llm.structured.client.PROVIDER_HANDLERS",
        {"openai": lambda **kw: "{broken"},
    )
    with pytest.raises(ValueError, match="tentativi"):
        generate_structured_response(
            provider="openai",
            model="x",
            schema={"type": "object"},
            max_output_tokens=100,
            temperature=0.5,
            system_prompt="",
            user_prompt="test",
            api_key="fake-key",
            max_retries=1,
        )
