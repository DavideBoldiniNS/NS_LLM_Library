import pytest
from ns_llm.tokenizer.client import count_tokens


def test_provider_valido(mocker, sample_tokenizer_params):
    mock = mocker.patch(
        "ns_llm.tokenizer.client.count_openai_tokens", return_value=10
    )
    result = count_tokens(provider="openai", **sample_tokenizer_params)
    mock.assert_called_once()
    assert result == 10


def test_provider_non_valido(sample_tokenizer_params):
    with pytest.raises(ValueError, match="Valori validi"):
        count_tokens(provider="inesistente", **sample_tokenizer_params)


@pytest.mark.parametrize("provider", ["openai", "anthropic", "cohere", "ollama", "together", "openrouter"])
def test_tutti_i_provider(mocker, sample_tokenizer_params, provider):
    handler_name = (
        "estimate_tokens"
        if provider in ("together", "openrouter")
        else f"count_{provider}_tokens"
    )
    mock = mocker.patch(
        f"ns_llm.tokenizer.client.{handler_name}", return_value=10
    )
    result = count_tokens(provider=provider, api_key="fake-key", **sample_tokenizer_params)
    mock.assert_called_once()
    assert result == 10


def test_text_vuoto():
    result = count_tokens(provider="together", model="m", text="")
    assert result == 0


def test_provider_vuoto():
    with pytest.raises(ValueError, match="provider"):
        count_tokens(provider="", model="m", text="test")


def test_model_vuoto():
    with pytest.raises(ValueError, match="model"):
        count_tokens(provider="openai", model="", text="test")


def test_text_non_stringa():
    with pytest.raises(ValueError, match="text"):
        count_tokens(provider="openai", model="m", text=123)


def test_estimation():
    from ns_llm.tokenizer.providers._estimation import estimate_tokens

    assert estimate_tokens("") == 0
    assert estimate_tokens("ciao") == 1
    assert estimate_tokens("ciao come stai") == 3
    assert estimate_tokens("x" * 100) == 25
