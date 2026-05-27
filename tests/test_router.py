import pytest
from ns_llm.inference.client import generate_response

def test_provider_valido(mocker, sample_params, sample_response):
    mock = mocker.patch("ns_llm.inference.client.call_openai", return_value=sample_response)
    result = generate_response(provider="openai", **sample_params)
    mock.assert_called_once()
    assert result == sample_response

def test_provider_non_valido(sample_params):
    with pytest.raises((KeyError, ValueError)):
        generate_response(provider="inesistente", **sample_params)
 
@pytest.mark.parametrize("provider", ["openai", "anthropic", "together", "ollama", "openrouter"])
def test_tutti_i_provider(mocker, sample_params, sample_response, provider):
    mock = mocker.patch(f"ns_llm.inference.client.call_{provider}", return_value=sample_response)
    result = generate_response(provider=provider, **sample_params)
    mock.assert_called_once()
    assert result == sample_response