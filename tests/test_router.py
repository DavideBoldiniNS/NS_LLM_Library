import pytest
from ns_llm.inference.client import generate_response
from ns_llm.embedding.client import generate_embedding

#   INFERENCE
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

#   EMBEDDING
def test_provider_embedding_valido(mocker, sample_embedding_params, sample_embedding_response):
    mock = mocker.patch("ns_llm.embedding.client.call_openrouter", return_value=sample_embedding_response)
    result = generate_embedding(provider="openrouter", **sample_embedding_params)
    mock.assert_called_once()
    assert result == sample_embedding_response

def test_provider_embedding_non_valido(sample_embedding_params):
    with pytest.raises((KeyError, ValueError)):
        generate_embedding(provider="inesistente", **sample_embedding_params)

@pytest.mark.parametrize("provider", ["together", "openrouter"])
def test_tutti_i_provider_embedding(mocker, sample_embedding_params, sample_embedding_response, provider):
    mock = mocker.patch(f"ns_llm.embedding.client.call_{provider}", return_value=sample_embedding_response)
    result = generate_embedding(provider=provider, **sample_embedding_params)
    mock.assert_called_once()
    assert result == sample_embedding_response