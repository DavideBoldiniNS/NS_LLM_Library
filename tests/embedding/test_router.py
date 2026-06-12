import pytest

from ns_llm.embedding.client import generate_embedding


@pytest.mark.parametrize("provider", ["together", "openrouter"])
def test_provider_valido(
    mocker, sample_embedding_params, sample_embedding_response, provider
):
    mock = mocker.patch(
        f"ns_llm.embedding.client.call_{provider}",
        return_value=sample_embedding_response,
    )
    result = generate_embedding(provider=provider, **sample_embedding_params)
    mock.assert_called_once()
    assert result == sample_embedding_response


@pytest.mark.parametrize(
    "provider_input",
    ["Together", "TOGETHER", "OpenRouter", "openrouter"],
)
def test_provider_case_insensitive(
    mocker, sample_embedding_params, sample_embedding_response, provider_input
):
    expected = "together" if "together" in provider_input.lower() else "openrouter"
    mock = mocker.patch(
        f"ns_llm.embedding.client.call_{expected}",
        return_value=sample_embedding_response,
    )
    result = generate_embedding(provider=provider_input, **sample_embedding_params)
    mock.assert_called_once()
    assert result == sample_embedding_response


def test_provider_non_valido_messaggio_elenca_validi(sample_embedding_params):
    with pytest.raises(ValueError, match="Valori validi"):
        generate_embedding(provider="inesistente", **sample_embedding_params)


def test_api_key_vuota_mossa_value_error(sample_embedding_params):
    params = sample_embedding_params.copy()
    params["api_key"] = ""
    with pytest.raises(ValueError, match="api_key"):
        generate_embedding(provider="together", **params)


def test_text_vuoto_mossa_value_error(sample_embedding_params):
    params = sample_embedding_params.copy()
    params["text"] = ""
    with pytest.raises(ValueError, match="text"):
        generate_embedding(provider="together", **params)


def test_text_solo_spazi_mossa_value_error(sample_embedding_params):
    params = sample_embedding_params.copy()
    params["text"] = "   "
    with pytest.raises(ValueError, match="text"):
        generate_embedding(provider="together", **params)


def test_model_vuoto_mossa_value_error(sample_embedding_params):
    params = sample_embedding_params.copy()
    params["model"] = ""
    with pytest.raises(ValueError, match="model"):
        generate_embedding(provider="together", **params)


def test_dimensions_negativo_mossa_value_error(sample_embedding_params):
    params = sample_embedding_params.copy()
    params["dimensions"] = -1
    with pytest.raises(ValueError, match="dimensions"):
        generate_embedding(provider="together", **params)


def test_dimensions_zero_accettato(mocker, sample_embedding_params, sample_embedding_response):
    params = sample_embedding_params.copy()
    params["dimensions"] = 0
    mock = mocker.patch(
        "ns_llm.embedding.client.call_together",
        return_value=sample_embedding_response,
    )
    generate_embedding(provider="together", **params)
    mock.assert_called_once()


def test_supports_input_type_true_passa_input_type(
    mocker, sample_embedding_params, sample_embedding_response
):
    params = sample_embedding_params.copy()
    params["supports_input_type"] = True

    mock = mocker.patch(
        "ns_llm.embedding.client.call_together",
        return_value=sample_embedding_response,
    )
    generate_embedding(provider="together", **params)

    handler_kwargs = mock.call_args.kwargs
    assert handler_kwargs["supports_input_type"] is True
    assert handler_kwargs["input_type"] == "search_query"


def test_supports_input_type_false_non_passa_input_type_al_provider(
    mocker, sample_embedding_params, sample_embedding_response
):
    mock = mocker.patch(
        "ns_llm.embedding.client.call_together",
        return_value=sample_embedding_response,
    )
    generate_embedding(provider="together", **sample_embedding_params)

    mock.assert_called_once()
    handler_kwargs = mock.call_args.kwargs
    assert handler_kwargs["supports_input_type"] is False


def test_supports_dimensions_true_passa_dimensions(
    mocker, sample_embedding_params, sample_embedding_response
):
    params = sample_embedding_params.copy()
    params["supports_dimensions"] = True
    params["dimensions"] = 256

    mock = mocker.patch(
        "ns_llm.embedding.client.call_together",
        return_value=sample_embedding_response,
    )
    generate_embedding(provider="together", **params)

    handler_kwargs = mock.call_args.kwargs
    assert handler_kwargs["supports_dimensions"] is True
    assert handler_kwargs["dimensions"] == 256
