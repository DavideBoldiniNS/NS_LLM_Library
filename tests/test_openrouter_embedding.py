import pytest
from ns_llm.embedding.providers.openrouter_provider import call_openrouter


class TestCallOpenRouterEmbedding:
    def test_formato_risposta(self, mocker, sample_embedding_params):
        mock_cls = mocker.patch(
            "ns_llm.embedding.providers.openrouter_provider.OpenRouter"
        )
        # OpenRouter usato come context manager
        mock_client = mock_cls.return_value.__enter__.return_value

        mock_response = mocker.MagicMock()
        mock_response.data = [mocker.MagicMock(embedding=[0.4, 0.5])]
        mock_response.usage.input_tokens = 7
        mock_client.embeddings.create.return_value = mock_response

        result = call_openrouter(**sample_embedding_params)

        assert isinstance(result, dict)
        assert "embedding" in result
        assert "input_tokens" in result
        assert isinstance(result["embedding"], list)
        assert all(isinstance(x, float) for x in result["embedding"])
        assert isinstance(result["input_tokens"], int)

    def test_senza_usage_input_tokens_zero(
        self, mocker, sample_embedding_params
    ):
        mock_cls = mocker.patch(
            "ns_llm.embedding.providers.openrouter_provider.OpenRouter"
        )
        mock_client = mock_cls.return_value.__enter__.return_value

        mock_response = mocker.MagicMock()
        mock_response.data = [mocker.MagicMock(embedding=[0.9])]
        del mock_response.usage
        mock_client.embeddings.create.return_value = mock_response

        result = call_openrouter(**sample_embedding_params)

        assert result["input_tokens"] == 0

    def test_dimensions_zero_non_passato(
        self, mocker, sample_embedding_params
    ):
        mock_cls = mocker.patch(
            "ns_llm.embedding.providers.openrouter_provider.OpenRouter"
        )
        mock_client = mock_cls.return_value.__enter__.return_value

        mock_response = mocker.MagicMock()
        mock_response.data = [mocker.MagicMock(embedding=[0.3])]
        mock_response.usage.input_tokens = 4
        mock_client.embeddings.create.return_value = mock_response

        params = sample_embedding_params.copy()
        params["dimensions"] = 0

        call_openrouter(**params)

        chiamata = mock_client.embeddings.create
        assert "dimensions" not in chiamata.call_args.kwargs

    @pytest.mark.parametrize(
        "model, should_pass_input_type",
        [
            ("openrouter/model-supporta-input-type", True),
            ("openrouter/model-senza-input-type", False),
        ],
    )
    def test_input_type_solo_se_supportato(
        self, mocker, sample_embedding_params, model, should_pass_input_type
    ):
        mock_cls = mocker.patch(
            "ns_llm.embedding.providers.openrouter_provider.OpenRouter"
        )
        mock_client = mock_cls.return_value.__enter__.return_value

        mock_response = mocker.MagicMock()
        mock_response.data = [mocker.MagicMock(embedding=[0.2])]
        mock_response.usage.input_tokens = 2
        mock_client.embeddings.create.return_value = mock_response

        params = sample_embedding_params.copy()
        params["model"] = model

        call_openrouter(**params)

        chiamata = mock_client.embeddings.create
        kwargs = chiamata.call_args.kwargs

        if should_pass_input_type:
            assert "input_type" in kwargs
        else:
            assert "input_type" not in kwargs
