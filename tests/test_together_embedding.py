import pytest
from ns_llm.embedding.providers.together_provider import call_together


class TestCallTogetherEmbedding:
    def test_formato_risposta(self, mocker, sample_embedding_params):
        # Mock del client Together usato nel provider
        mock_together_cls = mocker.patch(
            "ns_llm.embedding.providers.together_provider.Together"
        )
        mock_client = mock_together_cls.return_value

        # Simula risposta API: embedding + usage.input_tokens
        mock_response = mocker.MagicMock()
        mock_response.data = [mocker.MagicMock(embedding=[0.1, 0.2, 0.3])]
        mock_response.usage.input_tokens = 10
        mock_client.embeddings.create.return_value = mock_response

        result = call_together(**sample_embedding_params)

        assert isinstance(result, dict)
        assert "embedding" in result
        assert "input_tokens" in result
        assert isinstance(result["embedding"], list)
        assert all(isinstance(x, float) for x in result["embedding"])
        assert isinstance(result["input_tokens"], int)

    def test_senza_usage_input_tokens_zero(
        self, mocker, sample_embedding_params
    ):
        mock_together_cls = mocker.patch(
            "ns_llm.embedding.providers.together_provider.Together"
        )
        mock_client = mock_together_cls.return_value

        # Nessun usage nella risposta
        mock_response = mocker.MagicMock()
        mock_response.data = [mocker.MagicMock(embedding=[0.1, 0.2])]
        # Niente attributo usage
        del mock_response.usage
        mock_client.embeddings.create.return_value = mock_response

        result = call_together(**sample_embedding_params)

        assert result["input_tokens"] == 0

    def test_dimensions_zero_non_passato(
        self, mocker, sample_embedding_params
    ):
        mock_together_cls = mocker.patch(
            "ns_llm.embedding.providers.together_provider.Together"
        )
        mock_client = mock_together_cls.return_value

        mock_response = mocker.MagicMock()
        mock_response.data = [mocker.MagicMock(embedding=[0.1])]
        mock_response.usage.input_tokens = 5
        mock_client.embeddings.create.return_value = mock_response

        params = sample_embedding_params.copy()
        params["dimensions"] = 0

        call_together(**params)

        chiamata = mock_client.embeddings.create
        chiamata.assert_called_once()
        # dimensions NON deve essere nei kwargs
        assert "dimensions" not in chiamata.call_args.kwargs

    @pytest.mark.parametrize(
        "model, should_pass_input_type",
        [
            ("together/model-supporta-input-type", True),
            ("together/model-senza-input-type", False),
        ],
    )
    def test_input_type_solo_se_supportato(
        self, mocker, sample_embedding_params, model, should_pass_input_type
    ):
        mock_together_cls = mocker.patch(
            "ns_llm.embedding.providers.together_provider.Together"
        )
        mock_client = mock_together_cls.return_value

        mock_response = mocker.MagicMock()
        mock_response.data = [mocker.MagicMock(embedding=[0.1])]
        mock_response.usage.input_tokens = 3
        mock_client.embeddings.create.return_value = mock_response

        params = sample_embedding_params.copy()
        params["model"] = model

        call_together(**params)

        chiamata = mock_client.embeddings.create
        kwargs = chiamata.call_args.kwargs

        if should_pass_input_type:
            assert "input_type" in kwargs
        else:
            assert "input_type" not in kwargs
