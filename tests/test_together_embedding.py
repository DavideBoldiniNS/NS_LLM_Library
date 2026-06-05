import pytest
from ns_llm.embedding.providers.together_provider import call_together


class TestCallTogetherEmbedding:
    def test_formato_risposta(self, mocker, sample_embedding_params):
        mock_together_cls = mocker.patch(
            "ns_llm.embedding.providers.together_provider.Together"
        )
        mock_client = mock_together_cls.return_value

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

        mock_response = mocker.MagicMock()
        mock_response.data = [mocker.MagicMock(embedding=[0.1, 0.2])]
        del mock_response.usage
        mock_client.embeddings.create.return_value = mock_response

        result = call_together(**sample_embedding_params)

        assert result["input_tokens"] == 0

    def test_usage_prompt_tokens_fallback(
        self, mocker, sample_embedding_params
    ):
        mock_together_cls = mocker.patch(
            "ns_llm.embedding.providers.together_provider.Together"
        )
        mock_client = mock_together_cls.return_value

        mock_response = mocker.MagicMock()
        mock_response.data = [mocker.MagicMock(embedding=[0.1])]
        mock_response.usage.prompt_tokens = 42
        del mock_response.usage.input_tokens
        mock_client.embeddings.create.return_value = mock_response

        result = call_together(**sample_embedding_params)

        assert result["input_tokens"] == 42

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
        params["supports_dimensions"] = True

        call_together(**params)

        chiamata = mock_client.embeddings.create
        chiamata.assert_called_once()
        assert "dimensions" not in chiamata.call_args.kwargs

    @pytest.mark.parametrize(
        "supports_input_type, model, should_pass_input_type",
        [
            (True, "together/model-A", True),
            (False, "together/model-B", False),
        ],
    )
    def test_input_type_solo_se_supportato(
        self,
        mocker,
        sample_embedding_params,
        supports_input_type,
        model,
        should_pass_input_type,
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
        params["supports_input_type"] = supports_input_type
        params["model"] = model

        call_together(**params)

        kwargs = mock_client.embeddings.create.call_args.kwargs

        if should_pass_input_type:
            assert kwargs.get("input_type") == "search_query"
        else:
            assert "input_type" not in kwargs

    @pytest.mark.parametrize(
        "input_type, expected_text",
        [
            ("search_query", "query: test"),
            ("search_document", "passage: test"),
            ("other", "test"),
        ],
    )
    def test_applica_prefix_query_o_passage_se_input_type_rilevante(
        self,
        mocker,
        sample_embedding_params,
        input_type,
        expected_text,
    ):
        mock_together_cls = mocker.patch(
            "ns_llm.embedding.providers.together_provider.Together"
        )
        mock_client = mock_together_cls.return_value

        mock_response = mocker.MagicMock()
        mock_response.data = [mocker.MagicMock(embedding=[0.1])]
        mock_response.usage.input_tokens = 1
        mock_client.embeddings.create.return_value = mock_response

        params = sample_embedding_params.copy()
        params["input_type"] = input_type

        call_together(**params)

        kwargs = mock_client.embeddings.create.call_args.kwargs
        assert kwargs["input"] == expected_text

    def test_nessun_wrapping_di_eccezioni_sdk(
        self, mocker, sample_embedding_params
    ):
        from together.error import APIError

        mock_together_cls = mocker.patch(
            "ns_llm.embedding.providers.together_provider.Together"
        )
        mock_client = mock_together_cls.return_value
        mock_client.embeddings.create.side_effect = APIError("boom")

        with pytest.raises(APIError, match="boom"):
            call_together(**sample_embedding_params)
