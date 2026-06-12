import pytest
from ns_llm.inference.client import generate_response


class TestRouter:
    def test_provider_valido(self, mocker, sample_params, sample_response):
        mock = mocker.patch("ns_llm.inference.client.call_openai", return_value=sample_response)
        result = generate_response(provider="openai", **sample_params)
        mock.assert_called_once()
        assert result == sample_response

    def test_provider_non_valido(self, sample_params):
        with pytest.raises(ValueError, match="Valori validi"):
            generate_response(provider="inesistente", **sample_params)

    @pytest.mark.parametrize("provider", ["openai", "anthropic", "together", "ollama", "openrouter"])
    def test_tutti_i_provider(self, mocker, sample_params, sample_response, provider):
        mock = mocker.patch(f"ns_llm.inference.client.call_{provider}", return_value=sample_response)
        result = generate_response(provider=provider, **sample_params)
        mock.assert_called_once()
        assert result == sample_response


class TestStreaming:
    def test_stream_false(self, mocker, sample_params, sample_response):
        mock = mocker.patch("ns_llm.inference.client.call_openai", return_value=sample_response)
        result = generate_response(provider="openai", **sample_params, stream=False)
        assert isinstance(result, dict)
        assert result["text"] == "Risposta di test."

    def test_stream_true_restituisce_generatore(self, mocker, sample_params):
        mock_gen = ({"text": "ciao", "finish_reason": "stop"} for _ in range(1))
        mocker.patch("ns_llm.inference.client.call_openai", return_value=mock_gen)
        result = generate_response(provider="openai", **sample_params, stream=True)
        assert hasattr(result, "__iter__")
        chunks = list(result)
        assert len(chunks) == 1
        assert chunks[0]["text"] == "ciao"

    def test_stream_concatena_chunk(self, mocker, sample_params):
        def mock_stream():
            for t in ("Hello ", "world", "!"):
                yield {"text": t, "finish_reason": None}
            yield {"text": "", "finish_reason": "stop"}

        mocker.patch("ns_llm.inference.client.call_openai", return_value=mock_stream())
        result = generate_response(provider="openai", **sample_params, stream=True)
        full = "".join(chunk["text"] for chunk in result)
        assert full == "Hello world!"

    @pytest.mark.parametrize("provider", ["openai", "anthropic", "together", "ollama", "openrouter"])
    def test_stream_tutti_i_provider(self, mocker, sample_params, provider):
        mock_gen = ({"text": "test", "finish_reason": "stop"} for _ in range(1))
        mocker.patch(f"ns_llm.inference.client.call_{provider}", return_value=mock_gen)
        result = generate_response(provider=provider, **sample_params, stream=True)
        chunks = list(result)
        assert len(chunks) == 1
        assert chunks[0]["finish_reason"] == "stop"

    def test_stream_invalido(self, sample_params):
        with pytest.raises(ValueError, match="stream"):
            generate_response(provider="openai", **sample_params, stream="yes")

    def test_finish_reason_trailing_chunk(self, mocker, sample_params):
        def mock_stream():
            yield {"text": "test", "finish_reason": "stop"}

        mocker.patch("ns_llm.inference.client.call_openai", return_value=mock_stream())
        result = generate_response(provider="openai", **sample_params, stream=True)
        last = list(result)[-1]
        assert last["finish_reason"] == "stop"

    def test_finish_reason_length(self, mocker, sample_params):
        def mock_stream():
            for _ in range(3):
                yield {"text": "a", "finish_reason": None}
            yield {"text": "", "finish_reason": "length"}

        mocker.patch("ns_llm.inference.client.call_openai", return_value=mock_stream())
        result = generate_response(provider="openai", **sample_params, stream=True)
        chunks = list(result)
        assert chunks[-1]["finish_reason"] == "length"
