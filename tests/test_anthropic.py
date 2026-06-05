import pytest

from ns_llm.inference.providers.anthropic_provider import call_anthropic


class TestCallAnthropic:
    def test_senza_reasoning(self, mocker, sample_params):
        mock_anthropic_cls = mocker.patch("ns_llm.inference.providers.anthropic_provider.Anthropic")
        mock_client = mock_anthropic_cls.return_value

        mock_block = mocker.MagicMock()
        mock_block.type = "text"
        mock_block.text = "Risposta Anthropic"

        mock_response = mocker.MagicMock()
        mock_response.content = [mock_block]
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 5

        mock_client.messages.create.return_value = mock_response

        result = call_anthropic(**sample_params)

        assert result == {
            "text": "Risposta Anthropic",
            "input_tokens": 10,
            "output_tokens": 5,
        }

        mock_anthropic_cls.assert_called_once_with(api_key="fake-key-12345")

        kwargs = mock_client.messages.create.call_args.kwargs
        assert kwargs["temperature"] == pytest.approx(0.7)
        assert kwargs["thinking"] == {"type": "disabled"}

    def test_con_reasoning_su_modello_non_opus(self, mocker, sample_params):
        mock_anthropic_cls = mocker.patch("ns_llm.inference.providers.anthropic_provider.Anthropic")
        mock_client = mock_anthropic_cls.return_value

        mock_block = mocker.MagicMock()
        mock_block.type = "text"
        mock_block.text = "Risposta Anthropic"
        mock_response = mocker.MagicMock()
        mock_response.content = [mock_block]
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 5
        mock_client.messages.create.return_value = mock_response

        params = sample_params.copy()
        params["reasoning"] = True
        params["model"] = "claude-haiku-4-5"

        result = call_anthropic(**params)

        assert result == {
            "text": "Risposta Anthropic",
            "input_tokens": 10,
            "output_tokens": 5,
        }

        kwargs = mock_client.messages.create.call_args.kwargs
        assert "thinking" in kwargs
        assert kwargs["thinking"]["type"] == "enabled"
        assert "temperature" not in kwargs
        assert kwargs["thinking"]["budget_tokens"] >= 1024

    def test_opus_senza_reasoning_passa_temperature(self, mocker, sample_params):
        mock_anthropic_cls = mocker.patch("ns_llm.inference.providers.anthropic_provider.Anthropic")
        mock_client = mock_anthropic_cls.return_value

        mock_block = mocker.MagicMock()
        mock_block.type = "text"
        mock_block.text = "ok"
        mock_response = mocker.MagicMock()
        mock_response.content = [mock_block]
        mock_response.usage.input_tokens = 1
        mock_response.usage.output_tokens = 1
        mock_client.messages.create.return_value = mock_response

        params = sample_params.copy()
        params["model"] = "claude-opus-4-1"
        params["reasoning"] = False

        call_anthropic(**params)

        kwargs = mock_client.messages.create.call_args.kwargs
        assert kwargs["temperature"] == pytest.approx(0.7)
        assert kwargs["thinking"] == {"type": "disabled"}

    def test_opus_con_reasoning_emette_warning_e_adaptive(self, mocker, sample_params):
        mock_anthropic_cls = mocker.patch("ns_llm.inference.providers.anthropic_provider.Anthropic")
        mock_client = mock_anthropic_cls.return_value

        mock_block = mocker.MagicMock()
        mock_block.type = "text"
        mock_block.text = "ok"
        mock_response = mocker.MagicMock()
        mock_response.content = [mock_block]
        mock_response.usage.input_tokens = 1
        mock_response.usage.output_tokens = 1
        mock_client.messages.create.return_value = mock_response

        params = sample_params.copy()
        params["model"] = "claude-opus-4-1"
        params["reasoning"] = True

        with pytest.warns(UserWarning, match="opus"):
            call_anthropic(**params)

        kwargs = mock_client.messages.create.call_args.kwargs
        assert kwargs["thinking"]["type"] == "adaptive"
        assert "temperature" not in kwargs

    def test_risposta_senza_testo_emette_warning(self, mocker, sample_params):
        mock_anthropic_cls = mocker.patch("ns_llm.inference.providers.anthropic_provider.Anthropic")
        mock_client = mock_anthropic_cls.return_value

        mock_block = mocker.MagicMock()
        mock_block.type = "thinking"
        mock_block.text = "ragionamento..."
        mock_response = mocker.MagicMock()
        mock_response.content = [mock_block]
        mock_response.usage.input_tokens = 1
        mock_response.usage.output_tokens = 1
        mock_client.messages.create.return_value = mock_response

        with pytest.warns(UserWarning, match="senza blocchi di testo"):
            result = call_anthropic(**sample_params)

        assert result["text"] == ""

    def test_formato_risposta(self, mocker, sample_params):
        mock_anthropic_cls = mocker.patch("ns_llm.inference.providers.anthropic_provider.Anthropic")
        mock_client = mock_anthropic_cls.return_value

        mock_block = mocker.MagicMock()
        mock_block.type = "text"
        mock_block.text = "Risposta Anthropic"

        mock_response = mocker.MagicMock()
        mock_response.content = [mock_block]
        mock_response.usage.input_tokens = 5
        mock_response.usage.output_tokens = 3

        mock_client.messages.create.return_value = mock_response

        result = call_anthropic(**sample_params)

        assert "text" in result
        assert "input_tokens" in result
        assert "output_tokens" in result
        assert isinstance(result["text"], str)
        assert isinstance(result["input_tokens"], int)
        assert isinstance(result["output_tokens"], int)
