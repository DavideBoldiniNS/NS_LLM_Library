import warnings

import pytest

from ns_llm.inference.providers.openai_provider import call_openai


class TestCallOpenAI:
    def test_senza_reasoning(self, mocker, sample_params):
        mock_openai_cls = mocker.patch("ns_llm.inference.providers.openai_provider.openai.OpenAI")
        mock_client = mock_openai_cls.return_value

        mock_choice = mocker.MagicMock()
        mock_choice.message.content = "Ciao dal mock!"
        mock_response = mocker.MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_client.chat.completions.create.return_value = mock_response

        result = call_openai(**sample_params)
        assert result["text"] == "Ciao dal mock!"
        assert result["input_tokens"] == 10
        assert result["output_tokens"] == 5

    def test_con_reasoning(self, mocker, sample_params):
        mock_openai_cls = mocker.patch("ns_llm.inference.providers.openai_provider.openai.OpenAI")
        mock_client = mock_openai_cls.return_value

        params = sample_params.copy()
        params["reasoning"] = True

        mock_choice = mocker.MagicMock()
        mock_choice.message.content = "Risposta ragionata"
        mock_response = mocker.MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage.prompt_tokens = 20
        mock_response.usage.completion_tokens = 15
        mock_client.chat.completions.create.return_value = mock_response

        call_openai(**params)

        kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert kwargs["temperature"] == pytest.approx(0.7)
        assert kwargs["reasoning_effort"] == "high"
        assert kwargs["max_completion_tokens"] == 100
        assert kwargs["model"] == "test-model"

    def test_temperature_propagata(self, mocker, sample_params):
        mock_openai_cls = mocker.patch("ns_llm.inference.providers.openai_provider.openai.OpenAI")
        mock_client = mock_openai_cls.return_value

        mock_choice = mocker.MagicMock()
        mock_choice.message.content = "ok"
        mock_response = mocker.MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage.prompt_tokens = 1
        mock_response.usage.completion_tokens = 1
        mock_client.chat.completions.create.return_value = mock_response

        params = sample_params.copy()
        params["temperature"] = 1.5

        call_openai(**params)

        kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert kwargs["temperature"] == pytest.approx(1.5)
        assert kwargs["reasoning_effort"] is None

    def test_reasoning_false_imposta_explicit_off(self, mocker, sample_params):
        mock_openai_cls = mocker.patch("ns_llm.inference.providers.openai_provider.openai.OpenAI")
        mock_client = mock_openai_cls.return_value

        mock_choice = mocker.MagicMock()
        mock_choice.message.content = "ok"
        mock_response = mocker.MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage.prompt_tokens = 1
        mock_response.usage.completion_tokens = 1
        mock_client.chat.completions.create.return_value = mock_response

        call_openai(**sample_params)

        kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert "reasoning_effort" in kwargs
        assert kwargs["reasoning_effort"] is None

    def test_reasoning_true_usa_high_non_xhigh(self, mocker, sample_params):
        mock_openai_cls = mocker.patch("ns_llm.inference.providers.openai_provider.openai.OpenAI")
        mock_client = mock_openai_cls.return_value

        mock_choice = mocker.MagicMock()
        mock_choice.message.content = "ok"
        mock_response = mocker.MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage.prompt_tokens = 1
        mock_response.usage.completion_tokens = 1
        mock_client.chat.completions.create.return_value = mock_response

        params = sample_params.copy()
        params["reasoning"] = True

        call_openai(**params)

        kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert kwargs["reasoning_effort"] in ("low", "medium", "high")
        assert kwargs["reasoning_effort"] != "xhigh"

    def test_content_none_emette_warning(self, mocker, sample_params):
        mock_openai_cls = mocker.patch("ns_llm.inference.providers.openai_provider.openai.OpenAI")
        mock_client = mock_openai_cls.return_value

        mock_choice = mocker.MagicMock()
        mock_choice.message.content = None
        mock_response = mocker.MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage.prompt_tokens = 1
        mock_response.usage.completion_tokens = 1
        mock_client.chat.completions.create.return_value = mock_response

        with pytest.warns(UserWarning, match="senza contenuto testuale"):
            result = call_openai(**sample_params)

        assert result["text"] == ""

    def test_formato_risposta(self, mocker, sample_params):
        mock_openai_cls = mocker.patch(
            "ns_llm.inference.providers."
            "openai_provider.openai.OpenAI"
        )
        mock_client = mock_openai_cls.return_value
        mock_choice = mocker.MagicMock()
        mock_choice.message.content = "Test"
        mock_response = mocker.MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage.prompt_tokens = 5
        mock_response.usage.completion_tokens = 3
        mock_client.chat.completions.create.return_value = mock_response

        result = call_openai(**sample_params)

        assert "text" in result
        assert "input_tokens" in result
        assert "output_tokens" in result
        assert isinstance(result["text"], str)
        assert isinstance(result["input_tokens"], int)
        assert isinstance(result["output_tokens"], int)
