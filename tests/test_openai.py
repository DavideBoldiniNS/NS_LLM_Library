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
        
        args = mock_client.chat.completions.create.call_args.kwargs
        assert args["temperature"] == 1
        assert args["reasoning_effort"] == "xhigh"

    def test_formato_risposta(self, mocker, sample_params):
        """Verifica che il dizionario restituito
        abbia le chiavi e i tipi corretti."""
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
        mock_client.chat.completions.create.return_value = (
            mock_response
        )

        result = call_openai(**sample_params)

        assert "text" in result
        assert "input_tokens" in result
        assert "output_tokens" in result
        assert isinstance(result["text"], str)
        assert isinstance(result["input_tokens"], int)
        assert isinstance(result["output_tokens"], int)

