from ns_llm.inference.providers.ollama_provider import call_ollama

class TestCallOllama:
    def test_base_ollama(self, mocker, sample_params):
        mock_ollama_cls = mocker.patch("ns_llm.inference.providers.ollama_provider.Client")
        mock_client = mock_ollama_cls.return_value
        
        mock_client.chat.return_value = {
            "message": {"content": "Risposta da Ollama"},
            "prompt_eval_count": 15,
            "eval_count": 8,
        }
        

        result = call_ollama(**sample_params)
        
        assert result["text"] == "Risposta da Ollama"
        assert result["input_tokens"] == 15
        assert result["output_tokens"] == 8
        
        options = mock_client.chat.call_args.kwargs["options"]
        assert options["num_predict"] == 100 
        assert options["temperature"] == 0.7

    def test_ollama_reasoning(self, mocker, sample_params):
        mock_ollama_cls = mocker.patch("ns_llm.inference.providers.ollama_provider.Client")
        mock_client = mock_ollama_cls.return_value
        mock_client.chat.return_value = {
            "message": {"content": "Ragionamento Ollama"},
            "prompt_eval_count": 5,
            "eval_count": 5,
        }

        params = sample_params.copy()
        params["reasoning"] = True
    
        
        call_ollama(**params)
        
        options = mock_client.chat.call_args.kwargs["options"]
        assert options["reasoning"] is True

               
    def test_formato_risposta(self, mocker, sample_params):
        
        mock_ollama_cls = mocker.patch(
            "ns_llm.inference.providers."
            "ollama_provider.Client"
        )
        mock_client = mock_ollama_cls.return_value
        mock_client.chat.return_value = {
            "message": {"content": "Test ollama"},
            "prompt_eval_count": 10,
            "eval_count": 5,
        }
        mock_choice = mocker.MagicMock()
        mock_choice.message.content = "Test"
        mock_response = mocker.MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage.prompt_tokens = 5
        mock_response.usage.completion_tokens = 3
        mock_client.chat.completions.create.return_value = (
            mock_response
        )

        result = call_ollama(**sample_params)

        assert "text" in result
        assert "input_tokens" in result
        assert "output_tokens" in result
        assert isinstance(result["text"], str)
        assert isinstance(result["input_tokens"], int)
        assert isinstance(result["output_tokens"], int)
