from ns_llm.inference.providers.openrouter_provider import call_openrouter
class TestCallOpenRouter:
    def test_senza_reasoning(self, mocker,sample_params):
        mock_openrouter_cls=mocker.patch(
            "ns_llm.inference.providers."
            "openrouter_provider.OpenRouter"
        )
        mock_client=(
            mock_openrouter_cls.return_value
            .__enter__.return_value
        )
        mock_choice=mocker.MagicMock()
        mock_message=mocker.MagicMock()
        mock_message.content="Ciao dal mock!"
        mock_choice.message=mock_message
        mock_response=mocker.MagicMock()
        mock_response.choices=[mock_choice]
        mock_usage=mocker.MagicMock()
        mock_usage.prompt_tokens=10
        mock_usage.completion_tokens=5
        mock_response.usage=mock_usage
        mock_client.chat.send.return_value=mock_response
        result=call_openrouter(**sample_params)
        assert result=={
            "text":"Ciao dal mock!",
            "output_tokens":5,
            "input_tokens":10,
        }
        mock_openrouter_cls.assert_called_once_with(api_key="fake-key-12345")
        chiamata=mock_client.chat.send
        chiamata.assert_called_once()
        assert chiamata.call_args.kwargs["temperature"] == 0.7
        
        assert "reasoning" not in chiamata.call_args.kwargs
        
    def test_con_reasoning(self,mocker,sample_params):
        mock_openrouter_cls=mocker.patch(
            "ns_llm.inference.providers."
            "openrouter_provider.OpenRouter"
        )
        mock_client=(
            mock_openrouter_cls.return_value
            .__enter__.return_value
        )
        mock_choice=mocker.MagicMock()
        mock_message=mocker.MagicMock()
        mock_message.content="Risposta ragionata"
        mock_choice.message=mock_message
        mock_response=mocker.MagicMock()
        mock_usage=mocker.MagicMock()
        mock_response.choices=[mock_choice]
        mock_usage.prompt_tokens=20
        mock_usage.completion_tokens=15
        mock_response.usage=mock_usage
        mock_client.chat.send.return_value=mock_response
        params=sample_params.copy()
        params["reasoning"]={"enabled":True}
        result=call_openrouter(**params)
        assert result["text"]=="Risposta ragionata"
        assert result["input_tokens"]==20
        assert result["output_tokens"]==15
        chiamata=mock_client.chat.send
        chiamata.assert_called_once()

        assert chiamata.call_args.kwargs["temperature"] == 0.7
        
        assert chiamata.call_args.kwargs["reasoning"] == {"enabled":True}
        
    def test_formato_risposta(self, mocker, sample_params):
        mock_openrouter_cls=mocker.patch(
            "ns_llm.inference.providers."
            "openrouter_provider.OpenRouter"
        )
        mock_client=(
            mock_openrouter_cls.return_value
            .__enter__.return_value
        )
        mock_choice=mocker.MagicMock()
        mock_message=mocker.MagicMock()
        mock_message.content="Test"
        mock_choice.message=mock_message
        mock_response=mocker.MagicMock()
        mock_response.choices=[mock_choice]
        mock_usage=mocker.MagicMock()
        mock_usage.prompt_tokens=5
        mock_usage.completion_tokens=3
        mock_response.usage=mock_usage
        mock_client.chat.send.return_value=mock_response
        result=call_openrouter(**sample_params)
        assert "text" in result
        assert "input_tokens" in result
        assert "output_tokens" in result
        assert isinstance(result["text"],str)
        assert isinstance(result["input_tokens"],int)
        assert isinstance(result["output_tokens"],int)