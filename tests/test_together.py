from ns_llm.inference.providers.together_provider import call_together

class TestCallTogether:
    def test_senza_reasoning(self,mocker,sample_params):
        mock_together_cls=mocker.patch(
            "ns_llm.inference.providers."
            "together_provider.Together"
        )
        mock_client=mock_together_cls.return_value
        mock_choice=mocker.MagicMock()
        mock_choice.message.content="Ciao dal mock!"
        mock_response=mocker.MagicMock()
        mock_response.choices=[mock_choice]
        mock_response.usage.prompt_tokens=10
        mock_response.usage.completion_tokens=5
        mock_client.chat.completions.create.return_value=(
            mock_response
        )
        result=call_together(**sample_params)
        assert result=={
            "text":"Ciao dal mock!",
            "input_tokens":10,
            "output_tokens":5,
        }
        mock_together_cls.assert_called_once_with(
            api_key="fake-key-12345"
        )
        chiamata=mock_client.chat.completions.create
        chiamata.assert_called_once()
        assert(
            chiamata.call_args.kwargs["temperature"]
            ==1
        )
        assert(
            chiamata.call_args.kwargs["max_tokens"]
            ==100
        )
        assert(
            "max_output_tokens"
            not in chiamata.call_args.kwargs
        )
        assert(
            "reasoning"
            not in chiamata.call_args.kwargs
        )
    def test_con_reasoning(self, mocker,sample_params):
        mock_together_cls=mocker.patch(
            "ns_llm.inference.providers."
            "together_provider.Together"
        )
        mock_client=mock_together_cls.return_value
        mock_choice=mocker.MagicMock()
        mock_choice.message.content="Risposta ragionata"
        mock_response=mocker.MagicMock()
        mock_response.choices=[mock_choice]
        mock_response.usage.prompt_tokens=20
        mock_response.usage.completion_tokens=15
        mock_client.chat.completions.create.return_value=(
            mock_response
        )
        params=sample_params.copy()
        params["reasoning"]=True
        result=call_together(**params)
        assert result["text"]=="Risposta ragionata"
        assert result["input_tokens"]==20
        assert result["output_tokens"]==15
        chiamata=mock_client.chat.completions.create
        assert(
            chiamata.call_args.kwargs["temperature"]
            ==1 #0.7
        )
        assert(
            chiamata.call_args.kwargs["max_output_tokens"]
            ==100
        )
        assert(
            chiamata.call_args.kwargs["reasoning"]
            =={"enabled":True}
        )
        assert(
            "max_tokens"
            not in chiamata.call_args.kwargs
        )
    def test_formato_risposta(self,mocker,sample_params):
        mock_together_cls=mocker.patch(
            "ns_llm.inference.providers."
            "together_provider.Together"
        )
        mock_client=mock_together_cls.return_value
        mock_choice=mocker.MagicMock()
        mock_choice.message.content="Test"
        mock_response=mocker.MagicMock()
        mock_response.choices=[mock_choice]
        mock_response.usage.prompt_tokens=5
        mock_response.usage.completion_tokens=3
        mock_client.chat.completions.create.return_value=(
            mock_response
        )
        result=call_together(**sample_params)
        assert "text" in result
        assert "input_tokens" in result
        assert "output_tokens" in result
        assert isinstance(result["text"],str)
        assert isinstance(result["input_tokens"],int)
        assert isinstance(result["output_tokens"],int)