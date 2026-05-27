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

    def test_con_reasoning(self, mocker, sample_params):
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

        mock_param = sample_params.copy()
        mock_param["reasoning"] = True

        result = call_anthropic(**sample_params)

        assert result == {
                "text": "Risposta Anthropic",
                "input_tokens": 10,
                "output_tokens": 5,
                }
        
        mock_anthropic_cls.assert_called_once_with(api_key="fake-key-12345" )

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