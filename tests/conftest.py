import pytest

@pytest.fixture
def sample_params():
    return {
        "model": "test-model",
        "max_output_tokens": 100,
        "temperature": 0.7,
        "system_prompt": "Sei un assistente.",
        "user_prompt": "Ciao!",
        "reasoning": False,
        "api_key": "fake-key-12345",
    }

@pytest.fixture
def sample_response():
    return {
        "text": "Risposta di test.",
        "input_tokens": 10,
        "output_tokens": 5,
    }