import pytest

#   =====================
#   Fixture per INFERENCE
#   =====================
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

#   =====================
#   Fixture per EMBEDDING
#   =====================
@pytest.fixture
def sample_embedding_params():
    return {
        "model": "openai/text-embedding-3-small",
        "text": "test",
        "input_type": "search_query",
        "dimensions": 0,
        "api_key": "fake-key-12345",
    }

@pytest.fixture
def sample_embedding_response():
    return {
        "embedding": [0.1, 0.2, 0.3],
        "input_tokens": 10,
    }