from openrouter import OpenRouter
import requests

def call_openrouter(
    model: str,
    text: str,
    input_type: str,
    dimensions: int,
    api_key: str
) -> dict:
    url = "https://openrouter.ai/api/v1/embeddings"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model,
        "input": text
    }
    
    if input_type:
        payload["input_type"] = input_type
        
    if dimensions > 0:
        payload["dimensions"] = dimensions

    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    
    response_data = response.json()
    
    embedding_vector = response_data["data"][0]["embedding"]
    tokens_used = response_data["usage"]["prompt_tokens"]
    
    return {
        "embedding": embedding_vector,
        "input_tokens": tokens_used
    }