import requests


def count_ollama_tokens(model: str, text: str, api_key: str = "") -> int:
    if not text:
        return 0
    url = "http://localhost:11434/api/tokenize"
    response = requests.post(url, json={"model": model, "prompt": text})
    response.raise_for_status()
    return len(response.json()["tokens"])
