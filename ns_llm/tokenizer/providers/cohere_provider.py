from cohere import Client


def count_cohere_tokens(model: str, text: str, api_key: str) -> int:
    if not text:
        return 0
    client = Client(token=api_key)
    response = client.tokenize(text=text, model=model)
    return len(response.tokens)
