from anthropic import Anthropic


def count_anthropic_tokens(model: str, text: str, api_key: str) -> int:
    if not text:
        return 0
    client = Anthropic(api_key=api_key)
    response = client.messages.count_tokens(
        model=model,
        messages=[{"role": "user", "content": text}],
    )
    return response.input_tokens
