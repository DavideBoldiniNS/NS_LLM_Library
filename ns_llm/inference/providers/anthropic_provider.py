from anthropic import Anthropic

def call_anthropic(
    model: str,
    max_output_tokens: int,
    temperature: float,
    system_prompt: str,
    user_prompt: str,
    reasoning: bool,
    api_key: str
) -> dict:
    client = Anthropic(api_key=api_key)

    kwargs = {
        "model": model,
        "max_tokens": max_output_tokens,
        "temperature": temperature,
        "system": system_prompt,
        "messages": [
            {
                "role": "user",
                "content": user_prompt 
            }
        ]
    }

    if reasoning:
        kwargs["temperature"] = 1.0
        kwargs["thinking"] = {
            "type": "enabled",
            "budget_tokens": min(max_output_tokens//2, 2048)
        }
        
    response = client.messages.create(**kwargs)

    text = ""

    for block in response.content:
        if block.type == "text": text += block.text

    return {
        "text": text,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens
    }
