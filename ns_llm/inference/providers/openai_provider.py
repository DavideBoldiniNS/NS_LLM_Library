import openai

def call_openai(
        model: str,
        max_output_tokens: int,
        temperature: float,
        system_prompt: str,
        user_prompt: str,
        reasoning: bool,
        api_key: str
) -> dict:

    client = openai.OpenAI(api_key=api_key)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    request = {
        "model": model,
        "messages": messages,
        "max_completion_tokens": max_output_tokens,
    }

    request["temperature"] = 1

    if reasoning:
        request["reasoning_effort"] = "xhigh"
    else:
        request["reasoning_effort"] = "none"

    response = client.chat.completions.create(**request)

    return {
        "text": response.choices[0].message.content,
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
    }
