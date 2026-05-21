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

    if reasoning:
        request["reasoning_effort"] = "medium"
        request["temperature"] = 1 
    else:
        request["temperature"] = temperature

    response = client.chat.completions.create(**request)

    return {
        "text": response.choices[0].message.content,
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
    }

print(call_openai(
    model="gpt-5.4-nano",
    max_output_tokens=100,
    temperature=0.7,
    system_prompt="Sei un mio caro amico",
    user_prompt="Ciao, come stai?",
    reasoning=True,
    api_key="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"
))