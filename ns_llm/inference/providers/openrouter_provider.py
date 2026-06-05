from openrouter import OpenRouter

def call_openrouter(
    model: str,
    max_output_tokens: int,
    temperature: float,
    system_prompt: str,
    user_prompt: str,
    reasoning: bool,
    api_key: str
) -> dict:
    with OpenRouter(api_key=api_key) as client:
        messages=[
            {"role":"system","content":system_prompt},
            {"role":"user","content":user_prompt},
        ]
        extra_args = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_output_tokens,
        }
        if reasoning:
            extra_args["reasoning"] = {"enabled":True}
        response=client.chat.send(**extra_args)
        text=response.choices[0].message.content
        input_tokens=response.usage.prompt_tokens
        output_tokens=response.usage.completion_tokens
        return {
            "text":text,
            "input_tokens":input_tokens,
            "output_tokens":output_tokens,
        }
    