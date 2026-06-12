import json
from ns_llm.inference.client import generate_response


def call_openrouter_structured(
    model: str,
    max_output_tokens: int,
    temperature: float,
    system_prompt: str,
    user_prompt: str,
    api_key: str,
    schema: dict,
    reasoning: bool = False,
) -> str:
    try:
        return _call_openrouter_native(
            model, max_output_tokens, temperature,
            system_prompt, user_prompt, api_key, schema, reasoning,
        )
    except Exception:
        return _call_openrouter_fallback(
            model, max_output_tokens, temperature,
            system_prompt, user_prompt, api_key, schema, reasoning,
        )


def _call_openrouter_native(
    model: str,
    max_output_tokens: int,
    temperature: float,
    system_prompt: str,
    user_prompt: str,
    api_key: str,
    schema: dict,
    reasoning: bool = False,
) -> str:
    from openrouter import OpenRouter

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    with OpenRouter(api_key=api_key) as client:
        response = client.chat.send(
            model=model,
            messages=messages,
            max_tokens=max_output_tokens,
            temperature=temperature,
            reasoning={"effort": "high" if reasoning else "none"},
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "strict": True,
                    "schema": schema,
                },
            },
        )
        return response.choices[0].message.content or ""


def _call_openrouter_fallback(
    model: str,
    max_output_tokens: int,
    temperature: float,
    system_prompt: str,
    user_prompt: str,
    api_key: str,
    schema: dict,
    reasoning: bool = False,
) -> str:
    full_prompt = (
        f"{system_prompt}\n\n"
        f"Rispondi SOLO con un JSON valido che segue questo schema:\n"
        f"{json.dumps(schema, indent=2)}"
    )
    result = generate_response(
        provider="openrouter",
        model=model,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        system_prompt=full_prompt,
        user_prompt=user_prompt,
        reasoning=reasoning,
        api_key=api_key,
    )
    return result["text"]
