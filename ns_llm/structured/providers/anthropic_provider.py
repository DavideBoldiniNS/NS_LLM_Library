import json
from anthropic import Anthropic
from ns_llm.inference.client import generate_response


def call_anthropic_structured(
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
        return _call_anthropic_native(
            model, max_output_tokens, temperature,
            system_prompt, user_prompt, api_key, schema,
        )
    except Exception:
        return _call_anthropic_fallback(
            model, max_output_tokens, temperature,
            system_prompt, user_prompt, api_key, schema, reasoning,
        )


def _call_anthropic_native(
    model: str,
    max_output_tokens: int,
    temperature: float,
    system_prompt: str,
    user_prompt: str,
    api_key: str,
    schema: dict,
) -> str:
    client = Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model,
        max_tokens=max_output_tokens,
        temperature=temperature,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
        tools=[
            {
                "name": "output",
                "description": "Generate structured output matching the provided schema",
                "input_schema": schema,
            }
        ],
        tool_choice={"type": "tool", "name": "output"},
    )
    for block in response.content:
        if getattr(block, "type", None) == "tool_use" and block.name == "output":
            return json.dumps(block.input)
    return ""


def _call_anthropic_fallback(
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
        provider="anthropic",
        model=model,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        system_prompt=full_prompt,
        user_prompt=user_prompt,
        reasoning=reasoning,
        api_key=api_key,
    )
    return result["text"]
