import json
from ns_llm.inference.client import generate_response


def call_together_structured(
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
        provider="together",
        model=model,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        system_prompt=full_prompt,
        user_prompt=user_prompt,
        reasoning=reasoning,
        api_key=api_key,
    )
    return result["text"]
