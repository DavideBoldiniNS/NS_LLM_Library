from .providers.anthropic_provider import call_anthropic
from .providers.openai_provider import call_openai
from .providers.together_provider import call_together
from .providers.ollama_provider import call_ollama
from .providers.openrouter_provider import call_openrouter

def generate_response(
    provider: str,
    model: str,
    max_output_tokens: int,
    temperature: float,
    system_prompt: str,
    user_prompt: str,
    reasoning: bool,
    api_key: str
) -> dict:

    match provider:
        case "anthropic": handler = call_anthropic
        case "openai": handler = call_openai
        case "together": handler = call_together
        case "ollama": handler = call_ollama
        case "openrouter": handler = call_openrouter
        case _: raise ValueError(f"Provider '{provider}' non supportato.")

    return handler(
        model=model,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        reasoning=reasoning,
        api_key=api_key
    )