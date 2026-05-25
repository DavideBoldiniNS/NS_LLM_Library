from .providers.anthropic_provider import call_anthropic
from .providers.openai_provider import call_openai
from .providers.together_provider import call_together
from .providers.ollama_provider import call_ollama
from .providers.openrouter_provider import call_openrouter

_providers = {
    "anthropic": call_anthropic,
    "openai": call_openai,
    "together": call_together,
    "ollama": call_ollama,
    "openrouter": call_openrouter
}

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

    handler = _providers[provider]

    return handler(
        model=model,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        reasoning=reasoning,
        api_key=api_key
    )