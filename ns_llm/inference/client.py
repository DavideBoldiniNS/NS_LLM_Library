from .providers.anthropic_provider import call_anthropic
from .providers.ollama_provider import call_ollama
from .providers.openai_provider import call_openai
from .providers.openrouter_provider import call_openrouter
from .providers.together_provider import call_together

PROVIDER_NAMES = {
    "anthropic": "call_anthropic",
    "ollama": "call_ollama",
    "openai": "call_openai",
    "openrouter": "call_openrouter",
    "together": "call_together",
}


def _validate_input(
    provider: str,
    model: str,
    max_output_tokens: int,
    temperature: float,
    api_key: str,
) -> None:
    if not isinstance(provider, str) or not provider.strip():
        raise ValueError("Il parametro 'provider' deve essere una stringa non vuota.")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("Il parametro 'model' deve essere una stringa non vuota.")
    if not isinstance(api_key, str) or not api_key.strip():
        raise ValueError("Il parametro 'api_key' deve essere una stringa non vuota.")
    if (
        not isinstance(max_output_tokens, int)
        or isinstance(max_output_tokens, bool)
        or max_output_tokens <= 0
    ):
        raise ValueError(
            "Il parametro 'max_output_tokens' deve essere un intero positivo."
        )
    if not isinstance(temperature, (int, float)) or isinstance(temperature, bool):
        raise ValueError("Il parametro 'temperature' deve essere un numero.")
    if temperature < 0 or temperature > 2:
        raise ValueError(
            "Il parametro 'temperature' deve essere compreso tra 0 e 2."
        )


def generate_response(
    provider: str,
    model: str,
    max_output_tokens: int,
    temperature: float,
    system_prompt: str,
    user_prompt: str,
    reasoning: bool,
    api_key: str,
) -> dict:
    """Dispatch an inference call to the requested provider.

    Args:
        provider: Case-insensitive provider name (``openai``, ``anthropic``,
            ``together``, ``ollama``, ``openrouter``).
        model: Provider-specific model identifier.
        max_output_tokens: Upper bound for the generated tokens (positive int).
        temperature: Sampling temperature in ``[0, 2]``.
        system_prompt: System message for the conversation.
        user_prompt: User message.
        reasoning: Forwarded to the provider as a reasoning toggle.
        api_key: Provider API key (non-empty string).

    Returns:
        A dict with ``text``, ``input_tokens`` and ``output_tokens`` as
        returned by the selected provider.

    Raises:
        ValueError: if ``provider`` is unknown or any input fails validation.
    """
    _validate_input(provider, model, max_output_tokens, temperature, api_key)

    handler_name = PROVIDER_NAMES.get(provider.lower())
    if handler_name is None:
        valid = ", ".join(sorted(PROVIDER_NAMES))
        raise ValueError(
            f"Provider '{provider}' non supportato. Valori validi: {valid}."
        )

    handler = globals()[handler_name]
    return handler(
        model=model,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        reasoning=reasoning,
        api_key=api_key,
    )
