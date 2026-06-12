from .providers.anthropic_provider import count_anthropic_tokens
from .providers.cohere_provider import count_cohere_tokens
from .providers.ollama_provider import count_ollama_tokens
from .providers.openai_provider import count_openai_tokens
from .providers._estimation import estimate_tokens

PROVIDER_NAMES = {
    "openai": "count_openai_tokens",
    "anthropic": "count_anthropic_tokens",
    "cohere": "count_cohere_tokens",
    "ollama": "count_ollama_tokens",
    "together": "estimate_tokens",
    "openrouter": "estimate_tokens",
}


def _validate_input(
    provider: str,
    model: str,
    text: str,
) -> None:
    if not isinstance(provider, str) or not provider.strip():
        raise ValueError("Il parametro 'provider' deve essere una stringa non vuota.")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("Il parametro 'model' deve essere una stringa non vuota.")
    if not isinstance(text, str):
        raise ValueError("Il parametro 'text' deve essere una stringa.")


def count_tokens(
    provider: str,
    model: str,
    text: str,
    api_key: str = "",
) -> int:
    """Count tokens for a given text using the requested provider's tokenizer.

    Args:
        provider: Case-insensitive provider name (``openai``, ``anthropic``,
            ``cohere``, ``ollama``, ``together``, ``openrouter``).
        model: Model identifier (used to select the correct tokenizer).
        text: Text to tokenize. Empty string returns 0.
        api_key: Provider API key. Required for ``anthropic`` and ``cohere``;
            ignored for other providers.

    Returns:
        Number of tokens as an integer.

    Raises:
        ValueError: if ``provider`` is unknown or any input fails validation.
    """
    _validate_input(provider, model, text)

    handler_name = PROVIDER_NAMES.get(provider.lower())
    if handler_name is None:
        valid = ", ".join(sorted(PROVIDER_NAMES))
        raise ValueError(
            f"Provider '{provider}' non supportato. Valori validi: {valid}."
        )

    handler = globals()[handler_name]
    if provider.lower() in ("anthropic", "cohere"):
        return handler(model=model, text=text, api_key=api_key)
    if provider.lower() == "ollama":
        return handler(model=model, text=text)
    if provider.lower() in ("together", "openrouter"):
        return handler(text=text)
    return handler(model=model, text=text)
