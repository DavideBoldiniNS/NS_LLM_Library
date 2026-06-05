from .providers.openrouter_provider import call_openrouter
from .providers.together_provider import call_together

EMBEDDING_PROVIDER_NAMES = {
    "openrouter": "call_openrouter",
    "together": "call_together",
}


def _validate_input(
    provider: str,
    model: str,
    text: str,
    dimensions: int,
    api_key: str,
) -> None:
    if not isinstance(provider, str) or not provider.strip():
        raise ValueError("Il parametro 'provider' deve essere una stringa non vuota.")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("Il parametro 'model' deve essere una stringa non vuota.")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Il parametro 'text' deve essere una stringa non vuota.")
    if not isinstance(api_key, str) or not api_key.strip():
        raise ValueError("Il parametro 'api_key' deve essere una stringa non vuota.")
    if (
        not isinstance(dimensions, int)
        or isinstance(dimensions, bool)
        or dimensions < 0
    ):
        raise ValueError(
            "Il parametro 'dimensions' deve essere un intero non negativo."
        )


def generate_embedding(
    provider: str,
    model: str,
    text: str,
    input_type: str,
    dimensions: int,
    api_key: str,
    supports_input_type: bool = False,
    supports_dimensions: bool = False,
) -> dict:
    """Dispatch an embedding call to the requested provider.

    Args:
        provider: Case-insensitive provider name (``together`` or
            ``openrouter``).
        model: Provider-specific model identifier.
        text: Input string to embed; must be non-empty after stripping.
        input_type: Hint forwarded only when ``supports_input_type=True``;
            recognized values are ``search_query`` and ``search_document``
            (Together applies the ``query:`` / ``passage:`` prefix), plus
            any other value passed through verbatim.
        dimensions: Target embedding size; forwarded only when
            ``supports_dimensions=True`` and strictly positive.
        api_key: Provider API key (non-empty string).
        supports_input_type: Set to ``True`` if the chosen model actually
            accepts the ``input_type`` parameter. Defaults to ``False`` so
            the parameter is omitted on models that would reject it.
        supports_dimensions: Set to ``True`` if the chosen model supports
            the ``dimensions`` parameter. Defaults to ``False``.

    Returns:
        A dict with ``embedding`` (list of floats) and ``input_tokens``.

    Raises:
        ValueError: if ``provider`` is unknown or any input fails validation.
    """
    _validate_input(provider, model, text, dimensions, api_key)

    handler_name = EMBEDDING_PROVIDER_NAMES.get(provider.lower())
    if handler_name is None:
        valid = ", ".join(sorted(EMBEDDING_PROVIDER_NAMES))
        raise ValueError(
            f"Provider '{provider}' non supportato. Valori validi: {valid}."
        )

    handler = globals()[handler_name]
    return handler(
        model=model,
        text=text,
        input_type=input_type,
        dimensions=dimensions,
        supports_input_type=supports_input_type,
        supports_dimensions=supports_dimensions,
        api_key=api_key,
    )
