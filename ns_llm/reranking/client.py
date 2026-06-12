from .providers.cohere_provider import call_cohere_rerank
from .providers.jina_provider import call_jina_rerank
from .providers.openrouter_provider import call_openrouter_rerank
from .providers.together_provider import call_together_rerank

PROVIDER_NAMES = {
    "cohere": "call_cohere_rerank",
    "jina": "call_jina_rerank",
    "openrouter": "call_openrouter_rerank",
    "together": "call_together_rerank",
}


def _validate_input(
    provider: str,
    model: str,
    query: str,
    documents: list,
    api_key: str,
) -> None:
    if not isinstance(provider, str) or not provider.strip():
        raise ValueError("Il parametro 'provider' deve essere una stringa non vuota.")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("Il parametro 'model' deve essere una stringa non vuota.")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Il parametro 'query' deve essere una stringa non vuota.")
    if not isinstance(documents, list) or len(documents) == 0:
        raise ValueError("Il parametro 'documents' deve essere una lista non vuota.")
    for i, doc in enumerate(documents):
        if not isinstance(doc, dict) or "id" not in doc or "text" not in doc:
            raise ValueError(
                f"Ogni documento deve essere un dict con 'id' e 'text'. "
                f"Elemento {i} non valido."
            )
    if not isinstance(api_key, str) or not api_key.strip():
        raise ValueError("Il parametro 'api_key' deve essere una stringa non vuota.")


def rerank(
    provider: str,
    model: str,
    query: str,
    documents: list[dict],
    top_n: int | None = None,
    api_key: str = "",
) -> list[dict]:
    """Dispatch a reranking call to the requested provider.

    Args:
        provider: Provider name (``cohere``, ``together``, ``jina``,
            ``openrouter``).
        model: Provider-specific reranking model identifier.
        query: Search query to rank documents against.
        documents: List of dicts, each with ``id`` (str) and ``text`` (str).
        top_n: Maximum number of results to return. ``None`` returns all
            documents sorted by score.
        api_key: Provider API key (non-empty string).

    Returns:
        A list of dicts sorted by ``score`` descending. Each dict contains
        ``id``, ``text``, ``score`` (float) and ``original_index`` (int).

    Raises:
        ValueError: if ``provider`` is unknown or any input fails validation.
    """
    _validate_input(provider, model, query, documents, api_key)

    handler_name = PROVIDER_NAMES.get(provider.lower())
    if handler_name is None:
        valid = ", ".join(sorted(PROVIDER_NAMES))
        raise ValueError(
            f"Provider '{provider}' non supportato. Valori validi: {valid}."
        )

    handler = globals()[handler_name]
    return handler(
        model=model,
        query=query,
        documents=documents,
        top_n=top_n,
        api_key=api_key,
    )
