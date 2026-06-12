from openrouter import OpenRouter


def call_openrouter_rerank(
    model: str,
    query: str,
    documents: list[dict],
    top_n: int | None,
    api_key: str,
) -> list[dict]:
    """Invoke OpenRouter reranking.

    The OpenRouter SDK is used as a context manager that owns the underlying
    HTTP client.

    Args:
        model: OpenRouter reranking model (e.g. ``cohere/rerank-v3.5``).
        query: Search query.
        documents: List of dicts with ``id`` and ``text``.
        top_n: Maximum results to return; ``None`` returns all.
        api_key: OpenRouter API key.

    Returns:
        Sorted list of dicts with ``id``, ``text``, ``score``, ``original_index``.

    Raises:
        openrouter.errors.OpenRouterError: any error from the SDK.
    """
    with OpenRouter(api_key=api_key) as client:
        texts = [doc["text"] for doc in documents]

        kwargs = {
            "model": model,
            "query": query,
            "documents": texts,
        }
        if top_n is not None:
            kwargs["top_n"] = top_n

        response = client.rerank.rerank(**kwargs)

        return [
            {
                "id": documents[r.index]["id"],
                "text": documents[r.index]["text"],
                "score": r.relevance_score,
                "original_index": r.index,
            }
            for r in response.results
        ]
