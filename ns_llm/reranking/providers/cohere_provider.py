from cohere import Client


def call_cohere_rerank(
    model: str,
    query: str,
    documents: list[dict],
    top_n: int | None,
    api_key: str,
) -> list[dict]:
    """Invoke Cohere reranking via the v2 API.

    Args:
        model: Cohere reranking model (e.g. ``rerank-v4.0-pro``).
        query: Search query.
        documents: List of dicts with ``id`` and ``text``.
        top_n: Maximum results to return; ``None`` returns all.
        api_key: Cohere API key.

    Returns:
        Sorted list of dicts with ``id``, ``text``, ``score``, ``original_index``.

    Raises:
        cohere.errors.CohereError: any error from the Cohere SDK.
    """
    client = Client(token=api_key)

    texts = [doc["text"] for doc in documents]

    kwargs = {
        "model": model,
        "query": query,
        "documents": texts,
    }
    if top_n is not None:
        kwargs["top_n"] = top_n

    response = client.v2.rerank(**kwargs)

    return [
        {
            "id": documents[r.index]["id"],
            "text": documents[r.index]["text"],
            "score": r.relevance_score,
            "original_index": r.index,
        }
        for r in response.results
    ]
