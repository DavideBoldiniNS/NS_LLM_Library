from together import Together


def call_together_rerank(
    model: str,
    query: str,
    documents: list[dict],
    top_n: int | None,
    api_key: str,
) -> list[dict]:
    """Invoke Together AI reranking.

    Args:
        model: Together reranking model (e.g. ``Salesforce/Llama-Rank-V1``).
        query: Search query.
        documents: List of dicts with ``id`` and ``text``.
        top_n: Maximum results to return; ``None`` returns all.
        api_key: Together API key.

    Returns:
        Sorted list of dicts with ``id``, ``text``, ``score``, ``original_index``.

    Raises:
        together.errors.APIError: any error from the Together SDK.
    """
    client = Together(api_key=api_key)

    texts = [doc["text"] for doc in documents]

    kwargs = {
        "model": model,
        "query": query,
        "documents": texts,
    }
    if top_n is not None:
        kwargs["top_n"] = top_n

    response = client.rerank.create(**kwargs)

    return [
        {
            "id": documents[r.index]["id"],
            "text": documents[r.index]["text"],
            "score": r.relevance_score,
            "original_index": r.index,
        }
        for r in response.results
    ]
