import requests


def call_jina_rerank(
    model: str,
    query: str,
    documents: list[dict],
    top_n: int | None,
    api_key: str,
) -> list[dict]:
    """Invoke Jina AI reranking via direct REST call.

    Jina has no dedicated Python SDK; this provider uses ``requests`` directly.

    Args:
        model: Jina reranking model (e.g. ``jina-reranker-v3``).
        query: Search query.
        documents: List of dicts with ``id`` and ``text``.
        top_n: Maximum results to return; ``None`` returns all.
        api_key: Jina API key.

    Returns:
        Sorted list of dicts with ``id``, ``text``, ``score``, ``original_index``.

    Raises:
        requests.exceptions.HTTPError: any HTTP error from the Jina API.
    """
    url = "https://api.jina.ai/v1/rerank"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    texts = [doc["text"] for doc in documents]

    payload = {
        "model": model,
        "query": query,
        "documents": texts,
        "return_documents": True,
    }
    if top_n is not None:
        payload["top_n"] = top_n

    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    data = response.json()

    return [
        {
            "id": documents[r["index"]]["id"],
            "text": documents[r["index"]]["text"],
            "score": r["relevance_score"],
            "original_index": r["index"],
        }
        for r in data["results"]
    ]
