from together import Together


def call_together(
    model: str,
    text: str,
    input_type: str,
    dimensions: int,
    supports_input_type: bool,
    supports_dimensions: bool,
    api_key: str,
) -> dict:
    """Invoke a Together AI embeddings model and normalize the response.

    The Together API accepts an optional ``input_type`` hint that some
    models use to specialize the embedding (e.g. ``search_query``,
    ``search_document``). For those models, the provider pre-pends the
    ``query:`` / ``passage:`` prefix required by the underlying model
    family (BGE / Moka / similar). The ``dimensions`` parameter is
    forwarded only when the model advertises support for it and a positive
    value is requested.

    Both ``supports_input_type`` and ``supports_dimensions`` are explicit
    caller signals: this avoids relying on the model name to infer
    capabilities and keeps the wire payload predictable.

    Args:
        model: Together model identifier.
        text: Input string to embed; must be non-empty.
        input_type: One of ``search_query``, ``search_document``, or any
            custom value. Sent as ``input_type`` and used to derive the
            ``query:`` / ``passage:`` prefix only when
            ``supports_input_type=True``.
        dimensions: Target embedding size; forwarded only when
            ``supports_dimensions=True`` and strictly positive.
        supports_input_type: Set to ``True`` to forward ``input_type``.
        supports_dimensions: Set to ``True`` to forward ``dimensions``.
        api_key: Together API key.

    Returns:
        A dict with ``embedding`` (list of floats) and ``input_tokens``.

    Raises:
        together.TogetherError: any error propagated from the Together SDK.
    """
    if input_type == "search_query":
        processed_text = f"query: {text}"
    elif input_type == "search_document":
        processed_text = f"passage: {text}"
    else:
        processed_text = text

    client = Together(api_key=api_key)

    params = {
        "model": model,
        "input": processed_text,
    }

    if supports_input_type:
        params["input_type"] = input_type

    if supports_dimensions and dimensions > 0:
        params["dimensions"] = dimensions

    response = client.embeddings.create(**params)

    input_tokens = 0
    usage = getattr(response, "usage", None)
    if usage is not None:
        for attr in ("input_tokens", "prompt_tokens", "total_tokens"):
            value = getattr(usage, attr, None)
            if isinstance(value, int):
                input_tokens = value
                break

    return {
        "embedding": response.data[0].embedding,
        "input_tokens": input_tokens,
    }
