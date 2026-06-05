from openrouter import OpenRouter


def call_openrouter(
    model: str,
    text: str,
    input_type: str,
    dimensions: int,
    supports_input_type: bool,
    supports_dimensions: bool,
    api_key: str,
) -> dict:
    """Invoke an OpenRouter embeddings model and normalize the response.

    The OpenRouter SDK exposes itself as a context manager that owns the
    underlying HTTP client; this provider follows the same pattern.

    ``input_type`` and ``dimensions`` are forwarded only when the caller
    explicitly signals support via ``supports_input_type`` and
    ``supports_dimensions``. This keeps the wire payload predictable and
    avoids sending parameters that some upstream providers would reject.

    Args:
        model: OpenRouter model identifier.
        text: Input string to embed; must be non-empty.
        input_type: Hint forwarded only when ``supports_input_type=True``.
        dimensions: Target embedding size; forwarded only when
            ``supports_dimensions=True`` and strictly positive.
        supports_input_type: Set to ``True`` to forward ``input_type``.
        supports_dimensions: Set to ``True`` to forward ``dimensions``.
        api_key: OpenRouter API key.

    Returns:
        A dict with ``embedding`` (list of floats) and ``input_tokens``.

    Raises:
        openrouter.OpenRouterError: any error propagated from the SDK.
    """
    with OpenRouter(api_key=api_key) as client:
        kwargs = {
            "model": model,
            "input": text,
        }

        if supports_input_type:
            kwargs["input_type"] = input_type

        if supports_dimensions and dimensions > 0:
            kwargs["dimensions"] = dimensions

        response = client.embeddings.create(**kwargs)

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
