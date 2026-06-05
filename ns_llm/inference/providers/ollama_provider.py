from ollama import Client


def call_ollama(
    model: str,
    max_output_tokens: int,
    temperature: float,
    system_prompt: str,
    user_prompt: str,
    reasoning: bool,
    api_key: str,
) -> dict:
    """Invoke an Ollama Cloud model and normalize the response.

    The Ollama Python client is configured for the managed Ollama Cloud
    service (``https://ollama.com``) and authenticates with a Bearer token.
    Local Ollama instances are intentionally out of scope for this provider.

    The ``think`` parameter is exposed by the Ollama SDK as a top-level
    keyword on ``Client.chat``: that is where the cloud API expects the
    reasoning toggle, not inside ``options``. We set it **explicitly** on
    every call (``True`` or ``False``) so the caller intent is never
    ambiguous and we never rely on the SDK default.

    Args:
        model: Ollama model identifier (e.g. ``gpt-oss:20b-cloud``).
        max_output_tokens: Upper bound for the generated tokens; passed as
            ``num_predict`` in the Ollama ``options`` payload.
        temperature: Sampling temperature forwarded as-is to the SDK.
        system_prompt: System message prepended to the conversation.
        user_prompt: User message.
        reasoning: When ``True`` the request carries ``think=True``; when
            ``False`` the request carries ``think=False``.
        api_key: Ollama Cloud API key used as Bearer token.

    Returns:
        A dict with ``text``, ``input_tokens`` and ``output_tokens``.

    Raises:
        ollama.ResponseError: any error propagated from the Ollama SDK.
    """
    client = Client(
        host="https://ollama.com",
        headers={"Authorization": f"Bearer {api_key}"},
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    options = {
        "temperature": temperature,
        "num_predict": max_output_tokens,
    }

    response = client.chat(
        model=model,
        messages=messages,
        think=bool(reasoning),
        options=options,
    )

    return {
        "text": response["message"]["content"],
        "input_tokens": response["prompt_eval_count"],
        "output_tokens": response["eval_count"],
    }
