from typing import Generator

from openrouter import OpenRouter


def call_openrouter(
    model: str,
    max_output_tokens: int,
    temperature: float,
    system_prompt: str,
    user_prompt: str,
    reasoning: bool,
    api_key: str,
    stream: bool = False,
) -> dict | Generator[dict, None, None]:
    """Invoke an OpenRouter chat model and normalize the response.

    The OpenRouter SDK exposes itself as a context manager that owns the
    underlying HTTP client; this provider follows the same pattern.

    The ``reasoning`` body field is set **explicitly** on every call to the
    OpenRouter ``Reasoning`` schema:

    * ``reasoning=True``  -> ``reasoning={"effort": "high"}``
    * ``reasoning=False`` -> ``reasoning={"effort": "none"}``

    We never rely on SDK defaults because the schema is enforced server-side
    and silent omissions have produced inconsistent behavior across upstream
    providers.

    When ``stream=True`` the function returns a generator yielding
    ``{"text": str, "finish_reason": str | None}`` chunks.

    Args:
        model: OpenRouter model identifier (e.g. ``anthropic/claude-4.5-sonnet``).
        max_output_tokens: Upper bound for the generated tokens; passed as
            ``max_tokens`` to the OpenRouter API.
        temperature: Sampling temperature forwarded as-is to the SDK.
        system_prompt: System message prepended to the conversation.
        user_prompt: User message.
        reasoning: When ``True`` the request carries ``reasoning={"effort":
            "high"}``; when ``False`` it carries ``reasoning={"effort":
            "none"}`` to make the intent explicit.
        api_key: OpenRouter API key.
        stream: When ``True`` enable streaming response.

    Returns:
        A dict with ``text``, ``input_tokens`` and ``output_tokens`` when
        ``stream=False``, or a generator of ``{"text", "finish_reason"}``
        chunks when ``stream=True``.

    Raises:
        openrouter.OpenRouterError: any error propagated from the SDK.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    extra_args = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_output_tokens,
        "reasoning": {"effort": "high" if reasoning else "none"},
    }

    if stream:
        return _stream_openrouter(api_key, extra_args)

    with OpenRouter(api_key=api_key) as client:
        response = client.chat.send(**extra_args)

        return {
            "text": response.choices[0].message.content,
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
        }


def _stream_openrouter(api_key, extra_args) -> Generator[dict, None, None]:
    with OpenRouter(api_key=api_key) as client:
        response = client.chat.send(**extra_args, stream=True)
        for chunk in response:
            if chunk.choices:
                delta = chunk.choices[0].delta
                finish_reason = chunk.choices[0].finish_reason
                yield {
                    "text": delta.content or "",
                    "finish_reason": finish_reason,
                }
