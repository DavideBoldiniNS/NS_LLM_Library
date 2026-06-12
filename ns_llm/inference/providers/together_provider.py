from typing import Generator

from together import Together


def call_together(
    model: str,
    max_output_tokens: int,
    temperature: float,
    system_prompt: str,
    user_prompt: str,
    reasoning: bool,
    api_key: str,
    stream: bool = False,
) -> dict | Generator[dict, None, None]:
    """Invoke a Together AI Chat Completions model and normalize the response.

    The ``reasoning`` body field is set **explicitly** on every call: enabled
    payloads when ``reasoning=True`` and disabled payloads when ``reasoning=
    False``. We never omit the field, because the Together SDK ships no
    opinionated default and silent omissions have produced inconsistent
    behavior in the past.

    When ``stream=True`` the function returns a generator yielding
    ``{"text": str, "finish_reason": str | None}`` chunks.

    Args:
        model: Together model identifier (e.g. ``openai/gpt-oss-20b``).
        max_output_tokens: Upper bound for the generated tokens; passed as
            ``max_tokens`` to the Together API.
        temperature: Sampling temperature forwarded as-is to the SDK.
        system_prompt: System message prepended to the conversation.
        user_prompt: User message.
        reasoning: When ``True`` Together receives ``reasoning={"enabled":
            True}``; when ``False`` it receives ``reasoning={"enabled":
            False}`` to make the intent explicit.
        api_key: Together API key.
        stream: When ``True`` enable streaming response.

    Returns:
        A dict with ``text``, ``input_tokens`` and ``output_tokens`` when
        ``stream=False``, or a generator of ``{"text", "finish_reason"}``
        chunks when ``stream=True``.

    Raises:
        together.TogetherError: any error propagated from the Together SDK.
    """
    client = Together(api_key=api_key)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    extra_args = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_output_tokens,
        "reasoning": {"enabled": bool(reasoning)},
    }

    if stream:
        response = client.chat.completions.create(**extra_args, stream=True)
        return _stream_together(response)

    response = client.chat.completions.create(**extra_args)

    return {
        "text": response.choices[0].message.content,
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
    }


def _stream_together(response) -> Generator[dict, None, None]:
    for chunk in response:
        if chunk.choices:
            delta = chunk.choices[0].delta
            finish_reason = chunk.choices[0].finish_reason
            yield {
                "text": delta.content or "",
                "finish_reason": finish_reason,
            }
