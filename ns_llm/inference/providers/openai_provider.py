import warnings
from typing import Generator

import openai


def call_openai(
    model: str,
    max_output_tokens: int,
    temperature: float,
    system_prompt: str,
    user_prompt: str,
    reasoning: bool,
    api_key: str,
    stream: bool = False,
) -> dict | Generator[dict, None, None]:
    """Invoke an OpenAI Chat Completions model and normalize the response.

    The ``reasoning_effort`` parameter is set **explicitly** on every call:
    ``"high"`` (the maximum allowed value in the Chat Completions API) when
    ``reasoning=True`` and ``None`` when ``reasoning=False``. We never rely on
    SDK defaults because OpenAI may change them across SDK releases.

    When ``stream=True`` the function returns a generator yielding
    ``{"text": str, "finish_reason": str | None}`` chunks.

    Args:
        model: OpenAI model identifier (e.g. ``gpt-4.1-nano``).
        max_output_tokens: Upper bound for the generated tokens.
        temperature: Sampling temperature forwarded as-is to the SDK.
        system_prompt: System message prepended to the conversation.
        user_prompt: User message.
        reasoning: When ``True`` the request carries ``reasoning_effort="high"``;
            when ``False`` the request carries ``reasoning_effort=None`` to
            make the intent explicit.
        api_key: OpenAI API key.
        stream: When ``True`` enable streaming response.

    Returns:
        A dict with ``text``, ``input_tokens`` and ``output_tokens`` when
        ``stream=False``, or a generator of ``{"text", "finish_reason"}``
        chunks when ``stream=True``.

    Raises:
        openai.OpenAIError: any error propagated from the OpenAI SDK.
    """
    client = openai.OpenAI(api_key=api_key)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    request = {
        "model": model,
        "messages": messages,
        "max_completion_tokens": max_output_tokens,
        "temperature": temperature,
        "reasoning_effort": "high" if reasoning else None,
    }

    if stream:
        response = client.chat.completions.create(**request, stream=True)
        return _stream_openai(response)

    response = client.chat.completions.create(**request)

    text = response.choices[0].message.content
    if text is None:
        text = ""
        warnings.warn(
            "OpenAI ha restituito un messaggio senza contenuto testuale "
            "(possibile modello in modalita' solo ragionamento).",
            stacklevel=2,
        )

    return {
        "text": text,
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
    }


def _stream_openai(response) -> Generator[dict, None, None]:
    for chunk in response:
        if chunk.choices:
            delta = chunk.choices[0].delta
            finish_reason = chunk.choices[0].finish_reason
            yield {
                "text": delta.content or "",
                "finish_reason": finish_reason,
            }
