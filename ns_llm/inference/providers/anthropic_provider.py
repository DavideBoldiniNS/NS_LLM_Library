import warnings
from typing import Generator

from anthropic import Anthropic


def call_anthropic(
    model: str,
    max_output_tokens: int,
    temperature: float,
    system_prompt: str,
    user_prompt: str,
    reasoning: bool,
    api_key: str,
    stream: bool = False,
) -> dict | Generator[dict, None, None]:
    """Invoke an Anthropic Messages model and normalize the response.

    The ``thinking`` parameter is set **explicitly** on every call so the
    caller intent is never ambiguous:

    * ``reasoning=True`` on a non-opus model -> ``thinking={"type":"enabled",
      "budget_tokens":...}``.
    * ``reasoning=True`` on an opus model -> ``thinking={"type":"adaptive"}``
      and a warning is emitted (opus ignores ``temperature``).
    * ``reasoning=False`` -> ``thinking={"type":"disabled"}`` and the user
      supplied ``temperature`` is forwarded.

    When ``stream=True`` the function returns a generator yielding
    ``{"text": str, "finish_reason": str | None}`` chunks.

    Args:
        model: Anthropic model identifier (e.g. ``claude-haiku-4-5``).
        max_output_tokens: Upper bound for the generated tokens.
        temperature: Sampling temperature. Forwarded only when ``reasoning``
            is ``False`` and the model is not a Claude ``opus`` variant.
        system_prompt: System instruction passed to Anthropic as a top-level
            parameter (not inside ``messages``).
        user_prompt: User message.
        reasoning: When ``True`` Anthropic extended thinking is enabled.
        api_key: Anthropic API key.
        stream: When ``True`` enable streaming response.

    Returns:
        A dict with ``text``, ``input_tokens`` and ``output_tokens`` when
        ``stream=False``, or a generator of ``{"text", "finish_reason"}``
        chunks when ``stream=True``.

    Raises:
        anthropic.AnthropicError: any error propagated from the Anthropic SDK.
    """
    client = Anthropic(api_key=api_key)

    is_opus = "opus" in model

    kwargs = {
        "model": model,
        "max_tokens": max_output_tokens,
        "system": system_prompt,
        "messages": [
            {"role": "user", "content": user_prompt},
        ],
    }

    if reasoning:
        kwargs.pop("max_tokens", None)
        if is_opus:
            kwargs["thinking"] = {"type": "adaptive"}
            warnings.warn(
                "Modello opus con reasoning=True: la 'temperature' viene "
                "ignorata e viene attivato il thinking adattivo Anthropic.",
                stacklevel=2,
            )
        else:
            kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": max(1024, max_output_tokens // 2),
            }
    else:
        kwargs["temperature"] = temperature
        kwargs["thinking"] = {"type": "disabled"}

    if stream:
        return _stream_anthropic(client, kwargs)

    response = client.messages.create(**kwargs)

    text = ""
    for block in response.content:
        if getattr(block, "type", None) == "text":
            text += block.text

    if not text:
        warnings.warn(
            "Anthropic ha restituito una risposta senza blocchi di testo "
            "(possibile budget di thinking esaurito).",
            stacklevel=2,
        )

    return {
        "text": text,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }


def _stream_anthropic(client, kwargs) -> Generator[dict, None, None]:
    with client.messages.stream(**kwargs) as stream_response:
        for text in stream_response.text_stream:
            yield {"text": text, "finish_reason": None}
        yield {"text": "", "finish_reason": "stop"}
