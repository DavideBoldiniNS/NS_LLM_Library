from together import Together


def call_together(
    model: str,
    max_output_tokens: int,
    temperature: float,
    system_prompt: str,
    user_prompt: str,
    reasoning: bool,
    api_key: str,
) -> dict:
    """Invoke a Together AI Chat Completions model and normalize the response.

    The ``reasoning`` body field is set **explicitly** on every call: enabled
    payloads when ``reasoning=True`` and disabled payloads when ``reasoning=
    False``. We never omit the field, because the Together SDK ships no
    opinionated default and silent omissions have produced inconsistent
    behavior in the past.

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

    Returns:
        A dict with ``text``, ``input_tokens`` and ``output_tokens``.

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

    response = client.chat.completions.create(**extra_args)

    return {
        "text": response.choices[0].message.content,
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
    }
