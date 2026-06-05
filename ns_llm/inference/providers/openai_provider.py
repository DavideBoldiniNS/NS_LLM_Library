import warnings

import openai


def call_openai(
    model: str,
    max_output_tokens: int,
    temperature: float,
    system_prompt: str,
    user_prompt: str,
    reasoning: bool,
    api_key: str,
) -> dict:
    """Invoke an OpenAI Chat Completions model and normalize the response.

    The ``reasoning_effort`` parameter is set **explicitly** on every call:
    ``"high"`` (the maximum allowed value in the Chat Completions API) when
    ``reasoning=True`` and ``None`` when ``reasoning=False``. We never rely on
    SDK defaults because OpenAI may change them across SDK releases.

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

    Returns:
        A dict with ``text``, ``input_tokens`` and ``output_tokens``.

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
