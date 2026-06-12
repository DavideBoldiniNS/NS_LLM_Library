from typing import overload, Any
from pydantic import BaseModel

from .schema import convert_to_json_schema
from .parser import parse_json_response
from .providers.openai_provider import call_openai_structured
from .providers.anthropic_provider import call_anthropic_structured
from .providers.together_provider import call_together_structured
from .providers.ollama_provider import call_ollama_structured
from .providers.openrouter_provider import call_openrouter_structured


PROVIDER_HANDLERS = {
    "openai": call_openai_structured,
    "anthropic": call_anthropic_structured,
    "together": call_together_structured,
    "ollama": call_ollama_structured,
    "openrouter": call_openrouter_structured,
}


def _validate_input(
    provider: str,
    model: str,
    api_key: str,
    schema: dict | type[BaseModel],
    max_retries: int,
) -> None:
    if not isinstance(provider, str) or not provider.strip():
        raise ValueError("Il parametro 'provider' deve essere una stringa non vuota.")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("Il parametro 'model' deve essere una stringa non vuota.")
    if not isinstance(api_key, str) or not api_key.strip():
        raise ValueError("Il parametro 'api_key' deve essere una stringa non vuota.")
    if not (isinstance(schema, dict) or (isinstance(schema, type) and issubclass(schema, BaseModel))):
        raise TypeError(
            f"Il parametro 'schema' deve essere un dict o una sottoclasse di "
            f"BaseModel, non {type(schema).__name__}."
        )
    if not isinstance(max_retries, int) or max_retries < 0:
        raise ValueError("Il parametro 'max_retries' deve essere un intero >= 0.")


@overload
def generate_structured_response(
    provider: str,
    model: str,
    schema: dict,
    max_output_tokens: int,
    temperature: float,
    system_prompt: str,
    user_prompt: str,
    api_key: str,
    max_retries: int = 2,
    reasoning: bool = False,
) -> dict: ...


@overload
def generate_structured_response(
    provider: str,
    model: str,
    schema: type[BaseModel],
    max_output_tokens: int,
    temperature: float,
    system_prompt: str,
    user_prompt: str,
    api_key: str,
    max_retries: int = 2,
    reasoning: bool = False,
) -> BaseModel: ...


def generate_structured_response(
    provider: str,
    model: str,
    schema: dict | type[BaseModel],
    max_output_tokens: int,
    temperature: float,
    system_prompt: str,
    user_prompt: str,
    api_key: str,
    max_retries: int = 2,
    reasoning: bool = False,
) -> dict | BaseModel:
    _validate_input(provider, model, api_key, schema, max_retries)

    json_schema = convert_to_json_schema(schema)
    handler = PROVIDER_HANDLERS.get(provider.lower())
    if handler is None:
        valid = ", ".join(sorted(PROVIDER_HANDLERS))
        raise ValueError(
            f"Provider '{provider}' non supportato. Valori validi: {valid}."
        )

    is_pydantic = isinstance(schema, type) and issubclass(schema, BaseModel)

    last_error: str | None = None
    for attempt in range(max_retries + 1):
        text = handler(
            model=model,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            api_key=api_key,
            schema=json_schema,
            reasoning=reasoning,
        )
        try:
            parsed = parse_json_response(text)
        except ValueError as exc:
            last_error = str(exc)
            continue
        if is_pydantic:
            return schema.model_validate(parsed)
        return parsed

    raise ValueError(
        f"Impossibile ottenere un JSON valido dopo {max_retries + 1} tentativi. "
        f"Ultimo errore: {last_error}"
    )
