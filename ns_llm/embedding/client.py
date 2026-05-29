from .providers.together_provider import call_together
from .providers.openrouter_provider import call_openrouter

def generate_embedding(
    model: str,
    text: str,
    input_type: str,
    dimensions: int,
    provider: str,
    api_key: str
) -> dict:
    
    if provider == "together":
        return call_together(
            model=model,
            text=text,
            input_type=input_type,
            dimensions=dimensions,
            api_key=api_key
        )
    elif provider == "openrouter":
        return call_openrouter(
            model=model,
            text=text,
            input_type=input_type,
            dimensions=dimensions,
            api_key=api_key
        )
    else: raise ValueError(f"Provider '{provider}' non riconosciuto. "
            "I valori validi sono: 'together', 'openrouter'.")

