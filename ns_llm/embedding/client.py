# from ... import call_together
# from ... import call_openrouter

def generate_embedding(
    model: str,
    text: str,
    input_type: str,
    dimensions: int,
    provider: str,
    api_key: str
) -> dict:
    
    if provider == "together":
        # return call_together(...)
        pass
    elif provider == "openrouter":
        # return call_openrouter(...)
        pass
    else: raise ValueError(f"Provider '{provider}' non riconosciuto. "
            "I valori validi sono: 'together', 'openrouter'.")

