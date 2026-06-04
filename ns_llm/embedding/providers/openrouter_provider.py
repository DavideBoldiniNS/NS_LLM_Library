from openrouter import OpenRouter

def call_openrouter(
    model: str,
    text: str,
    input_type: str,
    dimensions: int,
    api_key: str
) -> dict:
    with OpenRouter(api_key=api_key) as client:
        kwargs={
            "model": model,
            "input": text,
        }
        if dimensions>0:
            kwargs["dimensions"]=dimensions
        if "supporta-input-type" in model:
            kwargs["input_type"]=input_type
        response=client.embeddings.create(**kwargs)
    embedding=response.data[0].embedding
    input_tokens=0
    if hasattr(response, "usage") and hasattr(response.usage, "input_tokens"):
        input_tokens=response.usage.input_tokens
    return{
        "embedding": embedding,
        "input_tokens": input_tokens
    }