from together import Together

def call_together(    
    model: str,
    text: str,
    input_type: str,
    dimensions: int,
    api_key: str
) -> dict:
    client=Together(api_key=api_key)
    params={
        "model": model,
        "input": text,
    }
    if "supporta-input-type" in model:
        params["input_type"]=input_type
    response=client.embeddings.create(**params)
    return{
        "embedding":response.data[0].embedding,
        "input_tokens":0,
    }