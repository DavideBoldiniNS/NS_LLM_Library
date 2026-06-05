from together import Together

def call_together(    
    model: str,
    text: str,
    input_type: str,
    dimensions: int,
    api_key: str
) -> dict:
    if not isinstance(text,str) or text.strip()=="":
        raise ValueError("Il parametro 'text' deve essere una stringa non vuota.")
    client=Together(api_key=api_key)
    if input_type=="search_query":
        processed_text= f"query: {text}"
    elif input_type=="search_document":
        processed_text=f"passage: {text}"
    else:
        processed_text=text
    params={
        "model": model,
        "input": processed_text,
    }
    if "supporta-input-type" in model:
        params["input_type"]=input_type
    if dimensions>0 and "supports-dimensions" in model:
        params["dimensions"]=dimensions
    try:
        response=client.embeddings.create(**params)
    except Exception as e:
        raise RuntimeError(f"Errore nella chiamata Together embeddings: {e}")
    if not hasattr(response, "data") or not response.data:
        raise RuntimeError("Risposta Together priva di campo 'data'.")
    if not hasattr(response.data[0], "embedding"):
        raise RuntimeError("Embedding mancante nella risposta Together.")
    input_tokens=0
    if hasattr(response,"usage") and response.usage is not None:
        if hasattr(response.usage,"input_tokens"):
            input_tokens=response.usage.input_tokens
        elif hasattr(response.usage,"prompt_tokens"):
            input_tokens=response.usage.prompt_tokens
        elif hasattr(response.usage,"total_tokens"):
            input_tokens=response.usage.total_tokens
    return{
        "embedding":response.data[0].embedding,
        "input_tokens":input_tokens
    }