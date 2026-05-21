from together import Together

def call_together(
    model: str,
    max_output_tokens: int,
    temperature: float,
    system_prompt: str,
    user_prompt: str,
    reasoning: bool,
    api_key: str
) -> dict:
    client=Together(api_key=api_key)
    messages=[
        {"role":"system","content":system_prompt},
        {"role":"user","content":user_prompt},
    ]
    extra_args={
        "model":model,
        "messages":messages,
        "temperature":temperature,
    }
    if reasoning:
        extra_args["reasoning"]= {"enabled":True}
        extra_args["max_output_tokens"]= max_output_tokens
    else:
        extra_args["max_tokens"]= max_output_tokens
    response=client.chat.completions.create(**extra_args)
    text=response.choices[0].message.content
    input_tokens=response.usage.prompt_tokens
    output_tokens=response.usage.completion_tokens
    return {
        "text":text,
        "input_tokens":input_tokens,
        "output_tokens":output_tokens,
    }