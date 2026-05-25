from ollama import Client


def call_ollama(
        model: str,
        max_output_tokens: int,
        temperature: float,
        system_prompt: str,
        user_prompt: str,
        reasoning: bool,
        api_key: str
) -> dict:
    
    client = Client(
        host="https://ollama.com",
        headers={'Authorization': f'Bearer {api_key}'}
    )

    messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': user_prompt},
    ]

    options = {
        'temperature': temperature,
        'num_predict': max_output_tokens
    }

    if reasoning:
        options['reasoning'] = True

    response = client.chat(
        model=model,
        messages=messages,
        options=options
    )

    return {
        "text": response['message']['content'],
        "input_tokens": response['prompt_eval_count'],
        "output_tokens": response['eval_count']
    }
