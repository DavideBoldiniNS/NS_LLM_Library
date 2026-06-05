from ns_llm import generate_response

sys_Prompt = "Sei un semplice assistente AI."
u_Prompt = "Ciao! Come stai?"

#   openai_provider.py
print("OpenAI:\n"+generate_response(
    provider="openai",
    model="gpt-5.4-nano",
    max_output_tokens=100,
    temperature=0.7,
    system_prompt=sys_Prompt,
    user_prompt=u_Prompt,
    reasoning=False,
    api_key=openai_api_key
    )["text"]
)

#   anthropic_provider.py
print("Anthropic:\n"+generate_response(
    provider="anthropic",
    model="claude-haiku-4-5",
    max_output_tokens=100,
    temperature=0.7,
    system_prompt=sys_Prompt,
    user_prompt=u_Prompt,
    reasoning=False,
    api_key=anthropic_api_key
    )["text"]
)

#   Ollama_provider.py
print("Ollama:\n"+generate_response(
    provider="ollama",
    model="gpt-oss:20b-cloud",
    max_output_tokens=100,
    temperature=0.7,
    system_prompt=sys_Prompt,
    user_prompt=u_Prompt,
    reasoning=False,
    api_key=ollama_api_key
    )["text"]
)

#   openrouter_provider.py
print("OpenRouter:\n"+generate_response(
    provider="openrouter",
    model="openrouter/owl-alpha",
    max_output_tokens=100,
    temperature=0.7,
    system_prompt=sys_Prompt,
    user_prompt=u_Prompt,
    reasoning=False,
    api_key=openrouter_api_key
    )["text"]
)

#   together_provider.py
print("Together:\n"+generate_response(
    provider="together",
    model="openai/gpt-oss-20b",
    max_output_tokens=100,
    temperature=0.7,
    system_prompt=sys_Prompt,
    user_prompt=u_Prompt,
    reasoning=False,
    api_key=together_api_key
    )["text"]
)
