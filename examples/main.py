"""Example usage of the ns_llm multi-provider inference layer.

Keys are read from environment variables; export them before running:

    export OPENAI_API_KEY=...
    export ANTHROPIC_API_KEY=...
    export OLLAMA_API_KEY=...
    export OPENROUTER_API_KEY=...
    export TOGETHER_API_KEY=...
    python examples/main.py
"""

import os

from ns_llm import generate_response

system_prompt = "Sei un semplice assistente AI."
user_prompt = "Ciao! Come stai?"

# openai_provider.py
print(
    "OpenAI:\n"
    + generate_response(
        provider="openai",
        model="gpt-4.1-nano",
        max_output_tokens=100,
        temperature=0.7,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        reasoning=False,
        api_key=os.environ["OPENAI_API_KEY"],
    )["text"]
)

# anthropic_provider.py
print(
    "Anthropic:\n"
    + generate_response(
        provider="anthropic",
        model="claude-haiku-4-5",
        max_output_tokens=100,
        temperature=0.7,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        reasoning=False,
        api_key=os.environ["ANTHROPIC_API_KEY"],
    )["text"]
)

# ollama_provider.py
print(
    "Ollama:\n"
    + generate_response(
        provider="ollama",
        model="gpt-oss:20b-cloud",
        max_output_tokens=100,
        temperature=0.7,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        reasoning=False,
        api_key=os.environ["OLLAMA_API_KEY"],
    )["text"]
)

# openrouter_provider.py
print(
    "OpenRouter:\n"
    + generate_response(
        provider="openrouter",
        model="openrouter/owl-alpha",
        max_output_tokens=100,
        temperature=0.7,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        reasoning=False,
        api_key=os.environ["OPENROUTER_API_KEY"],
    )["text"]
)

# together_provider.py
print(
    "Together:\n"
    + generate_response(
        provider="together",
        model="openai/gpt-oss-20b",
        max_output_tokens=100,
        temperature=0.7,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        reasoning=False,
        api_key=os.environ["TOGETHER_API_KEY"],
    )["text"]
)
