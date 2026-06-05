"""Example usage of the ns_llm embedding layer.

Keys are read from environment variables; export them before running:

    export TOGETHER_API_KEY=...
    export OPENROUTER_API_KEY=...
    python examples/embedding_main.py
"""

import os

from ns_llm import generate_embedding

text = "L'intelligenza artificiale sta cambiando il mondo."

# Together AI
print(
    "Together:\n"
    + str(
        generate_embedding(
            provider="together",
            model="together/bge-large-en-v1.5",
            text=text,
            input_type="search_document",
            dimensions=0,
            api_key=os.environ["TOGETHER_API_KEY"],
            supports_input_type=True,
            supports_dimensions=False,
        )
    )
)

# OpenRouter
print(
    "OpenRouter:\n"
    + str(
        generate_embedding(
            provider="openrouter",
            model="openai/text-embedding-3-small",
            text=text,
            input_type="search_query",
            dimensions=0,
            api_key=os.environ["OPENROUTER_API_KEY"],
            supports_input_type=False,
            supports_dimensions=False,
        )
    )
)
