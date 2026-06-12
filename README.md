# NS_LLM_Library

Libreria Python multi-provider per l'invocazione di modelli LLM. Progetto didattico per studenti ITIS che già conoscono Java e si avvicinano a Python.

## Panoramica

La libreria astrae cinque provider cloud (OpenAI, Anthropic, Together AI, Ollama Cloud, OpenRouter) dietro un'interfaccia uniforme. Il modulo `ns_llm/inference/client.py` funge da router per le chiamate di inferenza; `ns_llm/embedding/client.py` svolge lo stesso ruolo per gli embedding (Together AI e OpenRouter). Ogni provider è implementato in un file dedicato.

## Struttura del Progetto

```
ns_llm/
├── __init__.py
├── inference/
│   ├── __init__.py
│   ├── client.py                      # router (generate_response)
│   └── providers/
│       ├── openai_provider.py         # call_openai()
│       ├── anthropic_provider.py      # call_anthropic()
│       ├── together_provider.py       # call_together()
│       ├── ollama_provider.py         # call_ollama()
│       └── openrouter_provider.py     # call_openrouter()
├── embedding/
│   ├── __init__.py
│   ├── client.py                      # router (generate_embedding)
│   └── providers/
│       ├── together_provider.py       # call_together()
│       └── openrouter_provider.py     # call_openrouter()
├── reranking/
│   ├── __init__.py
│   ├── client.py                      # router (rerank)
│   └── providers/
│       ├── cohere_provider.py         # call_cohere_rerank()
│       ├── jina_provider.py           # call_jina_rerank()
│       ├── together_provider.py       # call_together_rerank()
│       └── openrouter_provider.py     # call_openrouter_rerank()
└── tokenizer/
    ├── __init__.py
    ├── client.py                      # router (count_tokens)
    └── providers/
        ├── openai_provider.py         # count_openai_tokens()
        ├── anthropic_provider.py      # count_anthropic_tokens()
        ├── cohere_provider.py         # count_cohere_tokens()
        ├── ollama_provider.py         # count_ollama_tokens()
        └── _estimation.py             # estimate_tokens()
```

## Provider Supportati

| Provider | Inference | Embedding | Reranking | Token counting | Libreria | Modello consigliato per il testing |
|---|---|---|---|---|---|---|
| OpenAI | `call_openai()` | — | — | tiktoken (locale) | `openai` | `gpt-4.1-nano` |
| Anthropic | `call_anthropic()` | — | — | API count_tokens | `anthropic` | `claude-haiku-4-5` |
| Together AI | `call_together()` | `call_together()` | `call_together_rerank()` | stima | `together` | `openai/gpt-oss-20b` (chat) / `together/bge-large-en-v1.5` (embed) |
| Ollama | `call_ollama()` | — | — | REST /tokenize | `ollama` | `gpt-oss:20b-cloud` |
| OpenRouter | `call_openrouter()` | `call_openrouter()` | `call_openrouter_rerank()` | stima | `openrouter` | modello gratuito (suffisso `:free`) |
| Cohere | — | — | `call_cohere_rerank()` | API tokenize | `cohere` | `rerank-v4.0-pro` |
| Jina AI | — | — | `call_jina_rerank()` | — | `requests` | `jina-reranker-v3` |

## Installazione Dipendenze

```bash
pip install openai anthropic together ollama openrouter
```

## Firma delle Funzioni

Ogni provider espone una funzione con la medesima firma contrattuale:

```python
def call_<provider>(
    model: str,           # nome del modello (es. "gpt-4.1-nano")
    max_output_tokens: int,
    temperature: float,   # 0.0 = deterministico, 2.0 = creativo
    system_prompt: str,   # istruzione di sistema / ruolo del modello
    user_prompt: str,     # messaggio dell'utente
    reasoning: bool,      # True = ragionamento esteso (chain-of-thought)
    api_key: str
) -> dict:
```

## Formato di Ritorno

Tutti i provider restituiscono un dizionario con struttura identica:

```python
{
    "text": str,          # testo generato dal modello
    "input_tokens": int,  # token del prompt inviato
    "output_tokens": int  # token della risposta ricevuta
}
```

## Utilizzo tramite router

```python
from ns_llm import generate_response

result = generate_response(
    provider="openai",
    model="gpt-4.1-nano",
    max_output_tokens=100,
    temperature=0.7,
    system_prompt="Sei un assistente.",
    user_prompt="Ciao!",
    reasoning=False,
    api_key="sk-...",
)
print(result["text"])
```

I nomi dei provider sono **case-insensitive** (`"OpenAI"`, `"openai"` e `"OPENAI"` sono equivalenti). Una validazione di base controlla che i campi non siano vuoti, che `max_output_tokens` sia un intero positivo e che `temperature` sia compreso in `[0, 2]`.

## Embedding

Anche il layer di embedding espone un router uniforme dietro `ns_llm.generate_embedding`:

```python
from ns_llm import generate_embedding

result = generate_embedding(
    provider="together",
    model="together/bge-large-en-v1.5",
    text="L'intelligenza artificiale sta cambiando il mondo.",
    input_type="search_document",
    dimensions=0,
    api_key="...",
    supports_input_type=True,   # il modello accetta 'input_type'
    supports_dimensions=False,  # il modello non supporta 'dimensions'
)
print(len(result["embedding"]), result["input_tokens"])
```

### Firma del router embedding

```python
def generate_embedding(
    provider: str,             # "together" | "openrouter" (case-insensitive)
    model: str,
    text: str,                 # non vuoto (dopo strip)
    input_type: str,           # "search_query" | "search_document" | altro
    dimensions: int,           # >= 0 (0 = non specificato)
    api_key: str,
    supports_input_type: bool = False,  # invia 'input_type' solo se True
    supports_dimensions: bool = False,  # invia 'dimensions' solo se True
) -> dict:
```

Il dizionario di ritorno ha forma:

```python
{
    "embedding": list[float],   # vettore embedding
    "input_tokens": int,        # 0 se l'API non restituisce usage
}
```

### Note implementative sull'embedding

- **Provider supportati**: solo `together` e `openrouter` (gli altri provider della libreria non espongono embeddings equivalenti).
- **`supports_input_type` / `supports_dimensions`**: sono flag espliciti passati dal chiamante, NON inferiti dal nome del modello. Questo evita lo sniffing fragile sul nome e tiene prevedibile il payload verso l'API. Impostare `True` solo se il modello scelto accetta davvero il parametro; in caso contrario l'API rifiuterà la chiamata.
- **Together e prefissi `query:` / `passage:`**: quando `input_type="search_query"` il provider antepone automaticamente `query: ` al testo, e analogamente `passage: ` per `search_document`. Questa convenzione è richiesta da alcuni modelli (es. BGE, Moka). OpenRouter **non** applica questi prefissi.
- **Validazione di base**: `text` non vuoto, `api_key` non vuoto, `dimensions >= 0`. Provider sconosciuto → `ValueError` con elenco dei validi.
- **Eccezioni SDK**: non vengono mai normalizzate; le eccezioni native di `together` o `openrouter` vengono propagate al chiamante.

## Reranking

Il modulo `ns_llm/reranking` riordina una lista di documenti per rilevanza rispetto a una query, restituendo score numerici.

```python
from ns_llm import rerank

documents = [
    {"id": "1", "text": "Parigi e la capitale della Francia."},
    {"id": "2", "text": "Berlino e la capitale della Germania."},
    {"id": "3", "text": "Roma e la capitale dell'Italia."},
]

results = rerank(
    provider="cohere",
    model="rerank-v4.0-pro",
    query="Qual e la capitale della Germania?",
    documents=documents,
    top_n=2,
    api_key="...",
)

for r in results:
    print(f"{r['score']:.3f} - {r['text']}")
```

### Firma del router reranking

```python
def rerank(
    provider: str,            # "cohere" | "together" | "jina" | "openrouter"
    model: str,
    query: str,
    documents: list[dict],    # [{"id": str, "text": str}, ...]
    top_n: int | None = None, # None = restituisce tutti
    api_key: str,
) -> list[dict]:              # [{"id", "text", "score", "original_index"}, ...]
```

## Note Implementative

- **Anthropic**: il `system_prompt` non fa parte del campo `messages` ma viene passato come parametro separato. La struttura della risposta è diversa dagli altri provider (lista di blocchi tipizzati). Quando `reasoning=True` su un modello `opus` viene attivato `thinking={"type": "adaptive"}` e `temperature` viene ignorata (con `UserWarning`).
- **OpenAI**: viene usata la Chat Completions API. `max_completion_tokens` controlla il budget di output. `reasoning_effort` è sempre impostato esplicitamente (`"high"` quando `reasoning=True`, `None` quando `reasoning=False`).
- **Together AI**: payload analogo a OpenAI Chat Completions. Il campo `reasoning={"enabled": True/False}` è sempre presente, in modo da non dipendere dai default SDK.
- **Ollama**: utilizza esclusivamente Ollama Cloud (host `https://ollama.com`, autenticazione Bearer). L'istanza locale non è supportata. Il toggle di ragionamento è il parametro top-level `think=True/False` (NON dentro `options`).
- **OpenRouter**: gateway multi-provider con SDK Python ufficiale. Il client si gestisce come context manager (`with`). Il payload `reasoning` usa lo schema OpenRouter: `{"effort": "high"}` quando acceso, `{"effort": "none"}` quando spento.
- **Reasoning esplicito**: il parametro `reasoning` non è supportato da tutti i modelli; verificare la documentazione del provider prima dell'uso. La libreria lo imposta **sempre esplicitamente** (anche su `False`) per evitare di dipendere dai default SDK, che possono cambiare tra release.

## Token Counting

Il modulo `ns_llm/tokenizer` conta i token di un testo prima di inviare richieste ai provider. Utile per stimare costi e verificare limiti di contesto.

```python
from ns_llm import count_tokens

# OpenAI - locale, velocissimo
n = count_tokens(provider="openai", model="gpt-4.1-nano", text="Ciao!")
print(n)

# Anthropic - API call
n = count_tokens(
    provider="anthropic",
    model="claude-haiku-4-5",
    text="Ciao!",
    api_key="sk-ant-...",
)

# Together - stima approssimata
n = count_tokens(
    provider="together",
    model="meta-llama/Llama-3-70b",
    text="Ciao!",
)
```

### Provider supportati per token counting

| Provider | Metodo | Accuratezza | Richiede API key |
|----------|--------|-------------|------------------|
| OpenAI | tiktoken (locale) | Esatta | No |
| Anthropic | API count_tokens | Esatta | Si |
| Cohere | API tokenize | Esatta | Si |
| Ollama | REST /api/tokenize | Esatta | No (locale) |
| Together | Stima (1 token ~ 4 chars) | Approssimata | No |
| OpenRouter | Stima (1 token ~ 4 chars) | Approssimata | No |

## Risorse

- [Tutorial Python ufficiale](https://docs.python.org/3/tutorial/)
- [W3Schools Python](https://www.w3schools.com/python/)
- [Real Python](https://realpython.com/)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference/chat)
- [Anthropic API Reference](https://docs.anthropic.com/en/api/messages)
- [Together AI API Reference](https://docs.together.ai/reference/chat-completions)
- [Ollama Cloud API Reference](https://docs.ollama.com/cloud)
- [OpenRouter SDK Python](https://openrouter.ai/docs/sdks/python)
