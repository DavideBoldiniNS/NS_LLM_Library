# NS_LLM_Library

Libreria Python multi-provider per l'invocazione di modelli LLM. Progetto didattico per studenti ITIS che già conoscono Java e si avvicinano a Python.

## Panoramica

La libreria astrae cinque provider cloud (OpenAI, Anthropic, Together AI, Ollama Cloud, OpenRouter) dietro un'interfaccia uniforme. Il modulo `ns_llm/inference/client.py` funge da router: riceve i parametri della chiamata e li smista al provider corretto. Ogni provider è implementato in un file dedicato nella cartella `ns_llm/inference/providers/`.

## Struttura del Progetto

```
ns_llm/
├── __init__.py
└── inference/
    ├── __init__.py
    ├── client.py                      # router (generate_response)
    └── providers/
        ├── openai_provider.py         # call_openai()
        ├── anthropic_provider.py      # call_anthropic()
        ├── together_provider.py       # call_together()
        ├── ollama_provider.py         # call_ollama()
        └── openrouter_provider.py     # call_openrouter()
```

## Provider Supportati

| Provider | Funzione | Libreria | Modello consigliato per il testing |
|---|---|---|---|
| OpenAI | `call_openai()` | `openai` | `gpt-4.1-nano` |
| Anthropic | `call_anthropic()` | `anthropic` | `claude-haiku-4-5` |
| Together AI | `call_together()` | `together` | `openai/gpt-oss-20b` |
| Ollama | `call_ollama()` | `ollama` | `gpt-oss:20b-cloud` |
| OpenRouter | `call_openrouter()` | `openrouter` | modello gratuito (suffisso `:free`) |

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

## Note Implementative

- **Anthropic**: il `system_prompt` non fa parte del campo `messages` ma viene passato come parametro separato. La struttura della risposta è diversa dagli altri provider (lista di blocchi tipizzati). Quando `reasoning=True` su un modello `opus` viene attivato `thinking={"type": "adaptive"}` e `temperature` viene ignorata (con `UserWarning`).
- **OpenAI**: viene usata la Chat Completions API. `max_completion_tokens` controlla il budget di output. `reasoning_effort` è sempre impostato esplicitamente (`"high"` quando `reasoning=True`, `None` quando `reasoning=False`).
- **Together AI**: payload analogo a OpenAI Chat Completions. Il campo `reasoning={"enabled": True/False}` è sempre presente, in modo da non dipendere dai default SDK.
- **Ollama**: utilizza esclusivamente Ollama Cloud (host `https://ollama.com`, autenticazione Bearer). L'istanza locale non è supportata. Il toggle di ragionamento è il parametro top-level `think=True/False` (NON dentro `options`).
- **OpenRouter**: gateway multi-provider con SDK Python ufficiale. Il client si gestisce come context manager (`with`). Il payload `reasoning` usa lo schema OpenRouter: `{"effort": "high"}` quando acceso, `{"effort": "none"}` quando spento.
- **Reasoning esplicito**: il parametro `reasoning` non è supportato da tutti i modelli; verificare la documentazione del provider prima dell'uso. La libreria lo imposta **sempre esplicitamente** (anche su `False`) per evitare di dipendere dai default SDK, che possono cambiare tra release.

## Risorse

- [Tutorial Python ufficiale](https://docs.python.org/3/tutorial/)
- [W3Schools Python](https://www.w3schools.com/python/)
- [Real Python](https://realpython.com/)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference/chat)
- [Anthropic API Reference](https://docs.anthropic.com/en/api/messages)
- [Together AI API Reference](https://docs.together.ai/reference/chat-completions)
- [Ollama Cloud API Reference](https://docs.ollama.com/cloud)
- [OpenRouter SDK Python](https://openrouter.ai/docs/sdks/python)
