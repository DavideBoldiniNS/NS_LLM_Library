# NS_LLM_Library

Libreria Python multi-provider per l'invocazione di modelli LLM. Progetto didattico per studenti ITIS che già conoscono Java e si avvicinano a Python.

BLABLABLA

## Panoramica
La libreria astrae tre provider cloud (OpenAI, Anthropic, Together AI) dietro un'interfaccia uniforme. Il modulo `llm_client.py` funge da router: riceve i parametri della chiamata e li smista al provider corretto. Ogni provider è implementato in un file dedicato nella cartella `providers/`.

## Struttura del Progetto

```
project/
├── providers/
│   ├── openai_provider.py       # call_openai()
│   ├── anthropic_provider.py    # call_anthropic()
│   ├── together_provider.py     # call_together()
│   └── ...
├── llm_client.py                # router
└── main.py
```

## Provider Supportati

| Provider | Funzione | Libreria | Modello consigliato per il testing |
|---|---|---|---|
| OpenAI | `call_openai()` | `openai` | `gpt-4o-mini` |
| Anthropic | `call_anthropic()` | `anthropic` | `claude-haiku-4-5` |
| Together AI | `call_together()` | `together` | `openai/gpt-oss-20b` |

## Installazione Dipendenze

```bash
pip install openai anthropic together
```

## Firma delle Funzioni

Ogni provider espone una funzione con la medesima firma contrattuale:

```python
def call_<provider>(
    model: str,           # nome del modello (es. "gpt-4o-mini")
    max_output_tokens: int,
    temperature: float,   # 0.0 = deterministico, 1.0 = creativo
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

## Note Implementative

- **Anthropic**: il `system_prompt` non fa parte del campo `messages` ma viene passato come parametro separato. Struttura della risposta diversa rispetto agli altri provider.
- **OpenAI**: disponibili due API distinte (Chat Completions e la nuova Responses API) con strutture e nomi dei contatori di token differenti.
- **Together AI**: formato di richiesta analogo a OpenAI Chat Completions. Supporta il ragionamento esteso su modelli ibridi tramite parametro dedicato.
- Il parametro `reasoning` non è supportato da tutti i modelli: verificare la documentazione del provider prima dell'uso.
- Per provider locali (es. Ollama) il parametro `api_key` può essere ignorato.

## Risorse

- [Tutorial Python ufficiale](https://docs.python.org/3/tutorial/)
- [W3Schools Python](https://www.w3schools.com/python/)
- [Real Python](https://realpython.com/)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference/chat)
- [Anthropic API Reference](https://docs.anthropic.com/en/api/messages)
- [Together AI API Reference](https://docs.together.ai/docs/chat-overview)
