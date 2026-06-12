import json
import re
from typing import Any


def parse_json_response(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if not stripped:
        raise ValueError("Impossibile parsare la risposta come JSON: testo vuoto.")
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    match = re.search(r"```json\s*\n(.*?)\n```", stripped, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass
    preview = stripped[:200].replace("\n", " ")
    raise ValueError(f"Impossibile parsare la risposta come JSON: '{preview}...'")
