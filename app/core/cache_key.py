import hashlib
import json
from typing import Any


def build_cache_key(prefix: str, **kwargs: Any) -> str:
    """Gera chave determinística para cache a partir de parâmetros serializáveis."""
    payload = json.dumps(kwargs, sort_keys=True, default=str)
    digest = hashlib.sha256(payload.encode()).hexdigest()
    return f"{prefix}:{digest}"
