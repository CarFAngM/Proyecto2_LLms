"""
Generación de embeddings con OpenAI en modo funcional.
"""

import os
from typing import List
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


def _get_client(api_key: str = None) -> OpenAI:
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError("Se requiere OPENAI_API_KEY")
    return OpenAI(api_key=key)


def get_embedding_dimension(model: str = DEFAULT_EMBEDDING_MODEL) -> int:
    return 1536 if "3-small" in model else 3072


def _truncate_text(text: str, max_tokens: int = 8000) -> str:
    limit = max_tokens * 4  # ~4 chars/token
    return text[:limit] if len(text) > limit else text


def generate_embedding(text: str, model: str = DEFAULT_EMBEDDING_MODEL, api_key: str = None) -> List[float]:
    try:
        client = _get_client(api_key)
        text = _truncate_text(text)
        response = client.embeddings.create(model=model, input=text)
        return response.data[0].embedding
    except Exception as exc:  # pragma: no cover - depende del API
        raise Exception(f"Error generando embedding: {exc}")


def generate_embeddings_batch(texts: List[str], model: str = DEFAULT_EMBEDDING_MODEL, api_key: str = None) -> List[List[float]]:
    try:
        client = _get_client(api_key)
        batch_size = 100
        all_embeddings: List[List[float]] = []

        for i in range(0, len(texts), batch_size):
            batch = [_truncate_text(text) for text in texts[i:i + batch_size]]
            response = client.embeddings.create(model=model, input=batch)
            all_embeddings.extend([item.embedding for item in response.data])

        return all_embeddings
    except Exception as exc:  # pragma: no cover
        raise Exception(f"Error generando embeddings en batch: {exc}")
