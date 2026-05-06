"""
Embedding generation via Ollama.

Single responsibility: convert text into a semantic float vector.
Does NOT cache, rank, or validate embeddings — those concerns live elsewhere.
Errors are surfaced immediately (fail loudly).
"""

import httpx
from config.settings import settings


def embed_text(text: str) -> list[float]:
    """
    Convert *text* into a semantic embedding vector using Ollama.

    Raises
    ------
    RuntimeError
        If the embedding service is unreachable or returns no embedding.
    """
    url = (
        f"{settings.ollama_base_url}/api/embeddings"
    )

    try:
        resp = httpx.post(
            url,
            json={"model": settings.EMBEDDING_MODEL, "prompt": text},
            timeout=30.0,
        )
        resp.raise_for_status()
        embedding = resp.json().get("embedding", [])
        if not embedding:
            raise ValueError("Ollama returned an empty embedding.")
        return embedding
    except Exception as exc:
        raise RuntimeError(f"Failed to generate embedding: {exc}") from exc