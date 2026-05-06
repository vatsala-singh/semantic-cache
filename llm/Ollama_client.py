"""
Ollama LLM client.

Wraps calls to Ollama's text-generation endpoint.
Keeping LLM interaction isolated allows the rest of the system to remain
model-agnostic.
"""

import httpx
from config.settings import settings


def generate_llm_response(prompt: str) -> str:
    """
    Send *prompt* to the configured Ollama LLM and return the response text.

    On any network or HTTP error, returns a string beginning with
    ``[LLM Error]`` so the caller can safely detect and discard it.
    """
    url = (
        f"{settings.ollama_base_url}/api/generate"
    )

    try:
        resp = httpx.post(
            url,
            json={
                "model": settings.llm_model,
                "prompt": prompt,
                "stream": False,
            },
            timeout=120.0,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except httpx.HTTPStatusError as exc:
        return f"[LLM Error] HTTP {exc.response.status_code}: {exc.response.text}"
    except httpx.RequestError as exc:
        return f"[LLM Error] Request failed: {exc}"
    except Exception as exc:
        return f"[LLM Error] Unexpected error: {exc}"