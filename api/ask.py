"""
/ask endpoint — orchestration layer for the semantic caching pipeline.

Control flow (matches blog Figure 4):
  1. Input validation  (fail fast before any expensive work)
  2. Exact-match cache lookup  → return immediately on hit
  3. Embedding generation      (escalation; skipped on exact hit)
  4. Semantic cache lookup     → return if similarity ≥ threshold & confidence ≥ gate
  5. bypass_cache flag handling
  6. LLM fallback + cache population

The endpoint does NOT implement caching or similarity logic directly.
Those concerns live in app/cache/semantic_cache.py.
"""

import time

from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel, field_validator

from cache.semantic_cache import SemanticCache
from cache.poisoning import is_safe_to_cache, is_poisoned
from cache.ttl import is_expired
from llm.embedder import embed_text
from llm.Ollama_client import generate_llm_response
from observability import metrics
from config.settings import settings

router = APIRouter()

# One shared cache instance — stateless endpoint, all persistence in Redis
cache = SemanticCache()


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class AskRequest(BaseModel):
    query: str
    bypass_cache: bool = False

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        """
        Reject empty or whitespace-only queries before they enter the pipeline.
        This prevents cache pollution and unnecessary embedding / LLM calls.
        """
        if not v or not v.strip():
            raise ValueError("Query cannot be empty or whitespace-only")
        stripped = v.strip()
        if len(stripped) < 3:
            raise ValueError("Query is too short (minimum 3 characters)")
        return stripped


class AskResponse(BaseModel):
    response: str
    from_cache: bool
    similarity: float
    debug: dict


# ---------------------------------------------------------------------------
# /ask endpoint
# ---------------------------------------------------------------------------


@router.post("/ask", response_model=AskResponse)
def ask_endpoint(request: AskRequest) -> AskResponse:
    """
    Semantic caching pipeline.

    Diagnostic fields (``from_cache``, ``similarity``, ``debug``) make
    cache behavior transparent rather than opaque during development.
    """
    query = request.query
    miss_reason: str | None = None
    embedding: list[float] | None = None

    # ── Step 1: Exact-match cache lookup (fast path) ───────────────────
    if not request.bypass_cache:
        cached = cache.search(None, exact_query=query)

        if cached is not None:
            # Validate freshness and safety
            if is_expired(cached):
                miss_reason = "expired"
            elif is_poisoned(cached):
                miss_reason = "poisoned"
            elif float(cached.get("confidence", 0.0)) < settings.CONFIDENCE_THRESHOLD:
                miss_reason = "low_confidence"
            else:
                metrics.cache_hit()
                return AskResponse(
                    response=cached["response"],
                    from_cache=True,
                    similarity=float(cached.get("similarity", 1.0)),
                    debug={
                        "hit": True,
                        "cache_path": cached.get("match_type"),
                        "deduplicated": cached.get("deduplicated", True),
                        "query_hash": cached.get("query_hash"),
                        "similarity": float(cached.get("similarity", 1.0)),
                        "confidence": float(cached.get("confidence", 0.0)),
                        "explanation": cached.get("explanation", ""),
                    },
                )

    # ── Step 2: Embedding generation (escalation) ──────────────────────
    # Only reached when exact match failed or was bypassed.
    # Intentionally deferred to keep cheap checks first.
    if not request.bypass_cache:
        try:
            embedding = embed_text(query)
        except RuntimeError as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Embedding service unavailable: {exc}",
            )

        # ── Step 3: Semantic cache lookup ──────────────────────────────
        cached = cache.search(embedding)

        if cached is not None:
            conf = float(cached.get("confidence", 0.0))
            if is_expired(cached):
                miss_reason = "expired"
            elif is_poisoned(cached):
                miss_reason = "poisoned"
            elif conf < settings.CONFIDENCE_THRESHOLD:
                miss_reason = "low_confidence"
            else:
                metrics.cache_hit()
                return AskResponse(
                    response=cached["response"],
                    from_cache=True,
                    similarity=float(cached.get("similarity", 0.0)),
                    debug={
                        "hit": True,
                        "cache_path": cached.get("match_type"),
                        "deduplicated": cached.get("deduplicated", False),
                        "similarity": float(cached.get("similarity", 0.0)),
                        "confidence": conf,
                        "explanation": cached.get("explanation", ""),
                    },
                )
        else:
            miss_reason = miss_reason or "no_match"

    # ── Step 4: Explicit bypass ────────────────────────────────────────
    if request.bypass_cache:
        miss_reason = "bypass"
        metrics.bypass()

    # ── Step 5: LLM fallback ───────────────────────────────────────────
    metrics.cache_miss()
    response_text = generate_llm_response(query)
    metrics.llm_call()

    # Populate cache — never store error responses (cache poisoning guard)
    if not request.bypass_cache and is_safe_to_cache(response_text):
        # Generate embedding if we haven't already (bypass path)
        if embedding is None:
            try:
                embedding = embed_text(query)
            except RuntimeError:
                embedding = []  # store without embedding; exact-match will still work

        metadata = {
            "pipeline": "semantic_cache_v1",
            "model": settings.llm_model,
            "embedding_model": settings.EMBEDDING_MODEL,
        }
        cache.store(
            query=query,
            embedding=embedding,
            response=response_text,
            metadata=metadata,
        )

    return AskResponse(
        response=response_text,
        from_cache=False,
        similarity=0,
        debug={
            "hit": False,
            "match_type": None,
            "similarity": 0,
            "confidence": 0,
            "miss_reason": miss_reason,
        },
    )