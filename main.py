"""
Application entry point.

Registers all routers. Contains no business logic — only service wiring.
Every request to the semantic cache enters the system through this file.
"""

from fastapi import FastAPI
from api.ask import router as ask_router
from observability import metrics

app = FastAPI(
    title="Semantic Cache Basics",
    description=(
        "Layered semantic caching for LLMs using FastAPI, Redis, and Ollama. "
        "Implements exact-match -> semantic similarity -> LLM fallback pipeline."
    ),
    version="1.0.0",
)

# Register the /ask endpoint router (matches blog: app.include_router(ask_router))
app.include_router(ask_router)


# ---------------------------------------------------------------------------
# Utility endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/internal/metrics")
def get_metrics():
    """
    Basic in-memory cache metrics (hits, misses, LLM calls).
    Resets on server restart. For development and debugging only.
    """
    return metrics.snapshot()


@app.delete("/internal/cache")
def flush_cache():
    """Flush all semantic cache entries from Redis."""
    from cache.semantic_cache import SemanticCache
    removed = SemanticCache().flush()
    metrics.reset()
    return {"message": f"Removed {removed} cache entries. Metrics reset."}


@app.get("/internal/cache/stats")
def cache_stats():
    """Return current number of live cache entries."""
    from cache.semantic_cache import SemanticCache
    count = SemanticCache().entry_count()
    return {"total_entries": count}