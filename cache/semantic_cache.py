"""
Semantic cache backed by Redis.

Implements the layered strategy described in the blog:
  Layer 1 – Exact match via normalised SHA-256 hash   (O(1), ~1 ms)
  Layer 2 – Semantic match via cosine similarity scan  (O(N))
  Layer 3 – LLM fallback (handled by the API layer)

All Redis interaction is confined to this class; the API layer never
talks to Redis directly.
"""

import hashlib
import json
import time
import uuid

import numpy as np
import redis

from cache.schemas import CacheEntry
from cache.ttl import default_ttl, is_expired
from cache.poisoning import is_poisoned
from config.settings import settings


def _normalise(text: str) -> str:
    """Lowercase and strip whitespace for consistent hashing."""
    return " ".join(text.strip().lower().split())


def _hash_query(text: str) -> str:
    return hashlib.sha256(_normalise(text).encode()).hexdigest()


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom == 0.0:
        return 0.0
    return float(np.dot(va, vb) / denom)


def _confidence(similarity: float, entry: dict) -> float:
    """
    Combine similarity and freshness into a single confidence score.

    Score = similarity x freshness_factor
    freshness_factor decays linearly from 1.0 (brand new) to 0.0 (at TTL).
    """
    created_at = int(entry.get("created_at", 0))
    ttl = int(entry.get("ttl", settings.DEFAULT_TTL))
    age = time.time() - created_at
    freshness = max(0.0, 1.0 - (age / ttl))
    return round(similarity * freshness, 4)


class SemanticCache:
    """Encapsulates all Redis interaction and similarity logic."""

    def __init__(self) -> None:
        self.r = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            decode_responses=True,
        )
        self.similarity_threshold = settings.SIMILARITY_THRESHOLD
        self.namespace = settings.CACHE_NAMESPACE

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _redis_key(self, entry_id: str) -> str:
        return f"{self.namespace}:{entry_id}"

    def _index_key(self) -> str:
        return f"{self.namespace}:keys"

    def _hash_query(self, text: str) -> str:
        return _hash_query(text)

    # ------------------------------------------------------------------
    # Layer 1 + 2 – Search
    # ------------------------------------------------------------------

    def search(
        self,
        embedding: "list[float] | None",
        *,
        exact_query: "str | None" = None,
    ) -> "dict | None":
        """
        Attempt to find a reusable cache entry.

        Parameters
        ----------
        embedding    : Query embedding for semantic search; None for exact-only.
        exact_query  : If provided, attempt a hash-based exact match first.

        Returns the raw Redis hash dict augmented with:
          ``similarity``, ``confidence``, ``match_type``, ``deduplicated``
        or None if no suitable entry is found.
        """
        all_keys = self.r.smembers(self._index_key())
        stale_keys = []

        # ── Layer 1: Exact match ────────────────────────────────────────
        if exact_query is not None:
            query_hash = self._hash_query(exact_query)
            for key in all_keys:
                entry = self.r.hgetall(key)
                if not entry:
                    stale_keys.append(key)
                    continue

                if entry.get("query_hash") == query_hash:
                    if is_expired(entry):
                        stale_keys.append(key)
                        break
                    if is_poisoned(entry):
                        stale_keys.append(key)
                        break
                    self._cleanup(stale_keys)
                    entry["similarity"] = 1.0
                    entry["confidence"] = _confidence(1.0, entry)
                    entry["match_type"] = "exact_match"
                    entry["deduplicated"] = True
                    entry["explanation"] = (
                        "Exact cache hit using normalised query hash. "
                        "No embedding computation or LLM call was required."
                    )
                    return entry

        self._cleanup(stale_keys)

        # ── Layer 2: Semantic match ─────────────────────────────────────
        if embedding is None:
            return None

        all_keys = self.r.smembers(self._index_key())
        best_entry = None
        best_score = 0.0
        stale_keys = []

        for key in all_keys:
            entry = self.r.hgetall(key)
            if not entry:
                stale_keys.append(key)
                continue
            if is_expired(entry):
                stale_keys.append(key)
                continue
            if is_poisoned(entry):
                stale_keys.append(key)
                continue

            try:
                stored_embedding = json.loads(entry["embedding"])
            except (KeyError, json.JSONDecodeError):
                stale_keys.append(key)
                continue

            score = _cosine_similarity(embedding, stored_embedding)
            if score >= self.similarity_threshold and score > best_score:
                best_score = score
                best_entry = dict(entry)
                best_entry["similarity"] = round(score, 10)
                best_entry["confidence"] = _confidence(score, entry)
                best_entry["match_type"] = "semantic_match"
                best_entry["deduplicated"] = False
                best_entry["explanation"] = (
                    "Semantic similarity cache hit. "
                    "Embedding comparison was used to retrieve the cached response."
                )

        self._cleanup(stale_keys)
        return best_entry

    # ------------------------------------------------------------------
    # Layer 3 – Store
    # ------------------------------------------------------------------

    def store(
        self,
        query: str,
        embedding: "list[float]",
        response: str,
        metadata: "dict | None" = None,
    ) -> str:
        """Persist a new query/response pair. Returns the Redis key."""
        entry_id = str(uuid.uuid4())
        redis_key = self._redis_key(entry_id)

        entry = CacheEntry(
            id=entry_id,
            query=query,
            query_hash=self._hash_query(query),
            embedding=json.dumps(embedding),
            response=response,
            created_at=int(time.time()),
            ttl=default_ttl(),
            metadata=metadata or {},
        )

        mapping = entry.model_dump()
        mapping["metadata"] = json.dumps(mapping["metadata"])
        self.r.hset(redis_key, mapping=mapping)
        self.r.sadd(self._index_key(), redis_key)
        return redis_key

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _cleanup(self, keys: "list[str]") -> None:
        if not keys:
            return
        self.r.delete(*keys)
        self.r.srem(self._index_key(), *keys)

    def flush(self) -> int:
        all_keys = list(self.r.smembers(self._index_key()))
        if all_keys:
            self.r.delete(*all_keys)
        self.r.delete(self._index_key())
        return len(all_keys)

    def entry_count(self) -> int:
        return self.r.scard(self._index_key())