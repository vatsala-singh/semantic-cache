from typing import Optional, Dict
from pydantic import BaseModel, Field


class CacheEntry(BaseModel):
    """
    Schema for a single semantic cache entry stored as a Redis hash.

    Fields
    ------
    id          : UUID string used to construct the Redis key.
    query       : Original user input (for debugging / inspection).
    query_hash  : SHA-256 of the normalised query, enabling exact-match lookup.
    embedding   : JSON-serialised float vector (stored as string in Redis).
    response    : LLM response text.
    created_at  : Unix timestamp (int) of when the entry was created.
    ttl         : Lifetime in seconds; checked at read time, not by Redis TTL.
    metadata    : Optional contextual dict (pipeline, model, origin, etc.).
    """

    id: str
    query: str
    query_hash: str
    embedding: str          # JSON-serialised list[float]
    response: str
    created_at: int
    ttl: int
    metadata: Optional[Dict] = Field(default_factory=dict)