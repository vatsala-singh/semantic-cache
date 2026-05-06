"""
Lightweight in-memory metrics for cache observability.

Tracks hits, misses, and LLM calls since server start.
Intentionally simple — not suitable for production monitoring.
Reset on every restart.
"""

import threading

_lock = threading.Lock()

_counters = {
    "hits": 0,
    "misses": 0,
    "llm_calls": 0,
    "bypasses": 0,
}


def cache_hit() -> None:
    with _lock:
        _counters["hits"] += 1


def cache_miss() -> None:
    with _lock:
        _counters["misses"] += 1


def llm_call() -> None:
    with _lock:
        _counters["llm_calls"] += 1


def bypass() -> None:
    with _lock:
        _counters["bypasses"] += 1


def snapshot() -> dict:
    with _lock:
        return {
            **_counters,
            "_note": "In-memory metrics. Reset on restart. Not production-ready.",
        }


def reset() -> None:
    with _lock:
        for key in _counters:
            _counters[key] = 0