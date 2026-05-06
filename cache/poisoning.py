"""
Cache poisoning guards.

An entry is considered "poisoned" if its response text indicates an error or
is empty / whitespace-only. Poisoned entries are never served to users and
are removed from the index during cleanup.
"""

# Prefixes that identify LLM error responses
_ERROR_PREFIXES = ("[LLM Error]", "[Error]", "[ERROR]")


def is_poisoned(entry: dict) -> bool:
    """
    Return True if the cache entry should not be reused.

    An entry is poisoned when:
    - The response is absent or whitespace-only.
    - The response begins with a known error prefix.
    """
    response: str = entry.get("response", "")
    if not response or not response.strip():
        return True
    for prefix in _ERROR_PREFIXES:
        if response.startswith(prefix):
            return True
    return False


def is_safe_to_cache(response: str) -> bool:
    """
    Return True if a freshly generated LLM response is safe to store.

    This is the complement of is_poisoned, applied before writing.
    """
    if not response or not response.strip():
        return False
    for prefix in _ERROR_PREFIXES:
        if response.startswith(prefix):
            return False
    return True