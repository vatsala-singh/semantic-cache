import time
from config.settings import settings

def default_ttl() -> int:
    """
    Get the default Time-To-Live (TTL) value in seconds.
    """
    return settings.default_ttl

def is_expired(cached_item: dict) -> bool:
    """
    Check if a cached item has exceeded its Time-To-Live (TTL).
    Returns True if expired, False otherwise.
    """
    if not cached_item:
        return True
        
    # Example logic: assuming cached_item has 'created_at' and 'ttl' keys
    try:
        created_at = float(cached_item.get("created_at", time.time()))
    except (TypeError, ValueError):
        created_at = time.time()

    try:
        ttl = float(cached_item.get("ttl", 3600))  # default 1 hour
    except (TypeError, ValueError):
        ttl = 3600.0
    
    # If the item has lived longer than its TTL, it is expired
    if time.time() - created_at > ttl:
        return True
        
    return False
