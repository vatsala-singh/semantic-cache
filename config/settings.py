from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_url: str = "redis://localhost:6379"
    cache_prefix: str = "semcache:"
    default_ttl: int = 3600  # seconds
 
    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    embedding_model: str = "nomic-embed-text"
    llm_model: str = "llama3.2"
 
    # Similarity
    similarity_threshold: float = 0.90
    confidence_threshold: float = 0.75
    
    # Computed properties for uppercase constants
    @property
    def REDIS_HOST(self) -> str:
        return self.redis_host
    
    @property
    def REDIS_PORT(self) -> int:
        return self.redis_port
    
    @property
    def CACHE_NAMESPACE(self) -> str:
        return self.cache_prefix
    
    @property
    def DEFAULT_TTL(self) -> int:
        return self.default_ttl
    
    @property
    def SIMILARITY_THRESHOLD(self) -> float:
        return self.similarity_threshold

    @property
    def CONFIDENCE_THRESHOLD(self) -> float:
        return self.confidence_threshold

    @property
    def OLLAMA_HOST(self) -> str:
        return self.ollama_base_url
    @property
    def OLLAMA_PORT(self) -> int:
        # Extract port from base URL, default to 11434 if not specified
        try:
            return int(self.ollama_base_url.split(":")[-1])
        except (IndexError, ValueError):
            return 11434
    @property
    def EMBEDDING_MODEL(self) -> str:
        return self.embedding_model
 
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
 
 
settings = Settings()