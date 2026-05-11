Refer to 'setup_and_run.md' file for setup

- Fast API: for creating the API.
- UVicorn: for running the API.
- Redis: for storing the cache.
- HTTX: for making HTTP requests.
- Python-dotenv: for managing the environment variables.
- Numpy: for numerical operations.

To verify redis: redis-cli ping

- use brew to install redis
    - brew install redis
    - brew services start redis

- To start the server: uvicorn app.main:app --reload

Ollama Setting:
- Use 'nomic-embed-text' to generate embeddings for the input text.
command:
- ollama pull nomic-embed-text
- ollama pull llama3.2

Project structure for LLM backend system:
- API orchestration
- Caching
- Embedding
- Model Call
- Observability
