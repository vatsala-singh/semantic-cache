# Semantic Caching System

A layered caching pipeline built with **FastAPI**, **Redis**, and **Ollama** that reuses LLM responses for semantically similar queries — reducing latency and inference costs.

## Architecture

```
Request
  │
  ▼
[Phase 1] Exact Hash Match ──────────► Cache HIT (< 1 ms)
  │ miss
  ▼
[Phase 2] Generate Embedding (Ollama nomic-embed-text)
  │
  ▼
[Phase 3] Cosine Similarity Scan ───► Cache HIT (semantic)
  │ score < threshold
  ▼
[Phase 4] LLM Inference (Ollama llama3)
  │
  ▼
Store in Redis (embedding + response + metadata)
```

## Project Structure

```
semantic-cache/
├── app/
│   ├── main.py                  # FastAPI app + router wiring
│   ├── config.py                # Settings (env / .env file)
│   ├── api/
│   │   └── ask.py               # /ask endpoint — pipeline orchestration
│   ├── cache/
│   │   └── semantic_cache.py    # Redis storage, exact match, similarity search
│   ├── embeddings/
│   │   └── embedder.py          # Ollama embedding calls
│   └── llm/
│       └── ollama_client.py     # Ollama LLM generate calls
├── tests/
│   └── test_pipeline.py         # End-to-end smoke test
├── .env.example
├── requirements.txt
└── README.md
```

## Prerequisites

| Dependency | Version | Notes |
|---|---|---|
| Python | 3.10+ | |
| Redis | any recent | Running on `localhost:6379` |
| Ollama | latest | Running on `localhost:11434` |

### Pull required Ollama models

```bash
ollama pull nomic-embed-text   # embedding model
ollama pull llama3             # LLM (or swap for any model you prefer)
```

## Setup

```bash
# 1. Clone / enter project directory
cd semantic-cache

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment (optional — defaults work out of the box)
cp .env.example .env
# Edit .env to change models, TTL, or similarity threshold

# 5. Start the server
uvicorn app.main:app --reload --port 8000
```

## API Reference

### `POST /ask`

Run a query through the caching pipeline.

**Request**
```json
{
  "query": "What is the capital of France?",
  "ttl": 7200          // optional — overrides DEFAULT_TTL (seconds)
}
```

**Response**
```json
{
  "query": "What is the capital of France?",
  "response": "Paris is the capital of France.",
  "from_cache": true,
  "cache_type": "semantic",   // "exact" | "semantic" | "miss"
  "similarity": 0.9732,
  "latency_ms": 4.21
}
```

### `DELETE /cache`

Flush all cached entries.

### `GET /cache/stats`

Return the current number of cached entries.

### `GET /health`

Health check.

## Configuration

All settings can be set via environment variables or a `.env` file:

| Variable | Default | Description |
|---|---|---|
| `REDIS_URL` | `redis://localhost:6379` | Redis connection string |
| `CACHE_PREFIX` | `semcache:` | Namespace prefix for all Redis keys |
| `DEFAULT_TTL` | `3600` | Entry lifetime in seconds |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Ollama embedding model |
| `LLM_MODEL` | `llama3` | Ollama generation model |
| `SIMILARITY_THRESHOLD` | `0.90` | Minimum cosine score to reuse a cached response |

## Running the smoke test

With the server running:

```bash
python tests/test_pipeline.py
```

Expected output:
```
Cache flushed.

Query 1 (cold): 'What is the capital of France?'
  cache_type : miss
  from_cache : False
  latency_ms : 1842.5

Query 2 (exact repeat): 'What is the capital of France?'
  cache_type : exact
  from_cache : True
  latency_ms : 1.2

Query 3 (semantic): 'Which city is the capital of France?'
  cache_type : semantic
  similarity : 0.9821
  from_cache : True
  latency_ms : 38.7
...
```

## Tuning

- **Lower `SIMILARITY_THRESHOLD`** (e.g. `0.85`) → more cache hits, slightly less precision.
- **Higher `DEFAULT_TTL`** → stale responses live longer; useful for stable factual domains.
- **Swap `LLM_MODEL`** → any model Ollama supports works without code changes.