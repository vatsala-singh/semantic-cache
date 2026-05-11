# Semantic Cache Service - Setup and Run Guide

This guide provides detailed steps to set up and start the Semantic Cache service, a FastAPI-based LLM backend system with semantic caching capabilities.

## Overview

The Semantic Cache service is built with:
- **FastAPI**: For creating the API endpoints
- **UVicorn**: For running the ASGI application
- **Redis**: For storing cached responses
- **Ollama**: For LLM and embedding models
- **Pydantic**: For data validation

---

## Prerequisites

Before starting, ensure you have:
- Python 3.11+ installed
- macOS with Homebrew (for Redis installation)
- Terminal access
- Internet connection (for downloading packages and models)

---

## Step 1: Install System Dependencies

### 1.1 Install Redis using Homebrew

```bash
# Install Redis package manager
brew install redis

# Start Redis service in the background
brew services start redis

# Verify Redis is running
redis-cli ping
# Expected output: PONG
```

### 1.2 Install Ollama

Ollama is required for LLM responses and text embeddings.

```bash
# Download and install from https://ollama.ai
# Or use Homebrew if available:
brew install ollama

# Start Ollama in the background (if using Homebrew)
brew services start ollama
```

---

## Step 2: Python Environment Setup

### 2.1 Navigate to Project Directory

```bash
cd /Users/vatsalasingh/Documents/GitHub/Semantic-cache
```

### 2.2 Create Python Virtual Environment

```bash
# Create a virtual environment named 'semantic-cache'
python3 -m venv semantic-cache

# Activate the virtual environment
source semantic-cache/bin/activate

# Verify activation (you should see 'semantic-cache' in your prompt)
which python
```

### 2.3 Upgrade pip

```bash
pip install --upgrade pip setuptools wheel
```

### 2.4 Install Python Dependencies

```bash
# Install all required Python packages from requirements file
pip install -r requirement.txt

# Verify installations (optional but recommended)
pip list
```

**Expected packages:**
- fastapi
- uvicorn
- redis
- numpy
- pydantic
- httpx
- python-dotenv

---

## Step 3: Download Ollama Models

### 3.1 Download Embedding Model

The embedding model is used to generate vector embeddings for semantic similarity matching.

```bash
# Pull the nomic-embed-text model (lightweight, fast)
ollama pull nomic-embed-text

# This may take a few minutes depending on internet speed
```

### 3.2 Download LLM Model

The LLM model is used to generate responses when cache misses occur.

```bash
# Pull the llama3.2 model (or your preferred model)
ollama pull llama3.2

# This may take several minutes to download
```

### 3.3 Verify Models are Downloaded

```bash
# List all downloaded models
ollama list
```

You should see:
- `nomic-embed-text:latest`
- `llama3.2:latest`

---

## Step 4: Environment Configuration

### 4.1 Check Environment Variables

The service uses `.env` file for configuration. Check if it exists:

```bash
ls -la .env
```

### 4.2 Create or Update `.env` file (if needed)

If `.env` doesn't exist, check `config/settings.py` for default values or create one:

```bash
cat > .env << 'EOF'
# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
EMBEDDER_MODEL=nomic-embed-text
LLM_MODEL=llama3.2

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# Cache Configuration
SIMILARITY_THRESHOLD=0.85
CONFIDENCE_GATE=0.75
CACHE_TTL_SECONDS=3600
EOF
```

---

## Step 5: Verify Service Dependencies

### 5.1 Test Redis Connection

```bash
redis-cli ping
# Expected: PONG
```

### 5.2 Test Ollama Service

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Should return JSON with available models
```

### 5.3 Test Python Dependencies

```bash
# Run the test pipeline
python test_pipeline.py

# This validates all imports and basic functionality
```

---

## Step 6: Start the Service

### 6.1 Activate Virtual Environment (if not already active)

```bash
source semantic-cache/bin/activate
```

### 6.2 Start the FastAPI Service

```bash
# Start the development server with auto-reload
uvicorn main:app --reload

# For production, use:
# uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

**Expected output:**
```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete
```

---

## Step 7: Verify Service is Running

### 7.1 Test Health Endpoint (in a new terminal)

```bash
# Ensure virtual environment is still active
source semantic-cache/bin/activate

# Test health check
curl http://localhost:8000/health

# Expected response: {"status":"ok"}
```

### 7.2 Access API Documentation

Open your browser and navigate to:

```
http://localhost:8000/docs
```

This provides interactive Swagger UI documentation for all endpoints.

Alternative documentation:
```
http://localhost:8000/redoc
```

### 7.3 Test the /ask Endpoint

```bash
# Make a sample query
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What is semantic caching?", "bypass_cache": false}'

# Expected: JSON response with answer and cache metadata
```

### 7.4 Check Metrics

```bash
# View cache hit/miss statistics
curl http://localhost:8000/internal/metrics

# Expected: JSON with metrics like hits, misses, llm_calls
```

---

## Complete Service Startup Checklist

Use this checklist to ensure everything is running correctly:

- [ ] Redis is running: `redis-cli ping` returns `PONG`
- [ ] Ollama is running: `curl http://localhost:11434/api/tags` succeeds
- [ ] Models downloaded: `ollama list` shows `nomic-embed-text` and `llama3.2`
- [ ] Python venv activated: Prompt shows `(semantic-cache)`
- [ ] Dependencies installed: `pip list` shows all required packages
- [ ] Service started: `uvicorn main:app --reload` shows "Application startup complete"
- [ ] Health check passes: `curl http://localhost:8000/health` returns `{"status":"ok"}`
- [ ] API docs accessible: `http://localhost:8000/docs` opens in browser

---

## Available Endpoints

Once the service is running:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/ask` | POST | Submit a query for semantic caching pipeline |
| `/health` | GET | Health check endpoint |
| `/internal/metrics` | GET | View cache performance metrics |
| `/internal/cache` | DELETE | Flush all cache entries |
| `/docs` | GET | Interactive API documentation (Swagger UI) |
| `/redoc` | GET | Alternative API documentation (ReDoc) |

---

## Troubleshooting

### Redis Connection Error
```
Error: Unable to connect to Redis at localhost:6379
```
**Solution:**
```bash
brew services restart redis
redis-cli ping  # Should return PONG
```

### Ollama Model Not Found
```
Error: Model 'llama3.2' not found
```
**Solution:**
```bash
ollama pull llama3.2
# Wait for download to complete
```

### Port 8000 Already in Use
```
Error: Address already in use
```
**Solution:**
```bash
# Use a different port
uvicorn main:app --port 8001 --reload

# Or kill the process using port 8000
lsof -i :8000
kill -9 <PID>
```

### Virtual Environment Issues
```
Command not found: python
```
**Solution:**
```bash
# Ensure venv is activated
source semantic-cache/bin/activate

# Verify Python path
which python
```

---

## Stopping the Service

### Stop FastAPI Server
```bash
# Press Ctrl+C in the terminal running uvicorn
```

### Stop Redis
```bash
brew services stop redis
```

### Stop Ollama
```bash
brew services stop ollama
```

### Deactivate Virtual Environment
```bash
deactivate
```

---

## Next Steps

1. **Explore the API**: Visit `http://localhost:8000/docs` for interactive documentation
2. **Make queries**: Use the `/ask` endpoint to test the semantic caching pipeline
3. **Monitor performance**: Check `/internal/metrics` for cache statistics
4. **Review logs**: Check the terminal output for request/response details
5. **Develop**: Start building on top of the API with your own clients

---

## Additional Resources

- **FastAPI Documentation**: https://fastapi.tiangolo.com/
- **Ollama Models**: https://ollama.ai/
- **Redis Documentation**: https://redis.io/documentation
- **Project Documentation**: See `Readme.md` and `BLOG.md` for architecture details
