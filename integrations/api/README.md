"""
Quick start guide for MeVe REST API.

This file provides setup and usage examples for the FastAPI REST interface.
"""

# MeVe REST API - Quick Start Guide

## Overview

The MeVe REST API exposes the 5-phase RAG retrieval pipeline as a production-ready FastAPI service. It provides:

- **Query Retrieval** - POST `/retrieve` with per-query config override
- **Metrics & Analytics** - Phase-by-phase retrieval statistics  
- **Configuration Management** - Dynamic tuning of pipeline hyperparameters
- **Health Monitoring** - System status and data source availability
- **Data Management** - Load/upload/list chunks in ChromaDB collections

## Installation

No additional dependencies beyond the base MeVe framework (FastAPI and uvicorn are already declared).

## Starting the Server

### Option 1: Using the Built-in Example

```bash
python examples/api_server.py
```

This will:
1. Load sample data (HotpotQA or synthetic)
2. Initialize MeVeEngine with default config
3. Start FastAPI server on `http://localhost:8000`
4. Auto-generate interactive API docs at `/docs`

### Option 2: Programmatic Setup

```python
from meve import MeVeEngine, MeVeConfig, ContextChunk
from integrations.api import create_app
import uvicorn

# Load your data
chunks = {...}  # Dict[str, ContextChunk]

# Initialize engine
config = MeVeConfig(k_init=20, tau_relevance=0.5, t_max=512)
engine = MeVeEngine(config, vector_store=chunks, bm25_index=chunks)

# Create and run app
app = create_app(engine)
uvicorn.run(app, host="0.0.0.0", port=8000)
```

## API Endpoints

### Health & Status

**GET /health** - System health check
```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "vector_db_ready": true,
  "bm25_index_ready": true,
  "current_config": {
    "k_init": 20,
    "tau_relevance": 0.5,
    "n_min": 3,
    "theta_redundancy": 0.85,
    "lambda_mmr": 0.6,
    "t_max": 512
  }
}
```

### Query Retrieval

**POST /retrieve** - Execute retrieval pipeline
```bash
curl -X POST http://localhost:8000/retrieve \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is photosynthesis?",
    "config": {
      "k_init": 10,
      "tau_relevance": 0.6,
      "t_max": 256
    }
  }'
```

Response:
```json
{
  "status": "success",
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "query": "What is photosynthesis?",
  "context": "Photosynthesis is a process... [full context string for LLM]",
  "chunks": [
    {
      "content": "Photosynthesis is...",
      "doc_id": "bio_1",
      "relevance_score": 0.92,
      "token_count": 45
    }
  ],
  "metrics": {
    "phase_1_candidates": 20,
    "phase_2_verified": 5,
    "phase_3_triggered": false,
    "phase_4_deduplicated": 4,
    "phase_5_final_tokens": 256,
    "total_chunks_selected": 4
  },
  "processing_time_ms": 245.3
}
```

**GET /retrieve/metrics/{request_id}** - Get metrics for previous request
```bash
curl http://localhost:8000/retrieve/metrics/550e8400-e29b-41d4-a716-446655440000
```

### Configuration Management

**GET /config** - Get current configuration
```bash
curl http://localhost:8000/config
```

**POST /config** - Update configuration
```bash
curl -X POST http://localhost:8000/config \
  -H "Content-Type: application/json" \
  -d '{
    "config": {
      "k_init": 25,
      "tau_relevance": 0.45,
      "n_min": 2,
      "theta_redundancy": 0.80,
      "lambda_mmr": 0.7,
      "t_max": 1024
    }
  }'
```

**POST /config/reset** - Reset to default configuration
```bash
curl -X POST http://localhost:8000/config/reset
```

### Data Source Management

**POST /data-sources/load** - Load ChromaDB collection
```bash
curl -X POST http://localhost:8000/data-sources/load \
  -H "Content-Type: application/json" \
  -d '{
    "config": {
      "collection_name": "my_docs",
      "is_persistent": true,
      "load_existing": true
    }
  }'
```

**GET /data-sources/info** - Get data source statistics
```bash
curl http://localhost:8000/data-sources/info
```

**POST /data-sources/upload** - Add chunks
```bash
curl -X POST http://localhost:8000/data-sources/upload \
  -H "Content-Type: application/json" \
  -d '{
    "chunks": [
      {
        "content": "New chunk text...",
        "doc_id": "new_doc_1"
      }
    ]
  }'
```

**GET /data-sources/list** - List chunks
```bash
curl "http://localhost:8000/data-sources/list?limit=10&doc_id=bio"
```

## Python Client

For programmatic access from Python, use the `MeVeClient`:

```python
from integrations.api.client import MeVeClient

# Create client
client = MeVeClient("http://localhost:8000")

# Health check
health = client.health()
print(f"System status: {health['status']}")

# Retrieve context
result = client.retrieve("What is photosynthesis?")
print(result['context'])
print(f"Retrieved in {result['processing_time_ms']:.1f}ms")

# Get metrics
metrics = client.get_metrics(result['request_id'])
print(f"Phase 2 verified: {metrics['metrics']['phase_2_verified']} chunks")

# Update configuration
client.update_config(k_init=15, t_max=800)

# List chunks
chunks = client.list_chunks(limit=20)
print(f"Total chunks available: {chunks['total_available']}")

client.close()
```

Or use context manager:
```python
with MeVeClient("http://localhost:8000") as client:
    result = client.retrieve("Your query here")
    print(result['context'])
```

## Running Tests

Test the client against a running server:

```bash
# Terminal 1: Start server
python examples/api_server.py

# Terminal 2: Run tests
python examples/test_api_client.py
```

Expected output shows:
- Health checks
- Retrieval with metrics
- Configuration management
- Data source operations
- Per-query config overrides

## API Features

### Per-Query Configuration Override

Override pipeline parameters for a single query without affecting global config:

```python
result = client.retrieve(
    "What is photosynthesis?",
    k_init=5,           # Only for this query
    tau_relevance=0.6,  # Only for this query
    t_max=256,          # Only for this query
)
```

### Request Tracing

Every request gets a unique `request_id` for tracking and debugging:

```python
result = client.retrieve("Query")
request_id = result['request_id']

# Retrieve detailed metrics later
metrics = client.get_metrics(request_id)
```

### Detailed Metrics

Access phase-by-phase statistics:

```python
metrics = result['metrics']
print(f"Phase 1 kNN candidates: {metrics['phase_1_candidates']}")
print(f"Phase 2 verified: {metrics['phase_2_verified']}")
print(f"Phase 3 BM25 triggered: {metrics['phase_3_triggered']}")
print(f"Phase 4 deduplicated: {metrics['phase_4_deduplicated']}")
print(f"Phase 5 tokens used: {metrics['phase_5_final_tokens']}")
```

## Architecture

The API consists of:

- **`app.py`** - Main FastAPI application factory
- **`schemas.py`** - Pydantic models for request/response validation
- **`request_handler.py`** - Query execution and metrics tracking
- **`routers/`** - Modular endpoint implementations:
  - `retrieval.py` - Query execution and metrics
  - `health.py` - System status
  - `config.py` - Configuration management
  - `data_sources.py` - Data source operations
- **`client.py`** - Python HTTP client library

## Error Handling

API returns structured error responses:

```json
{
  "status": "error",
  "code": "NO_DATA_SOURCE",
  "message": "No vector data source configured",
  "request_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

HTTP status codes:
- `200` - Success
- `400` - Invalid request
- `404` - Not found
- `500` - Server error

## Performance Considerations

- Phase 2 (cross-encoder verification) is typically the slowest phase
- To improve latency:
  - Reduce `k_init` (fewer initial candidates)
  - Increase `tau_relevance` (fewer cross-encoder calls)
  - Use async requests in client code
- Each request is traced with millisecond precision in `processing_time_ms`

## Integration with LLMs

Use the API as context provider for LLM chains:

```python
from langchain.llm_chain import LLMChain
from integrations.api.client import MeVeClient

client = MeVeClient("http://localhost:8000")

def get_context(query: str) -> str:
    result = client.retrieve(query)
    return result['context']

# Use with LangChain
context = get_context("What is photosynthesis?")
# Pass to LLM with context...
```

## Future Extensions

- [ ] Authentication (API key validation)
- [ ] Rate limiting per client
- [ ] Batch query processing
- [ ] WebSocket support for streaming results
- [ ] Caching of frequent queries
- [ ] Advanced analytics dashboard
