# MeVe Framework Architecture

## Overview

MeVe (Multi-phase Efficient Vector Retrieval) is a 5-phase pipeline designed for efficient context retrieval in RAG (Retrieval-Augmented Generation) systems.

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      User Query                              │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                   MeVe Engine                                │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Phase 1: kNN Search (Vector Similarity)             │  │
│  │  ├─ Encode query with SentenceTransformer            │  │
│  │  └─ Retrieve top-k candidates from vector DB         │  │
│  └───────────────────────────────────────────────────────┘  │
│                       │                                      │
│                       ▼                                      │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Phase 2: Verification (Cross-Encoder)                │  │
│  │  ├─ Score each candidate for relevance               │  │
│  │  └─ Filter by threshold τ                            │  │
│  └───────────────────────────────────────────────────────┘  │
│                       │                                      │
│                       ▼                                      │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Phase 3: Fallback (BM25) [Conditional]              │  │
│  │  ├─ Triggered if |verified| < n_min                  │  │
│  │  └─ Retrieve additional candidates via BM25          │  │
│  └───────────────────────────────────────────────────────┘  │
│                       │                                      │
│                       ▼                                      │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Phase 4: Prioritization                             │  │
│  │  ├─ Remove duplicate/redundant chunks                │  │
│  │  └─ Rank by relevance score                          │  │
│  └───────────────────────────────────────────────────────┘  │
│                       │                                      │
│                       ▼                                      │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Phase 5: Token Budgeting                            │  │
│  │  ├─ Tokenize chunks                                  │  │
│  │  └─ Pack within budget t_max                         │  │
│  └───────────────────────────────────────────────────────┘  │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Optimized Context for LLM                       │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Data Models (`meve/core/models.py`)

- **ContextChunk**: Fundamental unit of memory
  - `content`: Text content
  - `doc_id`: Unique identifier
  - `embedding`: Vector representation
  - `relevance_score`: Computed relevance
  - `token_count`: Token budget tracking

- **Query**: User input holder
  - `text`: Query string
  - `vector`: Vectorized representation

- **MeVeConfig**: Pipeline configuration
  - `k_init`: Initial retrieval count
  - `tau_relevance`: Relevance threshold
  - `n_min`: Fallback trigger
  - `theta_redundancy`: Deduplication threshold
  - `t_max`: Token budget

### 2. Engine (`meve/core/engine.py`)

The `MeVeEngine` orchestrates all five phases:

```python
class MeVeEngine:
    def run(self, query: str) -> str:
        # Execute 5-phase pipeline
        # Return optimized context
```

### 3. Phases (`meve/phases/`)

Each phase is independently implemented:

1. **phase1_knn.py**: Vector similarity search
2. **phase2_verification.py**: Cross-encoder scoring
3. **phase3_fallback.py**: BM25 retrieval
4. **phase4_prioritization.py**: Deduplication
5. **phase5_budgeting.py**: Token optimization

### 4. Services (`meve/services/`)

- **VectorDBClient**: ChromaDB integration for vector search
- **Future**: BM25Service, EmbeddingService

### 5. Integrations (`integrations/`)

- **MCP Server**: Model Context Protocol integration
- **Agent Tools**: LangChain, AutoGPT adapters
- **API**: REST/GraphQL endpoints (planned)

## Data Flow

```
Query Text
    │
    ├─→ [Phase 1] → Vector DB → Top-k candidates
    │                                │
    ├─→ [Phase 2] → Cross-Encoder → Verified chunks
    │                                │
    │                          [if |chunks| < n_min]
    │                                │
    ├─→ [Phase 3] → BM25 Index → Additional chunks
    │                                │
    │                          [Combine contexts]
    │                                │
    ├─→ [Phase 4] → Deduplication → Unique chunks
    │                                │
    └─→ [Phase 5] → Token Budget → Final context
                                     │
                              [Return to LLM]
```

## Configuration Management

Configurations are stored in `config/`:

- `default.yaml`: Base configuration
- `development.yaml`: Dev settings (debug logging, small data)
- `production.yaml`: Prod settings (optimized, persistent storage)

## Extension Points

### Adding Custom Phases

1. Create new phase file in `meve/phases/`
2. Implement `execute_phase_X(query, chunks, config)`
3. Update engine to incorporate phase
4. Add configuration parameters

### Custom Vector Stores

1. Implement `VectorStore` interface
2. Add to `meve/services/`
3. Update configuration

### Custom Integrations

1. Create directory in `integrations/`
2. Implement adapter/server
3. Add README and examples

## Performance Considerations

- **Phase 1**: O(k log n) for kNN search
- **Phase 2**: O(k) cross-encoder evaluations
- **Phase 3**: O(n) BM25 scoring (conditional)
- **Phase 4**: O(k²) for deduplication
- **Phase 5**: O(k) tokenization and packing

Total: O(k² + n) in worst case, O(k log n + k²) typical

## Security

- No external API calls by default
- All data processing is local
- Models cached locally
- Configurable persistence

## Monitoring & Logging

- Centralized logging via `meve/utils/logging.py`
- Performance metrics via `meve/utils/metrics.py`
- Per-phase timing and statistics
- Configurable log levels

## Future Architecture

- [ ] Distributed processing support
- [ ] Async pipeline execution
- [ ] Custom model fine-tuning
- [ ] Real-time index updates
- [ ] Multi-modal support
