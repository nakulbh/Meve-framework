# MeVe Framework Architecture

## Overview

MeVe (Multi-phase Efficient Vector Retrieval) is a 5-phase pipeline designed for efficient context retrieval in RAG (Retrieval-Augmented Generation) systems.

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│ PHASE 1: Vector Search (kNN)                            │
│ • Uses: embedding_model                                 │
│ • Returns: top k_init chunks                            │
└────────────────┬────────────────────────────────────────┘
                 ↓
┌─────────────────────────────────────────────────────────┐
│ PHASE 2: Verification (Cross-Encoder)                   │
│ • Filters by: tau_relevance threshold                   │
│ • If verified < n_min → trigger Phase 3                 │
└────────────────┬────────────────────────────────────────┘
                 ↓
        ┌────────┴────────┐
        ↓                 ↓
    [n_min check]    [Phase 3?]
        │                 │
        ↓                 ↓
┌──────────────┐  ┌─────────────────────────────────────┐
│ Phase 3 OFF  │  │ PHASE 3: BM25 Fallback (if n<n_min) │
│ (enough✓)    │  │ • Adds more candidates              │
└──────────────┘  │ • Merges with Phase 2 results       │
        │         └──────────────────┬──────────────────┘
        └────────────────┬───────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│ PHASE 4: MMR Deduplication                              │
│ • Uses: theta_redundancy (remove >80% similar)          │
│ • Uses: lambda_mmr (balance relevance vs diversity)     │
│ • Uses: embedding_model (similarity calculation)        │
└────────────────┬────────────────────────────────────────┘
                 ↓
┌─────────────────────────────────────────────────────────┐
│ PHASE 5: Token Budgeting                                │
│ • Packs chunks greedily until t_max budget reached      │
│ • Returns: final string + chunk list                    │
└─────────────────────────────────────────────────────────┘
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
