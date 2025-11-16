# MeVe Framework - AI Coding Agent Instructions

## Project Overview

MeVe is a **5-phase RAG retrieval pipeline** that optimizes context selection for LLMs through progressive filtering and budgeting. Unlike simple vector search, MeVe combines vector search, cross-encoder verification, BM25 fallback, deduplication, and token budgeting to deliver high-quality, budget-aware context.

## Architecture: The 5-Phase Pipeline

Each phase is **independently implemented** in `meve/phases/` and orchestrated by `MeVeEngine`:

1. **Phase 1 (kNN)**: `phase1_knn.py` - Vector similarity via ChromaDB, returns top `k_init` candidates
2. **Phase 2 (Verification)**: `phase2_verification.py` - Cross-encoder scoring, filters by `tau_relevance` threshold
3. **Phase 3 (Fallback)**: `phase3_fallback.py` - **Conditional** BM25 retrieval when `|verified| < n_min`
4. **Phase 4 (Prioritization)**: `phase4_prioritization.py` - MMR-based deduplication using `theta_redundancy`
5. **Phase 5 (Budgeting)**: `phase5_budgeting.py` - Greedy token packing within `t_max` budget

**Critical**: Phase 3 is conditional - only triggers when verification yields too few chunks. Engine logic in `meve/core/engine.py` manages this branching.

## Data Model (meve/core/models.py)

```python
# ContextChunk flows through all 5 phases, accumulating metadata
class ContextChunk:
    content: str           # The actual text
    doc_id: str           # Unique identifier
    embedding: List[float] # Set by VectorDBClient, used in Phase 1 & 4
    relevance_score: float # Set in Phase 2/3, used in Phase 4 & 5
    token_count: int      # Set in Phase 5

# MeVeConfig holds all 5 hyperparameters tuned together
MeVeConfig(k_init=20, tau_relevance=0.5, n_min=3, theta_redundancy=0.85, t_max=512)
```

**Pattern**: When adding features, preserve this metadata flow - each phase expects certain fields populated by previous phases.

## Critical Developer Workflows

### Running the System

```bash
make run              # Runs main.py with HotpotQA data (loads 50 examples by default)
make example          # Runs examples/basic_usage.py with synthetic data
make test             # Runs pytest on __tests__/ directory
```

### Data Loading Pattern (IMPORTANT)

The HotpotQA data structure in `meve/core/engine.py::load_hotpotqa_data()` expects:

```python
# JSON structure: context is a DICT with parallel arrays
example['context'] = {
    'title': ['Doc1', 'Doc2', ...],      # Array of titles
    'sentences': [[sent1, sent2], [...]] # Array of sentence lists
}
# Pair them with zip(titles, sentences_list)
```

**Not** a list of `[title, sentences]` tuples (common mistake - already fixed). When processing new datasets, check structure first.

### Testing Patterns

Tests use fixtures from `__tests__/fixtures/sample_data.py`. Pattern:

```python
@pytest.fixture
def sample_chunks():
    return {
        "doc1": ContextChunk("content", "doc1"),
        ...
    }

def test_engine_run(sample_chunks, default_config):
    engine = MeVeEngine(default_config, sample_chunks, sample_chunks)
    # Note: both vector_store and bm25_index use same chunks dict
```

## Project-Specific Conventions

### 1. Vector Store = BM25 Index (Current Implementation)

Both `vector_store` and `bm25_index` parameters receive **the same `Dict[str, ContextChunk]`**. They're separate parameters to allow future implementation of different indexes, but currently share data.

### 2. VectorDBClient Wraps ChromaDB

`meve/services/vector_db_client.py` abstracts ChromaDB. Pattern:

- Creates **in-memory** collection by default (`is_persistent=False`)
- Auto-encodes chunks using `SentenceTransformer('all-MiniLM-L6-v2')`
- Returns `(similarities, indices)` tuple mimicking FAISS API

**When debugging empty results**: Check that chunks list is non-empty before VectorDBClient initialization (ChromaDB requires non-empty lists).

### 3. Configuration Loading

YAML configs in `config/` use this hierarchy:

- `default.yaml` - base settings
- `development.yaml` - overrides for dev (smaller `t_max`, debug logging)
- `production.yaml` - overrides for prod

**Pattern**: Load config by environment, merge with defaults. Currently not fully implemented - configs exist but hardcoded `MeVeConfig()` calls are used in `main.py` and examples.

### 4. Phase Output Contracts

Each `execute_phase_X()` function:

- Takes `query`, `chunks/candidates`, `config`
- Returns `List[ContextChunk]` (except Phase 5 returns `(str, List[ContextChunk])`)
- Prints progress to console (not optional - debugging relies on it)
- Mutates chunk objects to add metadata (relevance_score, token_count)

### 5. Model Loading is Global

Cross-encoder in `phase2_verification.py` loads at module level:

```python
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
```

This is intentional - models persist across queries for performance. Don't move inside functions.

## Integration Points

### MCP Server (integrations/mcp/server.py)

Wraps `MeVeEngine` as Model Context Protocol server. Pattern:

```python
class MeVeContextServer:
    def get_context(self, query: str) -> str:
        return self.engine.run(query)
```

Not fully implemented yet - placeholder for AI agent integration.

### Future Agent Integration (integrations/agents/)

`context_provider.py` designed for LangChain integration. Will provide `ContextRetriever` interface.

## Common Pitfalls & Solutions

1. **Empty chunks returned**: VectorDBClient fails with empty list. Check data loading before initialization.

2. **Phase 3 never triggers**: If `tau_relevance` too low, Phase 2 always verifies enough chunks. Adjust threshold or check cross-encoder scores.

3. **Token budget exceeded**: Phase 5 uses GPT-2 tokenizer. Different from actual LLM tokenizer - expect ~10% variance. This is acceptable for budgeting.

4. **Imports break**: Package structure requires `from meve import X` not `from meve.core.models import X`. `meve/__init__.py` defines public API.

## Development Commands

```bash
make install-dev      # Installs with pytest, black, ruff
make download-data    # Downloads HotpotQA to data/ directory
make clean           # Removes __pycache__, .pytest_cache
make format          # Runs black formatter
make lint            # Runs ruff linter
```

## Adding New Features

**To add a custom phase**:

1. Create `meve/phases/phase6_yourfeature.py`
2. Follow signature: `def execute_phase_6(query, chunks, config) -> List[ContextChunk]`
3. Add config param to `MeVeConfig.__init__`
4. Update `MeVeEngine.run()` to call your phase
5. Add tests in `__tests__/unit/test_phase6.py`

**To swap vector store**:

1. Implement `VectorStore` interface in `meve/services/`
2. Replace `VectorDBClient` initialization in `phase1_knn.py`
3. Update config YAML with new settings

## Code Style

- **uv** for dependency management (not pip)
- **Conventional commits**: `feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`
- **Type hints** required for public APIs
- **Docstrings** follow numpy style with `[cite: X]` references to paper
- **Print statements** in phases are debugging tools, keep them
- **NEVER** create summary docs or markdown files to document changes unless explicitly requested
- **NEVER** write or update tests unless explicitly asked

## Performance Notes

- Phase 1: O(k log n) vector search
- Phase 2: O(k) cross-encoder calls - **slowest phase**
- Phase 3: O(n) BM25 scoring when triggered
- Phase 4: O(k²) MMR deduplication
- Phase 5: O(k) tokenization

Total typical: O(k² + k log n). Optimize Phase 2 first if slow (reduce k_init or batch cross-encoder calls).
