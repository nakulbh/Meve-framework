# MeVe Framework - AI Coding Agent Instructions

## Project Overview

MeVe (Multi-phase Efficient Vector Retrieval) is a **5-phase RAG retrieval pipeline** that optimizes context selection for LLMs through progressive filtering and intelligent budgeting. Unlike simple vector search, MeVe combines vector similarity, cross-encoder verification, BM25 fallback, MMR deduplication, and token budgeting to deliver high-quality, budget-aware context.

**Version**: 0.2.0  
**Python**: 3.13+  
**Package Manager**: uv (not pip)

## Architecture: The 5-Phase Pipeline

Each phase is **independently implemented** in `meve/phases/` and orchestrated by `MeVeEngine`:

1. **Phase 1 (kNN)**: `phase1_knn.py` - Vector similarity search via ChromaDB or in-memory store, returns top `k_init` candidates
2. **Phase 2 (Verification)**: `phase2_verification.py` - Cross-encoder scoring via ModelManager singleton, filters by `tau_relevance` threshold
3. **Phase 3 (Fallback)**: `phase3_fallback.py` - **Conditional** BM25 retrieval when `|verified| < n_min`
4. **Phase 4 (Prioritization)**: `phase4_prioritization.py` - MMR-based deduplication using `theta_redundancy` and `lambda_mmr`
5. **Phase 5 (Budgeting)**: `phase5_budgeting.py` - Greedy token packing within `t_max` budget using GPT-2 tokenizer

**Critical**: Phase 3 is conditional - only triggers when verification yields too few chunks. Engine logic in `meve/core/engine.py` manages this branching.

## Data Model (meve/core/models.py)

```python
# ContextChunk flows through all 5 phases, accumulating metadata
class ContextChunk:
    content: str              # The actual text
    doc_id: str              # Unique identifier
    embedding: List[float]   # Set by VectorDBClient, used in Phase 1 & 4
    relevance_score: float   # Set in Phase 2/3, used in Phase 4 & 5
    token_count: int         # Set in Phase 5

# MeVeConfig holds all hyperparameters tuned together
MeVeConfig(
    k_init=20,              # Phase 1: Initial retrieval count
    tau_relevance=0.5,      # Phase 2: Relevance threshold
    n_min=3,                # Phase 3: Minimum verified docs
    theta_redundancy=0.85,  # Phase 4: Redundancy threshold
    lambda_mmr=0.6,         # Phase 4: MMR lambda (relevance-diversity tradeoff)
    t_max=512               # Phase 5: Token budget
)
```

**Pattern**: When adding features, preserve this metadata flow - each phase expects certain fields populated by previous phases.

## Critical Developer Workflows

### Running the System

```bash
make run              # Runs example/meve.example.py with HotpotQA data
make example          # Runs examples/basic_usage.py with synthetic data
make compare          # Compare Basic RAG vs MeVe RAG with detailed metrics
make simple-compare   # Simple side-by-side comparison (clean output)
make basic-rag        # Run Basic RAG example
make meve-rag         # Run MeVe RAG example
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
    engine = MeVeEngine(default_config, samcompre ple_chunks, sample_chunks)
    # Note: both vector_store and bm25_index use same chunks dict
```

## Project-Specific Conventions

### 1. Vector Store = BM25 Index (Current Implementation)

Both `vector_store` and `bm25_index` parameters receive **the same `Dict[str, ContextChunk]`**. They're separate parameters to allow future implementation of different indexes, but currently share data.

**MeVeEngine Initialization Options**:
The engine supports flexible initialization through multiple modes:

1. **Legacy Mode**: Pass `vector_store` and `bm25_index` dicts
2. **External DB**: Pass pre-initialized `vector_db_client` instance
3. **Auto-connect**: Pass `vector_db_config` with `load_existing=True` to load existing ChromaDB collection
4. **Auto-create**: Pass `vector_db_config` with `chunks` list to create new collection

See `meve/core/engine.py::__init__()` for detailed parameter documentation.

### 2. VectorDBClient Wraps ChromaDB

`meve/services/vector_db_client.py` abstracts ChromaDB with flexible initialization options:

**Initialization Modes**:

- **Legacy Mode**: In-memory dict of chunks (for testing)
- **Auto-create**: Creates new ChromaDB collection with provided chunks
- **Auto-connect**: Loads existing ChromaDB collection
- **Cloud Mode**: Connects to ChromaDB Cloud (requires API key)

**Key Features**:

- Creates **in-memory** collection by default (`is_persistent=False`)
- Auto-encodes chunks using `SentenceTransformer('all-MiniLM-L6-v2')` via ModelManager
- Returns `(similarities, indices)` tuple mimicking FAISS API
- Supports persistent storage and cloud deployment

**When debugging empty results**: Check that chunks list is non-empty before VectorDBClient initialization (ChromaDB requires non-empty lists).

### 3. Configuration Loading

YAML configs in `config/` use this hierarchy:

- `default.yaml` - base settings
- `development.yaml` - overrides for dev (smaller `t_max`, debug logging)
- `production.yaml` - overrides for prod

**Current State**: Configs exist but are not actively used - `MeVeConfig()` is instantiated programmatically in code. YAML configs are prepared for future environment-based configuration loading.

### 4. Phase Output Contracts

Each `execute_phase_X()` function:

- Takes `query`, `chunks/candidates`, `config` as parameters
- Returns `List[ContextChunk]` (except Phase 5 returns `(str, List[ContextChunk])`)
- Prints progress to console (not optional - debugging relies on it)
- Mutates chunk objects to add metadata (e.g., `relevance_score`, `token_count`)
- Uses logger from `meve.utils` for structured logging

**Phase-specific notes**:

- Phase 1: May receive `vector_store` dict or `vector_db_client` instance
- Phase 3: Only called conditionally when `len(verified_chunks) < n_min`
- Phase 5: Returns tuple of `(final_context_string, final_chunks_list)`

### 5. Model Management is Centralized

Models are managed through `meve/utils/model_manager.py` using the singleton pattern:

```python
# ModelManager provides lazy-loaded, cached model instances
from meve.utils import get_sentence_transformer, get_cross_encoder, get_tokenizer

# Example: Cross-encoder in phase2_verification.py
cross_encoder = get_cross_encoder()  # Returns 'cross-encoder/ms-marco-MiniLM-L-6-v2'
```

**Why Singleton?** Models persist across queries for performance. Loading happens once on first access. Don't instantiate models directly in phase functions - always use ModelManager helper functions.

### 6. Logging System

The project uses a centralized logger from `meve/utils/logger.py`:

```python
from meve.utils import get_logger

logger = get_logger(__name__)  # Scoped logger with module name

# Usage
logger.info("Phase 1 starting")
logger.warn("Low verification count", context={"count": 2})
logger.error("Model loading failed", error=exception)
```

**Features**:

- Color-coded console output with emojis (🔍, ℹ️, ⚠️, ❌)
- Structured logging with context/metadata
- Performance timing utilities
- Environment-aware (respects production mode)
- Supports external services (Sentry integration ready)

**Pattern**: Use `get_logger(__name__)` at module level, not `logging.getLogger()`. This ensures consistent formatting and features across the codebase.

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

5. **Model loading errors**: Always use ModelManager helpers (`get_cross_encoder()`, `get_sentence_transformer()`, `get_tokenizer()`) instead of directly instantiating models. This ensures singleton pattern and lazy loading.

6. **Logger not formatting correctly**: Use `from meve.utils import get_logger` and `logger = get_logger(__name__)`, not `logging.getLogger()`. The custom logger provides color-coding and structured logging.

## Development Commands

```bash
make install          # Installs dependencies with uv
make install-dev      # Installs with dev dependencies (pytest, black, ruff)
make download-data    # Downloads HotpotQA to data/ directory
make clean            # Removes __pycache__, .pytest_cache
make format           # Runs black formatter
make lint             # Runs ruff linter
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
- **Logger usage**: Always use `get_logger(__name__)` from `meve.utils`, never use standard `logging` directly
- **Model loading**: Always use ModelManager helpers (`get_cross_encoder()`, etc.), never instantiate models directly
- **NEVER** create summary docs or markdown files to document changes unless explicitly requested
- **NEVER** write or update tests unless explicitly asked

## Performance Notes

- Phase 1: O(k log n) vector search
- Phase 2: O(k) cross-encoder calls - **slowest phase**
- Phase 3: O(n) BM25 scoring when triggered
- Phase 4: O(k²) MMR deduplication
- Phase 5: O(k) tokenization

Total typical: O(k² + k log n). Optimize Phase 2 first if slow (reduce k_init or batch cross-encoder calls).
