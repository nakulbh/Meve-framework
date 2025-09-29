# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the **MeVe Framework** - an implementation of the MeVe (Memory-Enhanced Vector) RAG pipeline research. MeVe is a five-phase retrieval system designed for context efficiency and control in RAG applications.

## Architecture

The system implements a modular 5-phase RAG pipeline:

1. **Phase 1** (`phase1_knn.py`): Initial kNN retrieval using vector similarity
2. **Phase 2** (`phase2_verification.py`): Cross-encoder relevance verification with threshold filtering
3. **Phase 3** (`phase3_fallback.py`): BM25 fallback retrieval when verified chunks < n_min
4. **Phase 4** (`phase4_prioritization.py`): Context prioritization considering relevance and redundancy
5. **Phase 5** (`phase5_budgeting.py`): Token budgeting with greedy packing algorithm

### Core Components

- `meve_engine.py`: Main orchestration engine that runs the complete pipeline
- `meve_data.py`: Core data structures (`ContextChunk`, `Query`, `MeVeConfig`)
- Each phase is implemented as a separate module with an `execute_phase_X` function

### Key Data Structures

- **ContextChunk**: Fundamental memory unit with content, embeddings, relevance scores, and token counts
- **Query**: User input with text and vector representations  
- **MeVeConfig**: Configurable hyperparameters for all pipeline phases

## Development Commands

**Package Manager**: UV is used for dependency management.

**Run the framework**:
```bash
uv run meve_engine.py
```

**Install dependencies**:
```bash
uv install
```

**Add new dependencies**:
```bash
uv add <package-name>
```

## Configuration Parameters

Key hyperparameters in `MeVeConfig`:
- `k_init`: Initial retrieval count (default: 20)
- `tau_relevance`: Relevance threshold for verification (default: 0.5)  
- `n_min`: Minimum verified docs to avoid fallback (default: 3)
- `theta_redundancy`: Redundancy threshold for prioritization (default: 0.85)
- `t_max`: Maximum token budget (default: 512)

## Dependencies

Key ML/NLP dependencies:
- sentence-transformers, transformers: For embedding models
- faiss-cpu: Vector similarity search
- rank-bm25: BM25 implementation for fallback
- torch: PyTorch backend
- numpy, pandas: Data manipulation
- nltk: Text processing

## Testing and Development Notes

- The current implementation uses simulated data and mock scoring functions for demonstration
- Real implementations would integrate with actual vector stores (FAISS) and cross-encoder models
- The system is designed to handle two scenarios: successful verification (minimal fallback) and failed verification (fallback required)