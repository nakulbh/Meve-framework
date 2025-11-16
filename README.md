# MeVe Framework

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-0.2.0-green.svg)](https://github.com/nakulbh/Meve-framework)

**Multi-phase Efficient Vector Retrieval** - A 5-phase RAG pipeline that optimizes context selection through progressive filtering and intelligent budgeting.

Unlike simple vector search, MeVe combines **vector similarity**, **cross-encoder verification**, **BM25 fallback**, **MMR deduplication**, and **token budgeting** to deliver high-quality, budget-aware context for LLMs.

---

## 🎯 Why MeVe?

Traditional RAG systems often:
- Return irrelevant chunks despite high similarity scores
- Waste tokens on redundant information
- Fail silently when vector search underperforms
- Ignore token budget constraints

MeVe solves these problems with a **smart 5-phase pipeline**:

✅ **Quality First** - Cross-encoder verification ensures relevance  
✅ **Adaptive Fallback** - BM25 backup when vector search fails  
✅ **Zero Redundancy** - MMR-based deduplication  
✅ **Budget Aware** - Greedy token packing within limits  
✅ **Production Ready** - Tested on HotpotQA dataset  

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/nakulbh/Meve-framework.git
cd meve

# Install with uv (recommended)
uv pip install -e .

# Or with pip
pip install -e .
```

### Basic Usage

```python
from meve import MeVeEngine, MeVeConfig, ContextChunk

# 1. Prepare your data
chunks = {
    "doc1": ContextChunk("The Eiffel Tower is in Paris, France.", "doc1"),
    "doc2": ContextChunk("Paris is the capital of France.", "doc2"),
    "doc3": ContextChunk("The Louvre Museum is in Paris.", "doc3"),
}

# 2. Configure the pipeline
config = MeVeConfig(
    k_init=10,           # Initial candidates from vector search
    tau_relevance=0.5,   # Relevance threshold (0-1)
    n_min=3,             # Min chunks to avoid fallback
    theta_redundancy=0.85,  # Similarity threshold for deduplication
    t_max=512            # Maximum token budget
)

# 3. Initialize engine (vector_store and bm25_index use same chunks)
engine = MeVeEngine(config, chunks, chunks)

# 4. Retrieve context
context = engine.run("Where is the Eiffel Tower?")
print(context)
```

### Run with HotpotQA Data

```bash
# Download HotpotQA dataset
make download-data

# Run with real data (loads 50 examples by default)
make run

# Or run the basic example
make example
```

---

## 📊 Pipeline Architecture

```
Query → kNN Search → Verification → [Fallback?] → Deduplication → Budgeting → Context
         ↓              ↓              ↓              ↓              ↓
      Phase 1        Phase 2        Phase 3        Phase 4        Phase 5
```

### The 5 Phases
<img width="836" height="474" alt="image" src="https://github.com/user-attachments/assets/eed74925-baee-4729-90d6-a76b9e96413b" />


#### Phase Descriptions

1. **Phase 1 (kNN)** - Vector similarity search via ChromaDB  
   Returns top `k_init` candidates using `all-MiniLM-L6-v2` embeddings

2. **Phase 2 (Verification)** - Cross-encoder re-ranking  
   Uses `cross-encoder/ms-marco-MiniLM-L-6-v2` to filter by `tau_relevance` threshold

3. **Phase 3 (Fallback)** - Conditional BM25 retrieval  
   **Only triggers when** `|verified| < n_min` - supplements with lexical search

4. **Phase 4 (Prioritization)** - MMR-based deduplication  
   Removes redundant chunks using `theta_redundancy` similarity threshold

5. **Phase 5 (Budgeting)** - Greedy token packing  
   Fits top chunks within `t_max` budget using GPT-2 tokenizer

---

## ⚙️ Configuration

### Hyperparameters

| Parameter          | Type  | Default | Description                            |
| ------------------ | ----- | ------- | -------------------------------------- |
| `k_init`           | int   | 10      | Initial candidates from vector search  |
| `tau_relevance`    | float | 0.5     | Cross-encoder threshold (0-1)          |
| `n_min`            | int   | 3       | Min verified chunks to skip fallback   |
| `theta_redundancy` | float | 0.85    | Similarity threshold for deduplication |
| `t_max`            | int   | 512     | Maximum token budget                   |

### Example Configurations

```python
# Development - Fast iteration
config = MeVeConfig(k_init=5, tau_relevance=0.3, n_min=2, t_max=256)

# Production - Quality focus
config = MeVeConfig(k_init=20, tau_relevance=0.6, n_min=5, t_max=1024)

# Tight budget - Minimal tokens
config = MeVeConfig(k_init=10, tau_relevance=0.7, n_min=2, t_max=128)
```

---

## 🛠️ Development

### Setup

```bash
# Install development dependencies
make install-dev

# Run tests
make test

# Format code
make format

# Lint code
make lint

# Clean cache
make clean
```

### Project Structure

```
meve/
├── core/
│   ├── engine.py          # MeVeEngine orchestrator
│   └── models.py          # ContextChunk, MeVeConfig, Query
├── phases/
│   ├── phase1_knn.py      # Vector search
│   ├── phase2_verification.py  # Cross-encoder
│   ├── phase3_fallback.py # BM25 fallback
│   ├── phase4_prioritization.py  # MMR deduplication
│   └── phase5_budgeting.py  # Token packing
├── services/
│   └── vector_db_client.py  # ChromaDB wrapper
└── utils/
```

### Adding Custom Phases

```python
# meve/phases/phase6_custom.py
def execute_phase_6(query: str, chunks: List[ContextChunk], config: MeVeConfig):
    """Your custom phase logic."""
    # Process chunks
    return processed_chunks

# Update MeVeEngine.run() to call your phase
# Add parameters to MeVeConfig if needed
```

---

## 📚 Use Cases

- **Question Answering** - Retrieve precise context for factual queries
- **Chatbots** - Budget-aware context for conversational AI
- **Document Search** - Hybrid vector + lexical retrieval
- **Knowledge Bases** - Deduplicated, relevant snippets

---

## 🧪 Testing

```bash
# Run all tests
pytest

# Run specific test file
pytest __tests__/unit/test_engine.py

# Run with coverage
pytest --cov=meve
```

Test fixtures available in `__tests__/fixtures/sample_data.py`.

---

## 📖 Documentation

- [Architecture Guide](docs/architecture.md)
- [Data Format Specification](docs/data.md)
- [API Reference](docs/api/)

---

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

**Commit Convention**: `feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`

---

## 📝 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 🔗 Related

- [HotpotQA Dataset](https://hotpotqa.github.io/)
- [ChromaDB](https://www.trychroma.com/)
- [Sentence Transformers](https://www.sbert.net/)
- [Cross-Encoders](https://www.sbert.net/examples/applications/cross-encoder/README.html)

---

**Built with ❤️ for the RAG community**
