# meve_data.py

from typing import List, Optional


class ContextChunk:
    """The fundamental unit of memory flowing through the MeVe pipeline."""

    def __init__(
        self,
        content: str,
        doc_id: str,
        embedding: Optional[List[float]] = None,
        title: Optional[str] = None,
        question_id: Optional[str] = None,
        source_type: str = "unknown",
        metadata: Optional[dict] = None
    ):
        self.content = content
        self.doc_id = doc_id
        self.embedding = embedding  # Used in Phase 1 and Phase 4
        self.title = title  # Document title from source dataset
        self.question_id = question_id  # Associated question ID (for HotpotQA)
        self.source_type = source_type  # Type of source (e.g., "hotpotqa", "wikipedia")
        self.metadata = metadata or {}  # Additional metadata dict for extensibility
        self.relevance_score: float = 0.0  # Set in Phase 2/3
        self.token_count: int = 0  # Set in Phase 5

    def __repr__(self):
        return (
            f"Chunk(ID={self.doc_id[:5]}, Title={self.title or 'N/A'}, "
            f"Score={self.relevance_score:.2f}, Tokens={self.token_count}, "
            f"Content='{self.content[:30]}...')"
        )


class Query:
    """Holds the user's input and its vectorized representation."""

    def __init__(self, text: str, vector: Optional[List[float]] = None):
        self.text = text
        self.vector = vector


class MeVeConfig:
    """Tunable hyperparameters for the MeVe pipeline [cite: 134, 349-355]."""

    def __init__(
        self,
        k_init: int = 20,  # Phase 1: Initial Retrieval Count (k) [cite: 351]
        tau_relevance: float = 0.5,  # Phase 2: Relevance Threshold (τ) [cite: 352]
        n_min: int = 3,  # Phase 3: Minimum Verified Documents (Nmin) [cite: 353]
        theta_redundancy: float = 0.85,  # Phase 4: Redundancy Threshold (θredundancy) [cite: 354]
        lambda_mmr: float = 0.6,  # Phase 4: MMR Lambda for relevance-diversity tradeoff
        t_max: int = 512,  # Phase 5: Token Budget (Tmax) [cite: 355]
        embedding_model: str = "text-embedding-3-small",  # Embedding model for vector DB
    ):
        self.k_init = k_init
        self.tau_relevance = tau_relevance
        self.n_min = n_min
        self.theta_redundancy = theta_redundancy
        self.lambda_mmr = lambda_mmr
        self.t_max = t_max
        self.embedding_model = embedding_model
