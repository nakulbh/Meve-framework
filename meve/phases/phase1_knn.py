# meve/phases/phase1_knn.py
# Phase 1: Preliminary Candidate Extraction (kNN Search)

from typing import Dict, List

from sentence_transformers import SentenceTransformer

from meve.core.models import ContextChunk, MeVeConfig, Query
from meve.services.vector_db_client import VectorDBClient
from meve.utils import get_logger

logger = get_logger(__name__)


def execute_phase_1(
    query: Query,
    config: MeVeConfig,
    vector_store: Dict[str, ContextChunk],
) -> List[ContextChunk]:
    """
    Phase 1 (kNN Search) [cite: MeVe Paper, Section 3.2.1]
    Initial retrieval using k-nearest neighbor search in vector space.
    """
    logger.info(f"--- Phase 1: Initial Retrieval (kNN={config.k_init}) ---")

    # Create VectorDBClient and encode query
    if not vector_store:
        logger.warning("No initial candidates to verify.")
        return []

    logger.debug("Encoding query using sentence transformer...")

    # Initialize sentence transformer for query encoding if needed
    if not query.vector:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        query.vector = model.encode(query.text).tolist()

    # Get all chunks from vector store
    all_chunks = list(vector_store.values())

    # Initialize ChromaDB Vector DB client (replaces FAISS operations)
    vector_db = VectorDBClient(all_chunks)

    # Perform kNN search using vector DB client
    k = min(config.k_init, len(all_chunks))  # Don't search for more than available
    similarities, indices = vector_db.query(query.vector, k)

    # Retrieve the top-k chunks
    initial_candidates = []
    for i, idx in enumerate(indices):
        chunk = all_chunks[idx]
        # Store similarity score
        chunk.relevance_score = float(similarities[i])
        initial_candidates.append(chunk)

    logger.info(f"Retrieved {len(initial_candidates)} initial candidates using Vector DB client.")
    return initial_candidates
