# meve/phases/phase1_knn.py
# Phase 1: Preliminary Candidate Extraction (kNN Search)

from typing import Dict, List, Optional

from meve.core.models import ContextChunk, MeVeConfig, Query
from meve.services.vector_db_client import VectorDBClient
from meve.utils import get_logger

logger = get_logger(__name__)


def execute_phase_1(
    query: Query,
    config: MeVeConfig,
    vector_store: Optional[Dict[str, ContextChunk]] = None,
    vector_db_client: Optional[VectorDBClient] = None,
) -> List[ContextChunk]:
    """
    Phase 1 (kNN Search) [cite: MeVe Paper, Section 3.2.1]
    Initial retrieval using k-nearest neighbor search in vector space.

    Args:
        query: Query object with text and optional vector
        config: MeVe configuration
        vector_store: Optional dict of chunks (legacy mode)
        vector_db_client: Optional pre-initialized VectorDBClient (preferred)

    Returns:
        List of initial candidate chunks with relevance scores

    Raises:
        ValueError: If no data source provided or query encoding fails
    """
    try:
        # Validate input
        if not vector_store and not vector_db_client:
            logger.error("Phase 1 failed: No vector data source provided.")
            raise ValueError("No vector data source provided for Phase 1")

        # Use provided vector_db_client or create one from vector_store
        if vector_db_client:
            db_client = vector_db_client
            all_chunks = db_client.chunks
        else:
            all_chunks = list(vector_store.values())
            if not all_chunks:
                logger.error("Phase 1 failed: No chunks in vector store.")
                raise ValueError("Vector store is empty")
            db_client = VectorDBClient(all_chunks)

        # Perform kNN search using ChromaDB's text-based query
        # ChromaDB handles embedding internally with batch queries
        k = min(config.k_init, len(all_chunks))  # Don't search for more than available
        similarities, indices = db_client.query_text([query.text], k)

        # Retrieve the top-k chunks
        initial_candidates = []
        for i, idx in enumerate(indices):
            chunk = all_chunks[idx]
            # Store similarity score
            chunk.relevance_score = float(similarities[i])
            initial_candidates.append(chunk)

        logger.info(f"Phase 1: Retrieved {len(initial_candidates)} initial candidates")
        return initial_candidates

    except Exception as e:
        logger.error(f"Phase 1 failed with error: {e}", error=e)
        return []
