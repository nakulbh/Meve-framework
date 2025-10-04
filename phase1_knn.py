# phase_1_knn.py
# Phase 1: Preliminary Candidate Extraction (kNN Search)

from meve_data import ContextChunk, Query, MeVeConfig
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from services.vector_db_client import VectorDBClient

def execute_phase_1(query: Query, config: MeVeConfig, vector_store: Dict[str, ContextChunk]) -> List[ContextChunk]:
    """
    Phase 1: Preliminary Candidate Extraction (kNN Search).
    Retrieves the k_init closest candidates based on dense similarity using Vector DB client.
    """
    print(f"--- Phase 1: Initial Retrieval (kNN={config.k_init}) ---")
    
    # Initialize sentence transformer for query encoding if needed
    if not query.vector:
        print("Encoding query using sentence transformer...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
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
    
    print(f"Retrieved {len(initial_candidates)} initial candidates using Vector DB client.")
    return initial_candidates