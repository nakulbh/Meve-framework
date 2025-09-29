# phase_3_fallback.py

from meve_data import ContextChunk, Query, MeVeConfig
from typing import List, Dict
# Assume a BM25 index of the knowledge base is available
# (e.g., built using a library like 'rank_bm25' [cite: 319, 356])

def execute_phase_3(query: Query, bm25_index: Dict[str, ContextChunk]) -> List[ContextChunk]:
    """
    Phase 3: Fallback Retrieval (BM25).
    Retrieves additional documents based on keyword matching.
    """
    print("--- Phase 3: Fallback Retrieval (BM25) ---")
    
    # *Simulated BM25 Search*
    # In practice, this would perform a keyword-based search.
    # We simulate retrieving a fixed number of backup documents.
    
    query_terms = query.text.lower().split()
    print(f"Searching using query terms: {query_terms}")
    
    # Simulate retrieving a fixed number of fallback chunks
    fallback_candidates: List[ContextChunk] = []
    
    # Simulate finding 5 documents, assigning a neutral/low relevance score
    all_chunks = list(bm25_index.values())
    for i in range(1, 6): # Get 5 fallback docs
        if i < len(all_chunks):
            chunk = all_chunks[i]
            # Fallback documents are typically assigned a lower score
            chunk.relevance_score = 0.55 # Assign a score slightly above default tau
            fallback_candidates.append(chunk)

    print(f"Retrieved {len(fallback_candidates)} fallback chunks (C_fallback)[cite: 76].")
    return fallback_candidates