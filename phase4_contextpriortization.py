# phase_4_prioritization.py

from meve_data import ContextChunk, Query, MeVeConfig
from typing import List
import numpy as np # Used for embedding operations

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Helper to calculate cosine similarity between two vectors."""
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)

def execute_phase_4(query: Query, combined_context: List[ContextChunk], config: MeVeConfig) -> List[ContextChunk]:
    """
    Phase 4: Context Prioritization (Relevance/Redundancy).
    Sorts by relevance and filters out redundant information[cite: 78, 101, 102].
    """
    print(f"--- Phase 4: Context Prioritization (Redundancy={config.theta_redundancy}) ---")
    
    # 1. Sort by relevance score (descending) [cite: 101]
    sorted_chunks = sorted(combined_context, key=lambda c: c.relevance_score, reverse=True)
    
    prioritized_context: List[ContextChunk] = []
    
    # 2. Iterative filtering to remove redundancy [cite: 102]
    for current_chunk in sorted_chunks:
        is_redundant = False
        if not current_chunk.embedding:
             # Simulate embedding calculation if missing
            current_chunk.embedding = np.random.rand(768).tolist()
            
        for kept_chunk in prioritized_context:
            sim = cosine_similarity(current_chunk.embedding, kept_chunk.embedding)
            
            # Discard if highly similar to a higher-ranked (already kept) document [cite: 102]
            if sim >= config.theta_redundancy:
                is_redundant = True
                break
        
        if not is_redundant:
            prioritized_context.append(current_chunk)
            
    print(f"Prioritized context size: {len(prioritized_context)} (Removed {len(sorted_chunks) - len(prioritized_context)} redundant chunks).")
    return prioritized_context