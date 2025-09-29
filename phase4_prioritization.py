# phase_4_prioritization.py

from meve_data import ContextChunk, Query, MeVeConfig
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

# Initialize sentence transformer model (same as Phase 1)
_model = None

def get_sentence_transformer():
    """Get or initialize the sentence transformer model."""
    global _model
    if _model is None:
        _model = SentenceTransformer('all-MiniLM-L6-v2')
    return _model

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)

def execute_phase_4(query: Query, combined_context: List[ContextChunk], config: MeVeConfig) -> List[ContextChunk]:
    """
    Phase 4: Context Prioritization (Relevance/Redundancy).
    Sorts by relevance and filters out redundant information as per MeVe paper.
    """
    print(f"--- Phase 4: Context Prioritization (Redundancy={config.theta_redundancy}) ---")
    
    if not combined_context:
        print("No context chunks to prioritize.")
        return []
    
    # Get sentence transformer model
    model = get_sentence_transformer()
    
    # 1. Sort by relevance score (descending) - highest relevance first
    sorted_chunks = sorted(combined_context, key=lambda c: c.relevance_score, reverse=True)
    
    prioritized_context: List[ContextChunk] = []
    
    # 2. Generate embeddings for chunks that don't have them
    for chunk in sorted_chunks:
        if not chunk.embedding:
            chunk.embedding = model.encode(chunk.content).tolist()
    
    # 3. Iterative filtering to remove redundancy (as per MeVe paper)
    for current_chunk in sorted_chunks:
        is_redundant = False
        
        # Check similarity with already selected chunks
        for kept_chunk in prioritized_context:
            similarity = cosine_similarity(current_chunk.embedding, kept_chunk.embedding)
            
            # If similarity exceeds threshold, consider it redundant
            if similarity >= config.theta_redundancy:
                print(f"  Removing redundant chunk (similarity={similarity:.3f} >= {config.theta_redundancy})")
                is_redundant = True
                break
        
        # Only add if not redundant
        if not is_redundant:
            prioritized_context.append(current_chunk)
            print(f"  Keeping chunk: relevance={current_chunk.relevance_score:.3f}, content='{current_chunk.content[:50]}...'")
    
    removed_count = len(sorted_chunks) - len(prioritized_context)
    print(f"Prioritized context size: {len(prioritized_context)} (Removed {removed_count} redundant chunks).")
    
    return prioritized_context