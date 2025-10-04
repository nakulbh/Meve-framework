# phase_4_prioritization.py

from meve_data import ContextChunk, Query, MeVeConfig
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from color_utils import phase_header, success_message

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

def calculate_mmr_score(candidate_chunk: ContextChunk, query: Query, selected_chunks: List[ContextChunk], 
                        lambda_param: float = 0.5) -> float:
    """
    Calculate Maximal Marginal Relevance (MMR) score for enhanced diversity.
    MMR = λ * Relevance(chunk, query) - (1-λ) * max_similarity(chunk, selected_chunks)
    """
    relevance_score = candidate_chunk.relevance_score
    
    if not selected_chunks:
        return relevance_score
    
    # Calculate maximum similarity with already selected chunks
    max_similarity = 0.0
    for selected_chunk in selected_chunks:
        similarity = cosine_similarity(candidate_chunk.embedding, selected_chunk.embedding)
        max_similarity = max(max_similarity, similarity)
    
    # MMR formula: balance relevance and diversity
    mmr_score = lambda_param * relevance_score - (1 - lambda_param) * max_similarity
    return mmr_score

def calculate_information_overlap(chunk1: ContextChunk, chunk2: ContextChunk) -> float:
    """
    Calculate semantic information overlap between two chunks using multiple metrics.
    Combines cosine similarity with content-based overlap analysis.
    """
    # 1. Semantic similarity (embedding-based)
    semantic_sim = cosine_similarity(chunk1.embedding, chunk2.embedding)
    
    # 2. Content overlap (token-based)
    tokens1 = set(chunk1.content.lower().split())
    tokens2 = set(chunk2.content.lower().split())
    
    if not tokens1 or not tokens2:
        content_overlap = 0.0
    else:
        intersection = len(tokens1.intersection(tokens2))
        union = len(tokens1.union(tokens2))
        content_overlap = intersection / union if union > 0 else 0.0
    
    # 3. Combined overlap score (weighted average)
    overlap_score = 0.7 * semantic_sim + 0.3 * content_overlap
    return overlap_score

def execute_phase_4(query: Query, combined_context: List[ContextChunk], config: MeVeConfig) -> List[ContextChunk]:
    """
    Phase 4: Enhanced Context Prioritization with MMR and Advanced Redundancy Detection.
    Implements sophisticated relevance-diversity tradeoff as per MeVe paper.
    """
    print(f"{phase_header(4, 'STARTING')} - Enhanced Context Prioritization (θ_redundancy={config.theta_redundancy})")
    print(f"[PHASE 4] Received {len(combined_context)} chunks from previous phases")
    
    if not combined_context:
        print("[PHASE 4] WARNING: No context chunks to prioritize")
        print("[PHASE 4] COMPLETED - Returning empty list")
        return []
    
    # Get sentence transformer model
    print("[PHASE 4] Loading sentence transformer for embedding generation...")
    model = get_sentence_transformer()
    
    # 1. Generate embeddings for chunks that don't have them
    chunks_without_embeddings = [c for c in combined_context if not c.embedding]
    if chunks_without_embeddings:
        print(f"[PHASE 4] Generating embeddings for {len(chunks_without_embeddings)} chunks...")
        for chunk in chunks_without_embeddings:
            chunk.embedding = model.encode(chunk.content).tolist()
    else:
        print("[PHASE 4] All chunks already have embeddings")
    
    # 2. Generate query embedding if not present
    if not query.vector:
        print("[PHASE 4] Generating query embedding...")
        query.vector = model.encode(query.text).tolist()
    else:
        print("[PHASE 4] Using existing query embedding")
    
    # 3. Enhanced selection using MMR-style algorithm
    remaining_chunks = combined_context.copy()
    prioritized_context: List[ContextChunk] = []
    
    print(f"[PHASE 4] Starting MMR-based selection with λ=0.6...")
    # Start with highest relevance chunk
    remaining_chunks.sort(key=lambda c: c.relevance_score, reverse=True)
    print(f"[PHASE 4] Initial ranking by relevance: {[f'{c.relevance_score:.3f}' for c in remaining_chunks[:5]]}...")
    
    iteration = 0
    while remaining_chunks:
        iteration += 1
        best_chunk = None
        best_score = float('-inf')
        
        # Find chunk with best MMR score
        for candidate in remaining_chunks:
            mmr_score = calculate_mmr_score(candidate, query, prioritized_context, lambda_param=0.6)
            
            if mmr_score > best_score:
                best_score = mmr_score
                best_chunk = candidate
        
        if best_chunk is None:
            print("[PHASE 4] No more candidates available")
            break
        
        print(f"[PHASE 4] Iteration {iteration}: Evaluating chunk {best_chunk.doc_id} (MMR={best_score:.3f})")
        
        # Check for redundancy using enhanced overlap detection
        is_redundant = False
        for selected_chunk in prioritized_context:
            overlap = calculate_information_overlap(best_chunk, selected_chunk)
            
            if overlap >= config.theta_redundancy:
                print(f"[PHASE 4]   → REDUNDANT: overlap={overlap:.3f} >= {config.theta_redundancy} with {selected_chunk.doc_id}")
                is_redundant = True
                break
        
        # Add to selected if not redundant
        if not is_redundant:
            prioritized_context.append(best_chunk)
            print(f"[PHASE 4]   → SELECTED: MMR={best_score:.3f}, relevance={best_chunk.relevance_score:.3f}")
            print(f"[PHASE 4]     Content: '{best_chunk.content[:50]}...'")
        
        # Remove from remaining candidates
        remaining_chunks.remove(best_chunk)
        
        # Stop if we have enough diverse chunks (heuristic)
        if len(prioritized_context) >= min(10, len(combined_context)):
            print(f"[PHASE 4] Reached maximum chunks limit ({min(10, len(combined_context))})")
            break
    
    removed_count = len(combined_context) - len(prioritized_context)
    print(f"{success_message('[PHASE 4] COMPLETED')} - Selected {len(prioritized_context)} chunks from {len(combined_context)} candidates")
    print(f"[PHASE 4] Removed {removed_count} redundant/low-diversity chunks")
    if prioritized_context:
        avg_relevance = np.mean([c.relevance_score for c in prioritized_context])
        print(f"[PHASE 4] Average relevance score: {avg_relevance:.3f}")
        print(f"[PHASE 4] Best relevance score: {max(c.relevance_score for c in prioritized_context):.3f}")
    print(f"{phase_header(4, 'HANDING OFF')} to Phase 5 (Budgeting)\n")
    
    return prioritized_context