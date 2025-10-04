# phase_2_verification.py

from meve_data import ContextChunk, Query, MeVeConfig
from typing import List
from sentence_transformers import CrossEncoder
import torch
from color_utils import phase_header, success_message

# Load cross-encoder model for relevance scoring
print("[PHASE 2] Loading cross-encoder model: 'cross-encoder/ms-marco-MiniLM-L-6-v2'")
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
print("[PHASE 2] Cross-encoder model loaded successfully")

def get_relevance_score(query_text: str, chunk_content: str) -> float:
    """Get actual relevance score using cross-encoder model."""
    try:
        # Cross-encoders take query-document pairs and output relevance scores
        score = cross_encoder.predict([(query_text, chunk_content)])
        
        # Handle different return types
        if isinstance(score, (list, tuple)):
            score = float(score[0])
        elif isinstance(score, torch.Tensor):
            score = score.item()
        else:
            score = float(score)
        
        # Apply sigmoid to normalize to [0,1] range if needed
        normalized_score = torch.sigmoid(torch.tensor(score)).item()
        return normalized_score
        
    except Exception as e:
        print(f"[PHASE 2] Error in cross-encoder scoring: {e}")
        # Fallback to simple similarity check
        return simulate_cross_encoder_fallback(query_text, chunk_content)

def simulate_cross_encoder_fallback(query_text: str, chunk_content: str) -> float:
    """Fallback simulation for cross-encoder scoring."""
    # Simple keyword matching as fallback
    query_words = set(query_text.lower().split())
    content_words = set(chunk_content.lower().split())
    overlap = len(query_words.intersection(content_words))
    return min(overlap / len(query_words) if query_words else 0.0, 1.0) 

def execute_phase_2(query: Query, initial_candidates: List[ContextChunk], config: MeVeConfig) -> List[ContextChunk]:
    """
    Phase 2: Relevance Verification (Cross-Encoder).
    Filters candidates based on the relevance threshold (tau)[cite: 75, 90].
    """
    print(f"{phase_header(2, 'STARTING')} - Relevance Verification (Tau={config.tau_relevance})")
    print(f"[PHASE 2] Received {len(initial_candidates)} candidates from Phase 1")
    
    if not initial_candidates:
        print("[PHASE 2] WARNING: No initial candidates to verify")
        print("[PHASE 2] COMPLETED - Returning empty list")
        return []
    
    verified_chunks: List[ContextChunk] = []
    print(f"[PHASE 2] Initializing cross-encoder for relevance scoring...")

    # Process each candidate through the cross-encoder
    print(f"[PHASE 2] Processing candidates through cross-encoder:")
    for i, chunk in enumerate(initial_candidates, 1):
        print(f"[PHASE 2] Processing candidate {i}/{len(initial_candidates)}: {chunk.doc_id}")
        score = get_relevance_score(query.text, chunk.content)
        chunk.relevance_score = score
        
        print(f"[PHASE 2] Chunk {chunk.doc_id}: relevance_score={score:.3f}")
        
        if score >= config.tau_relevance:
            verified_chunks.append(chunk)
            print(f"[PHASE 2]   → VERIFIED (score >= {config.tau_relevance})")
        else:
            print(f"[PHASE 2]   → FILTERED OUT (score < {config.tau_relevance})")
    
    print(f"{success_message('[PHASE 2] COMPLETED')} - Verified {len(verified_chunks)} out of {len(initial_candidates)} chunks")
    if verified_chunks:
        scores = [chunk.relevance_score for chunk in verified_chunks]
        print(f"[PHASE 2] Best relevance score: {max(scores):.3f}")
        print(f"[PHASE 2] Average relevance score: {sum(scores)/len(scores):.3f}")
    
    # Determine next phase
    if len(verified_chunks) >= config.n_min:
        print(f"[PHASE 2] SUCCESS: {len(verified_chunks)} >= {config.n_min} (n_min) - Skipping fallback")
        print(f"{phase_header(2, 'HANDING OFF')} to Phase 4 (Prioritization)\n")
    else:
        print(f"[PHASE 2] INSUFFICIENT: {len(verified_chunks)} < {config.n_min} (n_min) - Triggering fallback")
        print(f"{phase_header(2, 'HANDING OFF')} to Phase 3 (Fallback)\n")
    
    return verified_chunks