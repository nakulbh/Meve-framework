# meve/phases/phase2_verification.py

from typing import List

import torch
from sentence_transformers import CrossEncoder

from meve.core.models import ContextChunk, MeVeConfig, Query

# Load cross-encoder model for relevance scoring
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


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
        print(f"Error in cross-encoder scoring: {e}")
        # Fallback to simple similarity check
        return simulate_cross_encoder_fallback(query_text, chunk_content)


def simulate_cross_encoder_fallback(query_text: str, chunk_content: str) -> float:
    """Fallback simulation for cross-encoder scoring."""
    # Simple keyword matching as fallback
    query_words = set(query_text.lower().split())
    content_words = set(chunk_content.lower().split())
    overlap = len(query_words.intersection(content_words))
    return min(overlap / len(query_words) if query_words else 0.0, 1.0)


def execute_phase_2(
    query: Query, initial_candidates: List[ContextChunk], config: MeVeConfig
) -> List[ContextChunk]:
    """
    Phase 2: Relevance Verification (Cross-Encoder).
    Filters candidates based on the relevance threshold (tau)[cite: 75, 90].
    """
    print(f"--- Phase 2: Relevance Verification (Tau={config.tau_relevance}) ---")

    if not initial_candidates:
        print("No initial candidates to verify.")
        return []

    verified_chunks: List[ContextChunk] = []

    # Process each candidate through the cross-encoder
    for chunk in initial_candidates:
        score = get_relevance_score(query.text, chunk.content)
        chunk.relevance_score = score

        print(f"Chunk {chunk.doc_id}: relevance_score={score:.3f}")

        if score >= config.tau_relevance:
            verified_chunks.append(chunk)
            print(f"  → VERIFIED (score >= {config.tau_relevance})")
        else:
            print(f"  → FILTERED OUT (score < {config.tau_relevance})")

    print(f"Verified {len(verified_chunks)} out of {len(initial_candidates)} chunks (C_ver).")
    return verified_chunks
