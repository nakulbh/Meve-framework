# meve/phases/phase2_verification.py

from typing import List

import torch

from meve.core.models import ContextChunk, MeVeConfig, Query
from meve.utils import get_logger, get_cross_encoder

logger = get_logger(__name__)


def _get_cross_encoder():
    """Get the cross-encoder model from the model manager."""
    return get_cross_encoder()


def get_relevance_score(query_text: str, chunk_content: str) -> float:
    """Get actual relevance score using cross-encoder model."""
    try:
        # Get the cross-encoder (lazy loading)
        encoder = _get_cross_encoder()

        # Cross-encoders take query-document pairs and output relevance scores
        score = encoder.predict([(query_text, chunk_content)])

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
        logger.error(f"Error in cross-encoder scoring: {e}", error=e)
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
    Uses batch prediction for improved performance.
    """
    if not initial_candidates:
        logger.warning("Phase 2: No initial candidates to verify.")
        return []

    verified_chunks: List[ContextChunk] = []

    # Prepare all query-document pairs for batch processing
    pairs = [(query.text, chunk.content) for chunk in initial_candidates]

    # Batch predict all pairs at once for better performance
    try:
        encoder = _get_cross_encoder()
        scores = encoder.predict(pairs)

        # Handle different return types
        if isinstance(scores, torch.Tensor):
            scores = scores.cpu().numpy()
        elif not isinstance(scores, (list, tuple)):
            scores = [scores]

        # Apply sigmoid normalization and filter by threshold
        for chunk, score in zip(initial_candidates, scores):
            # Normalize score to [0,1] range
            if isinstance(score, torch.Tensor):
                score = score.item()
            else:
                score = float(score)

            normalized_score = torch.sigmoid(torch.tensor(score)).item()
            chunk.relevance_score = normalized_score

            if normalized_score >= config.tau_relevance:
                verified_chunks.append(chunk)

    except Exception as e:
        logger.error(f"Batch cross-encoder scoring failed: {e}", error=e)
        # Fallback to individual scoring
        for chunk in initial_candidates:
            score = get_relevance_score(query.text, chunk.content)
            chunk.relevance_score = score

            if score >= config.tau_relevance:
                verified_chunks.append(chunk)

    return verified_chunks
