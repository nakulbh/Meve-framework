# meve/phases/phase3_fallback.py

import math
from collections import Counter, defaultdict
from typing import Dict, List

from meve.core.models import ContextChunk, Query
from meve.utils import get_logger

logger = get_logger(__name__)

# Assume a BM25 index of the knowledge base is available
# (e.g., built using a library like 'rank_bm25' [cite: 319, 356])


def calculate_bm25_okapi(
    query_terms: List[str], document_terms: List[str], corpus_stats: Dict, k1=1.2, b=0.75
) -> float:
    """
    Calculate BM25 Okapi score for a document given query terms.

    Args:
        query_terms: List of query terms
        document_terms: List of terms in the document
        corpus_stats: Dictionary containing 'doc_count', 'avg_doc_length', 'term_doc_freq'
        k1: Term frequency saturation parameter (default 1.2)
        b: Document length normalization parameter (default 0.75)
    """
    score = 0.0
    doc_length = len(document_terms)
    avg_doc_length = corpus_stats["avg_doc_length"]
    total_docs = corpus_stats["doc_count"]

    # Count term frequencies in document
    doc_term_freq = Counter(document_terms)

    for term in query_terms:
        if term in doc_term_freq:
            # Term frequency in document
            tf = doc_term_freq[term]

            # Document frequency (number of docs containing term)
            df = corpus_stats["term_doc_freq"].get(term, 0)

            if df > 0:
                # IDF calculation
                idf = math.log((total_docs - df + 0.5) / (df + 0.5))

                # BM25 formula
                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * (doc_length / avg_doc_length))

                score += idf * (numerator / denominator)

    return score


def build_corpus_stats(all_chunks: List[ContextChunk]) -> Dict:
    """Build corpus statistics needed for BM25 calculation."""
    total_docs = len(all_chunks)
    total_length = 0
    term_doc_freq = defaultdict(int)

    for chunk in all_chunks:
        terms = chunk.content.lower().split()
        total_length += len(terms)

        # Count unique terms in this document
        unique_terms = set(terms)
        for term in unique_terms:
            term_doc_freq[term] += 1

    avg_doc_length = total_length / total_docs if total_docs > 0 else 0

    return {
        "doc_count": total_docs,
        "avg_doc_length": avg_doc_length,
        "term_doc_freq": dict(term_doc_freq),
    }


def execute_phase_3(query: Query, bm25_index: Dict[str, ContextChunk]) -> List[ContextChunk]:
    """
    Phase 3: Fallback Retrieval (BM25 Okapi).
    Retrieves additional documents based on BM25 Okapi scoring.
    """
    logger.info("--- Phase 3: Fallback Retrieval (BM25 Okapi) ---")

    query_terms = query.text.lower().split()
    logger.debug(f"Searching using query terms: {query_terms}")

    # Get all chunks and build corpus statistics
    all_chunks = list(bm25_index.values())
    corpus_stats = build_corpus_stats(all_chunks)

    # Calculate BM25 scores for each chunk
    scored_chunks = []

    for chunk in all_chunks:
        document_terms = chunk.content.lower().split()

        # Calculate BM25 Okapi score
        bm25_score = calculate_bm25_okapi(query_terms, document_terms, corpus_stats)

        if bm25_score > 0:  # Only include chunks with positive BM25 score
            # Normalize score to 0-0.6 range for fallback (keeping it lower than verified chunks)
            normalized_score = min(0.6, bm25_score / 10.0)  # Scale down for reasonable range
            chunk.relevance_score = normalized_score
            scored_chunks.append(chunk)

    # Sort by BM25 score and take top 5
    scored_chunks.sort(key=lambda x: x.relevance_score, reverse=True)
    fallback_candidates = scored_chunks[:5]

    logger.info(f"Retrieved {len(fallback_candidates)} fallback chunks using BM25 Okapi.")
    for i, chunk in enumerate(fallback_candidates):
        logger.debug(
            f"  Fallback {i+1}: BM25 Score={chunk.relevance_score:.3f}, Content: {chunk.content[:50]}..."
        )

    return fallback_candidates
