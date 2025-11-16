"""
Performance metrics and logging utilities for RAG comparison.
"""

import time
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from meve.core.models import ContextChunk


@dataclass
class RetrievalMetrics:
    """Metrics for a single retrieval operation."""

    query: str
    method: str  # "basic_rag" or "meve_rag"

    # Timing
    total_time: float = 0.0
    phase_times: Dict[str, float] = field(default_factory=dict)

    # Retrieval stats
    chunks_retrieved: int = 0
    chunks_verified: int = 0  # Only for MeVe
    chunks_fallback: int = 0  # Only for MeVe
    chunks_final: int = 0

    # Quality metrics
    avg_relevance_score: float = 0.0
    max_relevance_score: float = 0.0
    min_relevance_score: float = 0.0

    # Context size
    context_chars: int = 0
    context_tokens: int = 0
    token_budget: Optional[int] = None
    budget_efficiency: float = 0.0  # % of budget used

    # Retrieved chunks (for detailed logging)
    retrieved_chunks: List[ContextChunk] = field(default_factory=list)

    def log_summary(self, logger) -> None:
        """Log a structured summary of metrics."""
        logger.info(f"\n{'='*70}")
        logger.info(f"📊 RETRIEVAL METRICS - {self.method.upper()}")
        logger.info(f"{'='*70}")
        logger.info(f"Query: {self.query}")
        logger.info(f"\n⏱️  Timing:")
        logger.info(f"   Total Time: {self.total_time:.2f}s")
        if self.phase_times:
            for phase, time_taken in self.phase_times.items():
                logger.info(f"   └─ {phase}: {time_taken:.2f}s")

        logger.info(f"\n📈 Retrieval Stats:")
        logger.info(f"   Initial Retrieved: {self.chunks_retrieved}")
        if self.chunks_verified > 0:
            logger.info(f"   Verified (τ>0.5): {self.chunks_verified}")
        if self.chunks_fallback > 0:
            logger.info(f"   Fallback (BM25): {self.chunks_fallback}")
        logger.info(f"   Final Chunks: {self.chunks_final}")

        logger.info(f"\n🎯 Quality Metrics:")
        logger.info(f"   Avg Relevance: {self.avg_relevance_score:.3f}")
        logger.info(f"   Max Relevance: {self.max_relevance_score:.3f}")
        logger.info(f"   Min Relevance: {self.min_relevance_score:.3f}")

        logger.info(f"\n📝 Context Size:")
        logger.info(f"   Characters: {self.context_chars:,}")
        logger.info(f"   Tokens: {self.context_tokens}")
        if self.token_budget:
            logger.info(
                f"   Budget: {self.token_budget} (Efficiency: {self.budget_efficiency:.1f}%)"
            )

        logger.info(f"\n📄 Retrieved Chunks Preview:")
        for i, chunk in enumerate(self.retrieved_chunks[:3], 1):
            score = getattr(chunk, "relevance_score", 0.0)
            preview = chunk.content[:80].replace("\n", " ")
            logger.info(f"   {i}. [Score: {score:.3f}] {preview}...")

        if len(self.retrieved_chunks) > 3:
            logger.info(f"   ... and {len(self.retrieved_chunks) - 3} more")

        logger.info(f"{'='*70}\n")


@dataclass
class ComparisonMetrics:
    """Comparison metrics between Basic RAG and MeVe RAG."""

    query: str
    basic_metrics: RetrievalMetrics
    meve_metrics: RetrievalMetrics

    def log_comparison(self, logger) -> None:
        """Log side-by-side comparison."""
        logger.info(f"\n{'='*70}")
        logger.info(f"⚖️  COMPARISON: Basic RAG vs MeVe RAG")
        logger.info(f"{'='*70}")
        logger.info(f"Query: {self.query}\n")

        # Create comparison table
        logger.info(f"{'Metric':<30} {'Basic RAG':<20} {'MeVe RAG':<20}")
        logger.info(f"{'-'*70}")

        # Timing
        logger.info(
            f"{'Total Time':<30} {self.basic_metrics.total_time:<20.2f}s {self.meve_metrics.total_time:<20.2f}s"
        )

        # Retrieval
        logger.info(
            f"{'Initial Retrieved':<30} {self.basic_metrics.chunks_retrieved:<20} {self.meve_metrics.chunks_retrieved:<20}"
        )
        logger.info(
            f"{'Final Chunks':<30} {self.basic_metrics.chunks_final:<20} {self.meve_metrics.chunks_final:<20}"
        )

        # Quality
        logger.info(
            f"{'Avg Relevance Score':<30} {self.basic_metrics.avg_relevance_score:<20.3f} {self.meve_metrics.avg_relevance_score:<20.3f}"
        )

        # Size
        logger.info(
            f"{'Context Characters':<30} {self.basic_metrics.context_chars:<20,} {self.meve_metrics.context_chars:<20,}"
        )
        logger.info(
            f"{'Context Tokens':<30} {self.basic_metrics.context_tokens:<20} {self.meve_metrics.context_tokens:<20}"
        )

        if self.meve_metrics.token_budget:
            logger.info(
                f"{'Budget Efficiency':<30} {'N/A':<20} {self.meve_metrics.budget_efficiency:<20.1f}%"
            )

        logger.info(f"{'-'*70}")

        # Winner analysis
        logger.info(f"\n🏆 Analysis:")

        if self.meve_metrics.avg_relevance_score > self.basic_metrics.avg_relevance_score:
            logger.info(
                f"   ✅ MeVe has HIGHER average relevance (+{(self.meve_metrics.avg_relevance_score - self.basic_metrics.avg_relevance_score):.3f})"
            )
        else:
            logger.info(f"   ⚠️  Basic RAG has higher avg relevance")

        if self.meve_metrics.context_chars < self.basic_metrics.context_chars:
            reduction = (
                (self.basic_metrics.context_chars - self.meve_metrics.context_chars)
                / self.basic_metrics.context_chars
            ) * 100
            logger.info(f"   ✅ MeVe uses {reduction:.1f}% LESS context (more efficient)")
        else:
            logger.info(f"   ⚠️  MeVe uses more context")

        time_overhead = (
            (self.meve_metrics.total_time - self.basic_metrics.total_time)
            / self.basic_metrics.total_time
        ) * 100
        logger.info(f"   ⏱️  MeVe has {time_overhead:+.1f}% time overhead")

        logger.info(f"{'='*70}\n")


class PerformanceTracker:
    """Context manager for tracking retrieval performance."""

    def __init__(self, metrics: RetrievalMetrics, phase_name: Optional[str] = None):
        self.metrics = metrics
        self.phase_name = phase_name
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start_time

        if self.phase_name:
            self.metrics.phase_times[self.phase_name] = elapsed
        else:
            self.metrics.total_time = elapsed
