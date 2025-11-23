"""
Request handler utilities for processing queries and managing state.
"""

import uuid
import time
from datetime import datetime
from typing import Optional, Dict, Any
from contextlib import contextmanager

from meve import MeVeEngine, MeVeConfig, ContextChunk
from meve.utils import get_logger

logger = get_logger(__name__)


class RequestContext:
    """Manages request-scoped state and metrics."""

    def __init__(self, request_id: Optional[str] = None):
        self.request_id = request_id or str(uuid.uuid4())
        self.query: Optional[str] = None
        self.start_time: float = time.time()
        self.end_time: Optional[float] = None
        self.metrics: Dict[str, Any] = {}
        self.timestamp: str = datetime.utcnow().isoformat()

    @property
    def processing_time_ms(self) -> float:
        """Calculate processing time in milliseconds."""
        end = self.end_time or time.time()
        return (end - self.start_time) * 1000

    def record_metric(self, key: str, value: Any) -> None:
        """Record a metric for this request."""
        self.metrics[key] = value

    def finalize(self) -> None:
        """Mark request as complete."""
        self.end_time = time.time()


class RequestHandler:
    """Handles query execution and result formatting."""

    def __init__(self, engine: MeVeEngine):
        self.engine = engine
        self.request_history: Dict[str, RequestContext] = {}

    def execute_query(
        self,
        query: str,
        config: Optional[MeVeConfig] = None,
        request_context: Optional[RequestContext] = None,
    ) -> Dict[str, Any]:
        """
        Execute query through MeVeEngine and gather metrics.

        Args:
            query: User query string
            config: Optional MeVeConfig override
            request_context: Optional pre-created context for tracing

        Returns:
            Dictionary with context, chunks, and metrics
        """
        ctx = request_context or RequestContext()
        ctx.query = query

        try:
            logger.info(
                f"Executing query", metadata={"request_id": ctx.request_id, "query": query[:50]}
            )

            # Run the engine
            context_str = self.engine.run(query)

            # Check for error returns
            if isinstance(context_str, str) and context_str.startswith("Error:"):
                logger.error(
                    "Engine returned error",
                    error=ValueError(context_str),
                    metadata={"request_id": ctx.request_id, "error": context_str},
                )
                raise ValueError(context_str)

            # Extract chunks from engine state
            chunks = self._format_chunks(self.engine.last_retrieved_chunks)

            # Calculate metrics
            metrics = self._extract_metrics()

            ctx.record_metric("phase_1_candidates", metrics.get("phase_1_candidates", 0))
            ctx.record_metric("phase_2_verified", metrics.get("phase_2_verified", 0))
            ctx.record_metric("phase_3_triggered", metrics.get("phase_3_triggered", False))
            ctx.record_metric("phase_4_deduplicated", metrics.get("phase_4_deduplicated", 0))
            ctx.record_metric("phase_5_final_tokens", metrics.get("phase_5_final_tokens", 0))
            ctx.finalize()

            # Store for later metrics retrieval
            self.request_history[ctx.request_id] = ctx

            logger.info(
                f"Query executed successfully",
                metadata={"request_id": ctx.request_id, "time_ms": ctx.processing_time_ms},
            )

            return {
                "request_id": ctx.request_id,
                "query": query,
                "context": context_str,
                "chunks": chunks,
                "metrics": metrics,
                "processing_time_ms": ctx.processing_time_ms,
                "timestamp": ctx.timestamp,
            }

        except Exception as e:
            ctx.finalize()
            logger.error("Query execution failed", error=e, metadata={"request_id": ctx.request_id})
            raise

    def _format_chunks(self, chunks_list) -> list:
        """Convert engine chunks to response format."""
        if not chunks_list:
            return []

        # Handle both list and dict formats
        chunks_to_format = chunks_list if isinstance(chunks_list, list) else chunks_list.values()

        return [
            {
                "content": chunk.content,
                "doc_id": chunk.doc_id,
                "relevance_score": chunk.relevance_score,
                "token_count": chunk.token_count,
            }
            for chunk in chunks_to_format
        ]

    def _extract_metrics(self) -> Dict[str, Any]:
        """Extract phase-by-phase metrics from last retrieval."""
        # Start with defaults
        metrics = {
            "phase_1_candidates": 0,
            "phase_2_verified": 0,
            "phase_3_triggered": False,
            "phase_4_deduplicated": 0,
            "phase_5_final_tokens": 0,
            "total_chunks_selected": 0,
        }

        if not self.engine.last_retrieved_chunks:
            return metrics

        # Count final chunks
        final_count = len(self.engine.last_retrieved_chunks)
        metrics["total_chunks_selected"] = final_count
        metrics["phase_5_final_tokens"] = sum(
            chunk.token_count or 0 for chunk in self.engine.last_retrieved_chunks
        )

        # Phase metrics are estimates based on final counts
        # In a future version, we can instrument each phase to track counts
        metrics["phase_1_candidates"] = final_count * 2  # Estimate: Phase 1 returns ~2x final
        metrics["phase_2_verified"] = final_count  # Phase 2 passes through
        metrics["phase_4_deduplicated"] = final_count  # Phase 4 might reduce slightly

        return metrics

    def get_request_metrics(self, request_id: str) -> Dict[str, Any]:
        """Retrieve stored metrics for a request."""
        if request_id not in self.request_history:
            raise ValueError(f"Request {request_id} not found in history")

        ctx = self.request_history[request_id]
        return {
            "request_id": request_id,
            "query": ctx.query,
            "metrics": ctx.metrics,
            "processing_time_ms": ctx.processing_time_ms,
            "timestamp": ctx.timestamp,
        }
