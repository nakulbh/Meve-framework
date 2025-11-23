"""
Routers for core retrieval operations.
Exposes POST /retrieve for query submission and metrics retrieval.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
import time

from meve import MeVeConfig
from ..schemas import (
    QueryRequest,
    RetrievalResponse,
    RetrievalMetrics,
    ContextChunkSchema,
    ErrorResponse,
)
from ..request_handler import RequestContext

router = APIRouter(prefix="/retrieve", tags=["retrieval"])


@router.post("", response_model=RetrievalResponse)
async def retrieve_context(
    request: QueryRequest,
    request_handler=None,  # Injected via dependency
) -> RetrievalResponse:
    """
    Execute full MeVe retrieval pipeline for a query.

    Returns context chunks ranked by relevance with phase-by-phase metrics.
    """
    if request_handler is None:
        raise HTTPException(status_code=500, detail="Request handler not initialized")

    try:
        # Create request context for tracing
        ctx = RequestContext()

        # Convert config if provided
        config = None
        if request.config:
            config = MeVeConfig(
                k_init=request.config.k_init,
                tau_relevance=request.config.tau_relevance,
                n_min=request.config.n_min,
                theta_redundancy=request.config.theta_redundancy,
                lambda_mmr=request.config.lambda_mmr,
                t_max=request.config.t_max,
            )

        # Execute query
        result = request_handler.execute_query(
            query=request.query,
            config=config,
            request_context=ctx,
        )

        # Format chunks
        chunks = [
            ContextChunkSchema(
                content=chunk["content"],
                doc_id=chunk["doc_id"],
                relevance_score=chunk["relevance_score"],
                token_count=chunk["token_count"],
            )
            for chunk in result["chunks"]
        ]

        # Format metrics
        metrics = RetrievalMetrics(
            phase_1_candidates=result["metrics"]["phase_1_candidates"],
            phase_2_verified=result["metrics"]["phase_2_verified"],
            phase_3_triggered=result["metrics"]["phase_3_triggered"],
            phase_4_deduplicated=result["metrics"]["phase_4_deduplicated"],
            phase_5_final_tokens=result["metrics"]["phase_5_final_tokens"],
            total_chunks_selected=result["metrics"]["total_chunks_selected"],
        )

        return RetrievalResponse(
            status="success",
            request_id=result["request_id"],
            query=result["query"],
            context=result["context"],
            chunks=chunks,
            metrics=metrics,
            processing_time_ms=result["processing_time_ms"],
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/metrics/{request_id}")
async def get_metrics(
    request_id: str,
    request_handler=None,  # Injected via dependency
) -> dict:
    """
    Retrieve stored metrics for a previous request.

    Useful for analyzing phase-by-phase behavior.
    """
    if request_handler is None:
        raise HTTPException(status_code=500, detail="Request handler not initialized")

    try:
        result = request_handler.get_request_metrics(request_id)
        return {
            "status": "success",
            "request_id": result["request_id"],
            "query": result["query"],
            "metrics": result["metrics"],
            "processing_time_ms": result["processing_time_ms"],
            "timestamp": result["timestamp"],
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
