"""
Routers for system health and status endpoints.
Exposes GET /health for checking system readiness.
"""

from fastapi import APIRouter, HTTPException

from ..schemas import HealthResponse, HealthStatus, MeVeConfigSchema

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check(engine=None) -> HealthResponse:
    """
    Check system health and readiness.

    Returns vector DB, BM25 index availability and current configuration.
    """
    if engine is None:
        raise HTTPException(status_code=500, detail="Engine not initialized")

    try:
        # Check data sources
        vector_db_ready = (
            engine.vector_db_client is not None if hasattr(engine, "vector_db_client") else False
        )
        bm25_ready = engine.bm25_index is not None if hasattr(engine, "bm25_index") else False

        # Determine overall status
        if vector_db_ready and bm25_ready:
            status = HealthStatus.HEALTHY
            message = "All systems operational"
        elif vector_db_ready or bm25_ready:
            status = HealthStatus.DEGRADED
            message = "Partial data sources available"
        else:
            status = HealthStatus.UNHEALTHY
            message = "No data sources available"

        # Get current config
        config = engine.config if hasattr(engine, "config") else None
        if config:
            config_schema = MeVeConfigSchema(
                k_init=config.k_init,
                tau_relevance=config.tau_relevance,
                n_min=config.n_min,
                theta_redundancy=config.theta_redundancy,
                lambda_mmr=config.lambda_mmr,
                t_max=config.t_max,
            )
        else:
            config_schema = MeVeConfigSchema()

        return HealthResponse(
            status=status,
            vector_db_ready=vector_db_ready,
            bm25_index_ready=bm25_ready,
            current_config=config_schema,
            message=message,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def status(engine=None) -> dict:
    """
    Alias for /health with simplified response.
    """
    if engine is None:
        raise HTTPException(status_code=500, detail="Engine not initialized")

    try:
        vector_db_ready = (
            engine.vector_db_client is not None if hasattr(engine, "vector_db_client") else False
        )
        bm25_ready = engine.bm25_index is not None if hasattr(engine, "bm25_index") else False

        return {
            "status": "healthy" if (vector_db_ready and bm25_ready) else "degraded",
            "vector_db_ready": vector_db_ready,
            "bm25_index_ready": bm25_ready,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
