"""
Main FastAPI application for MeVe REST API.

Exposes 5-phase RAG retrieval pipeline with configuration management,
health monitoring, and detailed metrics.

Usage:
    uvicorn integrations.api.app:app --reload

Or programmatically:
    from integrations.api import create_app
    app = create_app(engine)
"""

import os
from typing import Optional
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from meve import MeVeEngine, MeVeConfig, ContextChunk
from meve.utils import get_logger
from .request_handler import RequestHandler, RequestContext
from .schemas import (
    QueryRequest,
    RetrievalResponse,
    RetrievalMetrics,
    ContextChunkSchema,
    ConfigUpdateRequest,
    MeVeConfigSchema,
    HealthResponse,
    HealthStatus,
    DataSourceLoadRequest,
    DataSourceUploadRequest,
)

logger = get_logger(__name__)

# Global state
_engine: Optional[MeVeEngine] = None
_request_handler: Optional[RequestHandler] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage app lifecycle: startup and shutdown.
    """
    logger.info("Starting MeVe REST API")
    yield
    logger.info("Shutting down MeVe REST API")


def create_app(
    engine: Optional[MeVeEngine] = None,
    title: str = "MeVe REST API",
    description: str = "5-phase RAG retrieval pipeline with configuration tuning and metrics",
) -> FastAPI:
    """
    Create FastAPI application with MeVeEngine.

    Args:
        engine: Optional pre-initialized MeVeEngine. If None, uses global singleton.
        title: API title for documentation
        description: API description for documentation

    Returns:
        Configured FastAPI application
    """
    global _engine, _request_handler

    app = FastAPI(
        title=title,
        description=description,
        version="0.1.0",
        lifespan=lifespan,
    )

    # Setup CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Initialize engine and handler
    _engine = engine
    if _engine:
        _request_handler = RequestHandler(_engine)
        logger.info("MeVeEngine initialized", metadata={"config": str(_engine.config)})

    # Dependency injection
    def get_engine():
        if _engine is None:
            raise HTTPException(status_code=500, detail="Engine not initialized")
        return _engine

    def get_request_handler():
        if _request_handler is None:
            raise HTTPException(status_code=500, detail="Request handler not initialized")
        return _request_handler

    # Root endpoint
    @app.get("/")
    async def root():
        """API documentation and overview."""
        return {
            "title": title,
            "description": description,
            "version": "0.1.0",
            "docs_url": "/docs",
            "openapi_url": "/openapi.json",
            "endpoints": {
                "health": "/health - Check system status",
                "retrieve": "POST /retrieve - Execute retrieval pipeline",
                "metrics": "/retrieve/metrics/{request_id} - Get request metrics",
                "config": "GET /config - Get current configuration",
                "config_update": "POST /config - Update configuration",
                "data_sources": "POST /data-sources/load - Load data source",
            },
        }

    # ========================
    # Health & Status Endpoints
    # ========================

    @app.get("/health", response_model=HealthResponse)
    async def health_check(engine=Depends(get_engine)):
        """Check system health and readiness."""
        try:
            vector_db_ready = (
                engine.vector_db_client is not None
                if hasattr(engine, "vector_db_client")
                else False
            )
            bm25_ready = engine.bm25_index is not None if hasattr(engine, "bm25_index") else False

            if vector_db_ready and bm25_ready:
                status = HealthStatus.HEALTHY
                message = "All systems operational"
            elif vector_db_ready or bm25_ready:
                status = HealthStatus.DEGRADED
                message = "Partial data sources available"
            else:
                status = HealthStatus.UNHEALTHY
                message = "No data sources available"

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

    @app.get("/status")
    async def status(engine=Depends(get_engine)):
        """Simplified status check."""
        try:
            vector_db_ready = (
                engine.vector_db_client is not None
                if hasattr(engine, "vector_db_client")
                else False
            )
            bm25_ready = engine.bm25_index is not None if hasattr(engine, "bm25_index") else False

            return {
                "status": "healthy" if (vector_db_ready and bm25_ready) else "degraded",
                "vector_db_ready": vector_db_ready,
                "bm25_index_ready": bm25_ready,
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # ========================
    # Retrieval Endpoints
    # ========================

    @app.post("/retrieve", response_model=RetrievalResponse)
    async def retrieve_context(
        request: QueryRequest,
        request_handler=Depends(get_request_handler),
    ) -> RetrievalResponse:
        """Execute full MeVe retrieval pipeline for a query."""
        try:
            ctx = RequestContext()
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

            result = request_handler.execute_query(
                query=request.query,
                config=config,
                request_context=ctx,
            )

            chunks = [
                ContextChunkSchema(
                    content=chunk["content"],
                    doc_id=chunk["doc_id"],
                    relevance_score=chunk["relevance_score"],
                    token_count=chunk["token_count"],
                )
                for chunk in result["chunks"]
            ]

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

    @app.get("/retrieve/metrics/{request_id}")
    async def get_metrics(
        request_id: str,
        request_handler=Depends(get_request_handler),
    ) -> dict:
        """Retrieve stored metrics for a previous request."""
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

    # ========================
    # Configuration Endpoints
    # ========================

    @app.get("/config", response_model=MeVeConfigSchema)
    async def get_config(engine=Depends(get_engine)):
        """Get current pipeline configuration."""
        try:
            config = engine.config if hasattr(engine, "config") else MeVeConfig()
            return MeVeConfigSchema(
                k_init=config.k_init,
                tau_relevance=config.tau_relevance,
                n_min=config.n_min,
                theta_redundancy=config.theta_redundancy,
                lambda_mmr=config.lambda_mmr,
                t_max=config.t_max,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/config")
    async def update_config(
        request: ConfigUpdateRequest,
        engine=Depends(get_engine),
    ) -> dict:
        """Update pipeline configuration parameters."""
        try:
            new_config = MeVeConfig(
                k_init=request.config.k_init,
                tau_relevance=request.config.tau_relevance,
                n_min=request.config.n_min,
                theta_redundancy=request.config.theta_redundancy,
                lambda_mmr=request.config.lambda_mmr,
                t_max=request.config.t_max,
            )

            if hasattr(engine, "config"):
                engine.config = new_config
            else:
                raise ValueError("Engine configuration update not supported")

            return {
                "status": "success",
                "message": "Configuration updated",
                "config": MeVeConfigSchema(
                    k_init=new_config.k_init,
                    tau_relevance=new_config.tau_relevance,
                    n_min=new_config.n_min,
                    theta_redundancy=new_config.theta_redundancy,
                    lambda_mmr=new_config.lambda_mmr,
                    t_max=new_config.t_max,
                ),
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/config/reset")
    async def reset_config(engine=Depends(get_engine)) -> dict:
        """Reset configuration to defaults."""
        try:
            default_config = MeVeConfig()
            if hasattr(engine, "config"):
                engine.config = default_config

            return {
                "status": "success",
                "message": "Configuration reset to defaults",
                "config": MeVeConfigSchema(
                    k_init=default_config.k_init,
                    tau_relevance=default_config.tau_relevance,
                    n_min=default_config.n_min,
                    theta_redundancy=default_config.theta_redundancy,
                    lambda_mmr=default_config.lambda_mmr,
                    t_max=default_config.t_max,
                ),
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # ========================
    # Data Source Endpoints
    # ========================

    @app.post("/data-sources/load")
    async def load_data_source(
        request: DataSourceLoadRequest,
        engine=Depends(get_engine),
    ) -> dict:
        """Load or create a ChromaDB collection."""
        try:
            logger.info(
                f"Loading data source", metadata={"collection": request.config.collection_name}
            )

            vector_db_config = {
                "collection_name": request.config.collection_name,
                "is_persistent": request.config.is_persistent,
                "load_existing": request.config.load_existing,
            }

            if hasattr(engine, "vector_db_config"):
                engine.vector_db_config = vector_db_config

            logger.info(
                f"Data source loaded", metadata={"collection": request.config.collection_name}
            )

            return {
                "status": "success",
                "message": f"Data source '{request.config.collection_name}' loaded",
                "config": {
                    "collection_name": request.config.collection_name,
                    "is_persistent": request.config.is_persistent,
                    "load_existing": request.config.load_existing,
                },
            }
        except Exception as e:
            logger.error("Failed to load data source", error=e)
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/data-sources/upload")
    async def upload_chunks(
        request: DataSourceUploadRequest,
        engine=Depends(get_engine),
    ) -> dict:
        """Upload new chunks to the active data source."""
        try:
            if not request.chunks:
                raise ValueError("No chunks provided")

            logger.info(f"Uploading {len(request.chunks)} chunks", metadata={})

            chunks_dict = {}
            for chunk_schema in request.chunks:
                chunk = ContextChunk(
                    content=chunk_schema.content,
                    doc_id=chunk_schema.doc_id,
                )
                chunks_dict[chunk.doc_id] = chunk

            if hasattr(engine, "vector_store"):
                engine.vector_store.update(chunks_dict)

            if hasattr(engine, "bm25_index"):
                engine.bm25_index.update(chunks_dict)

            logger.info(f"Successfully uploaded {len(chunks_dict)} chunks", metadata={})

            return {
                "status": "success",
                "message": f"{len(chunks_dict)} chunks uploaded",
                "count": len(chunks_dict),
            }
        except Exception as e:
            logger.error("Failed to upload chunks", error=e)
            raise HTTPException(status_code=400, detail=str(e))

    @app.get("/data-sources/info")
    async def data_source_info(engine=Depends(get_engine)) -> dict:
        """Get information about active data sources."""
        try:
            vector_store_size = len(engine.vector_store) if hasattr(engine, "vector_store") else 0
            bm25_index_size = len(engine.bm25_index) if hasattr(engine, "bm25_index") else 0

            return {
                "status": "success",
                "vector_store": {
                    "type": "ChromaDB",
                    "size": vector_store_size,
                    "ready": vector_store_size > 0,
                },
                "bm25_index": {
                    "type": "BM25",
                    "size": bm25_index_size,
                    "ready": bm25_index_size > 0,
                },
                "total_chunks": max(vector_store_size, bm25_index_size),
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/data-sources/list")
    async def list_chunks(
        doc_id: Optional[str] = None,
        limit: int = 100,
        engine=Depends(get_engine),
    ) -> dict:
        """List chunks in active data source."""
        try:
            chunks_dict = engine.vector_store if hasattr(engine, "vector_store") else {}

            if doc_id:
                chunks_dict = {k: v for k, v in chunks_dict.items() if k.startswith(doc_id)}

            chunk_items = list(chunks_dict.items())[:limit]

            chunks_list = [
                {
                    "doc_id": doc_id,
                    "content": chunk.content[:100] + "..."
                    if len(chunk.content) > 100
                    else chunk.content,
                    "content_length": len(chunk.content),
                }
                for doc_id, chunk in chunk_items
            ]

            return {
                "status": "success",
                "total_available": len(chunks_dict),
                "returned": len(chunks_list),
                "limit": limit,
                "chunks": chunks_list,
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    logger.info("FastAPI app created successfully")

    return app


# Global app instance for uvicorn
app = None


def init_app(engine: MeVeEngine):
    """Initialize the global app instance."""
    global app
    app = create_app(engine)
    return app
