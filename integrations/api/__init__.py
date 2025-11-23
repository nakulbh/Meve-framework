"""
FastAPI REST API for MeVe Framework.

Exposes the 5-phase RAG retrieval pipeline with:
- Query retrieval with phase-by-phase metrics
- Configuration management and tuning
- Health monitoring and status checks
- Data source management (ChromaDB, BM25)

Example Usage:
    from meve import MeVeEngine, MeVeConfig
    from integrations.api import create_app

    # Initialize engine
    config = MeVeConfig()
    engine = MeVeEngine(config, vector_store={...}, bm25_index={...})

    # Create app
    app = create_app(engine)

    # Run with uvicorn
    # uvicorn integrations.api:app --reload

Endpoints:
    POST /retrieve - Execute retrieval pipeline
    GET /retrieve/metrics/{request_id} - Get request metrics
    GET /health - System health check
    GET /config - Get current configuration
    POST /config - Update configuration
    POST /data-sources/load - Load data source
    POST /data-sources/upload - Upload chunks
"""

from integrations.api.app import create_app, init_app
from integrations.api.schemas import (
    MeVeConfigSchema,
    QueryRequest,
    RetrievalResponse,
    ContextChunkSchema,
    RetrievalMetrics,
    ErrorResponse,
    DataSourceConfig,
    DataSourceLoadRequest,
    DataSourceUploadRequest,
    ConfigUpdateRequest,
    HealthResponse,
    HealthStatus,
    MetricsResponse,
)
from integrations.api.request_handler import RequestHandler, RequestContext

__all__ = [
    # App factory
    "create_app",
    "init_app",
    # Schemas
    "MeVeConfigSchema",
    "QueryRequest",
    "RetrievalResponse",
    "ContextChunkSchema",
    "RetrievalMetrics",
    "ErrorResponse",
    "DataSourceConfig",
    "DataSourceLoadRequest",
    "DataSourceUploadRequest",
    "ConfigUpdateRequest",
    "HealthResponse",
    "HealthStatus",
    "MetricsResponse",
    # Utilities
    "RequestHandler",
    "RequestContext",
]
