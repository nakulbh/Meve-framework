"""
Pydantic schemas for FastAPI request/response validation.
Defines the API contracts for query submission, configuration, and metrics.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


class MeVeConfigSchema(BaseModel):
    """Request schema for MeVeConfig parameters."""

    k_init: int = Field(20, description="Phase 1: Initial kNN retrieval count")
    tau_relevance: float = Field(
        0.5, ge=0.0, le=1.0, description="Phase 2: Relevance threshold (0-1)"
    )
    n_min: int = Field(3, ge=1, description="Phase 3: Minimum verified chunks before BM25 fallback")
    theta_redundancy: float = Field(
        0.85, ge=0.0, le=1.0, description="Phase 4: Redundancy dedup threshold"
    )
    lambda_mmr: float = Field(
        0.6, ge=0.0, le=1.0, description="Phase 4: MMR lambda (relevance-diversity tradeoff)"
    )
    t_max: int = Field(512, ge=1, description="Phase 5: Token budget constraint")


class QueryRequest(BaseModel):
    """Request schema for query submission."""

    query: str = Field(..., description="User query for context retrieval", min_length=1)
    config: Optional[MeVeConfigSchema] = Field(None, description="Optional config override")


class ContextChunkSchema(BaseModel):
    """Response schema for a single context chunk."""

    content: str = Field(..., description="Chunk text content")
    doc_id: str = Field(..., description="Unique document identifier")
    relevance_score: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Cross-encoder or BM25 score"
    )
    token_count: Optional[int] = Field(None, ge=0, description="Token count in chunk")


class RetrievalMetrics(BaseModel):
    """Detailed phase-by-phase metrics from retrieval."""

    phase_1_candidates: int = Field(..., description="Candidates from Phase 1 (kNN)")
    phase_2_verified: int = Field(..., description="Chunks verified in Phase 2")
    phase_3_triggered: bool = Field(
        ..., description="Whether Phase 3 (BM25) fallback was triggered"
    )
    phase_4_deduplicated: int = Field(..., description="Chunks after Phase 4 deduplication")
    phase_5_final_tokens: int = Field(..., description="Total tokens in final context (Phase 5)")
    total_chunks_selected: int = Field(..., description="Total chunks in final selection")


class RetrievalResponse(BaseModel):
    """Response schema for successful retrieval."""

    status: str = Field("success", description="Response status")
    request_id: str = Field(..., description="Unique request identifier for tracing")
    query: str = Field(..., description="Original query")
    context: str = Field(..., description="Final context string ready for LLM consumption")
    chunks: List[ContextChunkSchema] = Field(..., description="Individual chunks with metadata")
    metrics: RetrievalMetrics = Field(..., description="Phase-by-phase retrieval metrics")
    processing_time_ms: float = Field(
        ..., ge=0, description="Total query processing time in milliseconds"
    )


class ErrorResponse(BaseModel):
    """Response schema for errors."""

    status: str = Field("error", description="Response status")
    request_id: str = Field(..., description="Unique request identifier for tracing")
    code: str = Field(..., description="Error code for programmatic handling")
    message: str = Field(..., description="Human-readable error message")


class DataSourceConfig(BaseModel):
    """Schema for ChromaDB data source configuration."""

    collection_name: str = Field(..., description="Name of ChromaDB collection")
    is_persistent: bool = Field(True, description="Whether to persist collection to disk")
    load_existing: bool = Field(False, description="Load existing collection or create new")


class DataSourceLoadRequest(BaseModel):
    """Request to load a data source."""

    config: DataSourceConfig


class DataSourceUploadRequest(BaseModel):
    """Request to upload new chunks to data source."""

    chunks: List[ContextChunkSchema] = Field(..., min_items=1, description="Chunks to add")


class ConfigUpdateRequest(BaseModel):
    """Request to update pipeline configuration."""

    config: MeVeConfigSchema


class HealthStatus(str, Enum):
    """Enum for health status."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class HealthResponse(BaseModel):
    """Response schema for health check."""

    status: HealthStatus = Field(..., description="System health status")
    vector_db_ready: bool = Field(..., description="Vector database availability")
    bm25_index_ready: bool = Field(..., description="BM25 index availability")
    current_config: MeVeConfigSchema = Field(..., description="Current pipeline configuration")
    message: Optional[str] = Field(None, description="Additional status message")


class MetricsResponse(BaseModel):
    """Response schema for metrics query."""

    request_id: str = Field(..., description="Request ID to query metrics for")
    query: str = Field(..., description="Query that was executed")
    metrics: RetrievalMetrics = Field(..., description="Retrieval metrics")
    processing_time_ms: float = Field(..., description="Processing time")
    timestamp: str = Field(..., description="ISO 8601 timestamp of request")
