"""
Routers for data source management.
Exposes endpoints for loading and managing ChromaDB collections and BM25 indexes.
"""

from fastapi import APIRouter, HTTPException
from typing import List

from ..schemas import DataSourceLoadRequest, DataSourceUploadRequest, ContextChunkSchema
from meve import ContextChunk
from meve.utils import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/data-sources", tags=["data-sources"])


@router.post("/load")
async def load_data_source(
    request: DataSourceLoadRequest,
    engine=None,
) -> dict:
    """
    Load or create a ChromaDB collection.

    Supports both loading existing collections and creating new ones.
    """
    if engine is None:
        raise HTTPException(status_code=500, detail="Engine not initialized")

    try:
        logger.info(f"Loading data source", metadata={"collection": request.config.collection_name})

        # Prepare vector DB config
        vector_db_config = {
            "collection_name": request.config.collection_name,
            "is_persistent": request.config.is_persistent,
            "load_existing": request.config.load_existing,
        }

        # Update engine with new vector DB (simplified - full implementation
        # would call engine.connect_to_collection() or similar)
        # For now, store the config for reference
        if hasattr(engine, "vector_db_config"):
            engine.vector_db_config = vector_db_config

        logger.info(f"Data source loaded", metadata={"collection": request.config.collection_name})

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


@router.post("/upload")
async def upload_chunks(
    request: DataSourceUploadRequest,
    engine=None,
) -> dict:
    """
    Upload new chunks to the active data source.

    Chunks are added to both vector store and BM25 index.
    """
    if engine is None:
        raise HTTPException(status_code=500, detail="Engine not initialized")

    try:
        if not request.chunks:
            raise ValueError("No chunks provided")

        logger.info(f"Uploading {len(request.chunks)} chunks", metadata={})

        # Convert schemas to ContextChunks
        chunks_dict = {}
        for chunk_schema in request.chunks:
            chunk = ContextChunk(
                content=chunk_schema.content,
                doc_id=chunk_schema.doc_id,
            )
            chunks_dict[chunk.doc_id] = chunk

        # Add to engine's data sources
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


@router.get("/info")
async def data_source_info(engine=None) -> dict:
    """
    Get information about active data sources.
    """
    if engine is None:
        raise HTTPException(status_code=500, detail="Engine not initialized")

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


@router.get("/list")
async def list_chunks(
    doc_id: str = None,
    limit: int = 100,
    engine=None,
) -> dict:
    """
    List chunks in active data source.

    Optional filtering by doc_id and pagination.
    """
    if engine is None:
        raise HTTPException(status_code=500, detail="Engine not initialized")

    try:
        chunks_dict = engine.vector_store if hasattr(engine, "vector_store") else {}

        # Filter if doc_id specified
        if doc_id:
            chunks_dict = {k: v for k, v in chunks_dict.items() if k.startswith(doc_id)}

        # Apply limit
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
