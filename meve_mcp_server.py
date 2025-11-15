#!/usr/bin/env python3
"""
MeVe MCP Server - Model Context Protocol server for the MeVe RAG framework.

This server exposes the MeVe (Memory-Enhanced Vector) RAG pipeline through the
Model Context Protocol, allowing AI clients like Claude to access the 5-phase
retrieval system for intelligent context management.
"""

import json
import logging
import sys
import os
from typing import Dict, List, Optional, Any
from urllib.parse import urlparse
import io

# MCP imports
from mcp.server.fastmcp import FastMCP

# MeVe framework imports
from meve_data import ContextChunk, Query, MeVeConfig
from meve_engine import MeVeEngine, setup_meve_data
from phase1_knn import execute_phase_1
from phase2_verification import execute_phase_2
from phase3_fallback import execute_phase_3
from phase4_prioritization import execute_phase_4
from phase5_budgeting import execute_phase_5

# Set up logging to stderr (required for STDIO transport)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stderr)]
)
logger = logging.getLogger("meve-mcp-server")

# Initialize FastMCP server
mcp = FastMCP("meve-rag-server")

# ChromaDB Configuration
class ChromaDBConfig:
    def __init__(self, 
                 host: str = "localhost",
                 port: int = 8000,
                 use_persistent: bool = False,
                 persist_path: str = "./chroma_db",
                 collection_name: str = "meve_collection",
                 use_remote: bool = False):
        self.host = host
        self.port = port
        self.use_persistent = use_persistent
        self.persist_path = persist_path
        self.collection_name = collection_name
        self.use_remote = use_remote
    
    def get_client_settings(self):
        """Get ChromaDB client settings based on configuration."""
        import chromadb
        
        if self.use_remote:
            # Remote ChromaDB server
            return chromadb.HttpClient(host=self.host, port=self.port)
        elif self.use_persistent:
            # Local persistent ChromaDB
            return chromadb.PersistentClient(path=self.persist_path)
        else:
            # In-memory ChromaDB (default)
            return chromadb.Client()

def parse_chromadb_url(url: str) -> tuple[str, int]:
    """
    Parse a ChromaDB server URL and extract host and port.

    Examples:
        http://localhost:8000 -> ('localhost', 8000)
        https://my-chroma-server.com:8080 -> ('my-chroma-server.com', 8080)
        my-server.com -> ('my-server.com', 8000)

    Args:
        url: ChromaDB server URL (with or without http://)

    Returns:
        Tuple of (host, port)
    """
    # Add scheme if not present
    if not url.startswith(('http://', 'https://')):
        url = f'http://{url}'

    parsed = urlparse(url)
    host = parsed.hostname or 'localhost'
    port = parsed.port or 8000

    return host, port

# Global state for the MeVe framework
class MeVeServerState:
    def __init__(self):
        self.config: MeVeConfig = MeVeConfig()
        self.chroma_config: ChromaDBConfig = ChromaDBConfig()
        self.vector_store: Dict[str, ContextChunk] = {}
        self.bm25_index: Dict[str, ContextChunk] = {}
        self.questions: List[Dict] = []
        self.pipeline_stats: Dict[str, Any] = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "avg_retrieval_time": 0.0,
            "phase_usage": {
                "phase1": 0,
                "phase2": 0,
                "phase3": 0,
                "phase4": 0,
                "phase5": 0
            }
        }
        self.is_initialized: bool = False
        self.external_chroma_client = None  # For connecting to existing ChromaDB
    
    def reset_stats(self):
        """Reset pipeline statistics."""
        self.pipeline_stats = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "avg_retrieval_time": 0.0,
            "phase_usage": {
                "phase1": 0,
                "phase2": 0,
                "phase3": 0,
                "phase4": 0,
                "phase5": 0
            }
        }

# Global server state instance
server_state = MeVeServerState()

# Context manager to suppress stdout during MeVe pipeline execution
class SuppressStdout:
    """Redirect stdout to stderr to prevent print() from interfering with MCP JSON-RPC."""
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = sys.stderr
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._original_stdout
        return False

async def initialize_meve_data():
    """Initialize MeVe data if not already done."""
    if server_state.is_initialized:
        return
    
    try:
        logger.info("Initializing MeVe data...")
        vector_store, bm25_index, questions = setup_meve_data(
            data_dir="data", 
            max_examples=100
        )
        
        server_state.vector_store = vector_store
        server_state.bm25_index = bm25_index
        server_state.questions = questions
        server_state.is_initialized = True
        
        logger.info(f"MeVe data initialized: {len(vector_store)} chunks, {len(questions)} questions")
        
    except Exception as e:
        logger.error(f"Failed to initialize MeVe data: {str(e)}")
        # Initialize with empty data to allow server to start
        server_state.vector_store = {}
        server_state.bm25_index = {}
        server_state.questions = []
        server_state.is_initialized = True

def format_pipeline_results(
    final_context: str,
    final_chunks: List[ContextChunk],
    phase_results: Dict[str, Any]
) -> str:
    """Format pipeline results for MCP response."""
    result = {
        "final_context": final_context,
        "final_context_length": len(final_context),
        "final_chunks_count": len(final_chunks),
        "phase_breakdown": phase_results,
        "chunks": [
            {
                "doc_id": chunk.doc_id,
                "content": chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content,
                "relevance_score": chunk.relevance_score,
                "token_count": chunk.token_count
            }
            for chunk in final_chunks
        ]
    }
    
    return json.dumps(result, indent=2)

# ==================== MCP TOOLS ====================

@mcp.tool()
async def query_with_meve(
    query: str,
    k_init: int = 20,
    tau_relevance: float = 0.5,
    n_min: int = 3,
    theta_redundancy: float = 0.85,
    t_max: int = 512
) -> str:
    """
    Execute the complete MeVe 5-phase pipeline for a query.
    
    Args:
        query: The search query text
        k_init: Initial retrieval count for Phase 1 (default: 20)
        tau_relevance: Relevance threshold for Phase 2 (default: 0.5)
        n_min: Minimum verified docs to avoid fallback (default: 3)
        theta_redundancy: Redundancy threshold for Phase 4 (default: 0.85)
        t_max: Maximum token budget for Phase 5 (default: 512)
    
    Returns:
        JSON string with final context, metadata, and phase breakdown
    """
    try:
        await initialize_meve_data()
        
        if not server_state.vector_store:
            return json.dumps({"error": "No data available. Please load documents first."})
        
        # Create custom config for this query
        config = MeVeConfig(
            k_init=k_init,
            tau_relevance=tau_relevance,
            n_min=n_min,
            theta_redundancy=theta_redundancy,
            t_max=t_max
        )
        
        # Execute MeVe pipeline (suppress stdout to prevent print() from interfering with MCP)
        engine = MeVeEngine(config, server_state.vector_store, server_state.bm25_index)
        with SuppressStdout():
            final_context = engine.run(query)
        
        # Update stats
        server_state.pipeline_stats["total_queries"] += 1
        server_state.pipeline_stats["successful_queries"] += 1
        
        # For now, return the final context directly
        # In a full implementation, you'd capture detailed phase results
        result = {
            "final_context": final_context,
            "final_context_length": len(final_context),
            "query": query,
            "config_used": {
                "k_init": k_init,
                "tau_relevance": tau_relevance,
                "n_min": n_min,
                "theta_redundancy": theta_redundancy,
                "t_max": t_max
            },
            "status": "success"
        }
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        logger.error(f"Error in query_with_meve: {str(e)}")
        server_state.pipeline_stats["total_queries"] += 1
        server_state.pipeline_stats["failed_queries"] += 1
        
        return json.dumps({
            "error": f"Query failed: {str(e)}",
            "query": query,
            "status": "failed"
        })

@mcp.tool()
async def configure_pipeline(
    k_init: Optional[int] = None,
    tau_relevance: Optional[float] = None,
    n_min: Optional[int] = None,
    theta_redundancy: Optional[float] = None,
    t_max: Optional[int] = None
) -> str:
    """
    Update the default MeVe pipeline configuration.
    
    Args:
        k_init: Initial retrieval count (Phase 1)
        tau_relevance: Relevance threshold (Phase 2)
        n_min: Minimum verified docs threshold (Phase 3)
        theta_redundancy: Redundancy threshold (Phase 4)
        t_max: Token budget limit (Phase 5)
    
    Returns:
        JSON string with updated configuration
    """
    try:
        # Update only provided parameters
        if k_init is not None:
            server_state.config.k_init = k_init
        if tau_relevance is not None:
            server_state.config.tau_relevance = tau_relevance
        if n_min is not None:
            server_state.config.n_min = n_min
        if theta_redundancy is not None:
            server_state.config.theta_redundancy = theta_redundancy
        if t_max is not None:
            server_state.config.t_max = t_max
        
        result = {
            "status": "success",
            "message": "Configuration updated successfully",
            "new_config": {
                "k_init": server_state.config.k_init,
                "tau_relevance": server_state.config.tau_relevance,
                "n_min": server_state.config.n_min,
                "theta_redundancy": server_state.config.theta_redundancy,
                "t_max": server_state.config.t_max
            }
        }
        
        logger.info("Pipeline configuration updated")
        return json.dumps(result, indent=2)
        
    except Exception as e:
        logger.error(f"Error updating configuration: {str(e)}")
        return json.dumps({
            "error": f"Configuration update failed: {str(e)}",
            "status": "failed"
        })

@mcp.tool()
async def analyze_retrieval(query: str, k_init: int = 20) -> str:
    """
    Analyze retrieval quality with detailed phase-by-phase breakdown.
    
    Args:
        query: The search query text
        k_init: Initial retrieval count for analysis
    
    Returns:
        JSON string with detailed phase analysis
    """
    try:
        await initialize_meve_data()
        
        if not server_state.vector_store:
            return json.dumps({"error": "No data available for analysis."})
        
        query_obj = Query(text=query, vector=None)
        config = MeVeConfig(k_init=k_init)
        
        # Execute each phase separately for detailed analysis
        analysis = {
            "query": query,
            "k_init": k_init,
            "phases": {}
        }

        # Suppress stdout for all phase executions
        with SuppressStdout():
            # Phase 1: Initial Retrieval
            try:
                phase1_results = execute_phase_1(query_obj, config, server_state.vector_store)
                    analysis["phases"]["phase1"] = {
                        "description": "Initial kNN retrieval using ChromaDB",
                        "candidates_retrieved": len(phase1_results),
                        "top_candidates": [
                            {
                                "doc_id": chunk.doc_id,
                                "content_preview": chunk.content[:100] + "..." if len(chunk.content) > 100 else chunk.content,
                                "relevance_score": getattr(chunk, 'relevance_score', 0.0)
                            }
                            for chunk in phase1_results[:5]
                        ],
                        "status": "success"
                    }

                    # Phase 2: Verification
                    try:
                        phase2_results = execute_phase_2(query_obj, phase1_results, config)
                        analysis["phases"]["phase2"] = {
                            "description": "Cross-encoder relevance verification",
                            "input_candidates": len(phase1_results),
                            "verified_chunks": len(phase2_results),
                            "filter_ratio": len(phase2_results) / len(phase1_results) if phase1_results else 0,
                            "threshold_used": config.tau_relevance,
                            "status": "success"
                        }

                        # Phase 3: Fallback (conditional)
                        fallback_triggered = len(phase2_results) < config.n_min
                        if fallback_triggered:
                            try:
                                phase3_results = execute_phase_3(query_obj, server_state.bm25_index)
                                analysis["phases"]["phase3"] = {
                                    "description": "BM25 fallback retrieval (triggered)",
                                    "trigger_condition": f"verified_chunks ({len(phase2_results)}) < n_min ({config.n_min})",
                                    "fallback_chunks": len(phase3_results),
                                    "combined_total": len(phase2_results) + len(phase3_results),
                                    "status": "executed"
                                }
                                combined_context = phase2_results + phase3_results
                            except Exception as e:
                                analysis["phases"]["phase3"] = {"error": str(e), "status": "failed"}
                                combined_context = phase2_results
                        else:
                            analysis["phases"]["phase3"] = {
                                "description": "BM25 fallback retrieval (skipped)",
                                "trigger_condition": f"verified_chunks ({len(phase2_results)}) >= n_min ({config.n_min})",
                                "status": "skipped"
                            }
                            combined_context = phase2_results

                        # Phase 4: Prioritization
                        try:
                            phase4_results = execute_phase_4(query_obj, combined_context, config)
                            analysis["phases"]["phase4"] = {
                                "description": "Context prioritization and redundancy removal",
                                "input_chunks": len(combined_context),
                                "prioritized_chunks": len(phase4_results),
                                "redundancy_threshold": config.theta_redundancy,
                                "status": "success"
                            }

                            # Phase 5: Token Budgeting
                            try:
                                final_context, final_chunks = execute_phase_5(phase4_results, config)
                                analysis["phases"]["phase5"] = {
                                    "description": "Token budgeting with greedy packing",
                                    "input_chunks": len(phase4_results),
                                    "final_chunks": len(final_chunks),
                                    "token_budget": config.t_max,
                                    "final_context_length": len(final_context),
                                    "budget_efficiency": f"{(len(final_context.split()) / config.t_max * 100):.1f}%",
                                    "status": "success"
                                }

                                # Overall summary
                                analysis["summary"] = {
                                    "total_phases_executed": 5,
                                    "fallback_triggered": fallback_triggered,
                                    "final_chunks_count": len(final_chunks),
                                    "context_reduction_ratio": f"{(len(final_chunks) / k_init * 100):.1f}%",
                                    "pipeline_success": True
                                }

                            except Exception as e:
                                analysis["phases"]["phase5"] = {"error": str(e), "status": "failed"}
                        except Exception as e:
                            analysis["phases"]["phase4"] = {"error": str(e), "status": "failed"}
                    except Exception as e:
                        analysis["phases"]["phase2"] = {"error": str(e), "status": "failed"}
            except Exception as e:
                analysis["phases"]["phase1"] = {"error": str(e), "status": "failed"}
        
        return json.dumps(analysis, indent=2)
        
    except Exception as e:
        logger.error(f"Error in analyze_retrieval: {str(e)}")
        return json.dumps({
            "error": f"Analysis failed: {str(e)}",
            "query": query
        })

@mcp.tool()
async def get_pipeline_status() -> str:
    """
    Get current status of the MeVe pipeline server.
    
    Returns:
        JSON string with server status and statistics
    """
    try:
        await initialize_meve_data()
        
        status = {
            "server_status": "running",
            "data_initialized": server_state.is_initialized,
            "knowledge_base": {
                "vector_store_chunks": len(server_state.vector_store),
                "bm25_index_chunks": len(server_state.bm25_index),
                "available_questions": len(server_state.questions)
            },
            "current_config": {
                "k_init": server_state.config.k_init,
                "tau_relevance": server_state.config.tau_relevance,
                "n_min": server_state.config.n_min,
                "theta_redundancy": server_state.config.theta_redundancy,
                "t_max": server_state.config.t_max
            },
            "statistics": server_state.pipeline_stats
        }
        
        return json.dumps(status, indent=2)
        
    except Exception as e:
        logger.error(f"Error getting pipeline status: {str(e)}")
        return json.dumps({
            "error": f"Status check failed: {str(e)}",
            "server_status": "error"
        })

@mcp.tool()
async def execute_phase1_only(query: str, k_init: int = 20) -> str:
    """
    Execute only Phase 1 (ChromaDB kNN retrieval) for testing and analysis.
    
    Args:
        query: The search query text
        k_init: Initial retrieval count
    
    Returns:
        JSON string with Phase 1 results
    """
    try:
        await initialize_meve_data()
        
        if not server_state.vector_store:
            return json.dumps({"error": "No data available."})
        
        query_obj = Query(text=query, vector=None)
        config = MeVeConfig(k_init=k_init)

        # Execute Phase 1 only (suppress stdout)
        with SuppressStdout():
            phase1_results = execute_phase_1(query_obj, config, server_state.vector_store)
        
        result = {
            "phase": "Phase 1 - ChromaDB kNN Retrieval",
            "query": query,
            "k_init": k_init,
            "results": {
                "candidates_retrieved": len(phase1_results),
                "candidates": [
                    {
                        "doc_id": chunk.doc_id,
                        "content": chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content,
                        "relevance_score": chunk.relevance_score,
                        "embedding_dimensions": len(chunk.embedding) if chunk.embedding else None
                    }
                    for chunk in phase1_results
                ]
            },
            "status": "success"
        }
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        logger.error(f"Error in execute_phase1_only: {str(e)}")
        return json.dumps({
            "error": f"Phase 1 execution failed: {str(e)}",
            "query": query
        })

@mcp.tool()
async def execute_phase5_only(chunks_json: str, t_max: int = 512) -> str:
    """
    Execute only Phase 5 (token budgeting) with provided chunks.
    
    Args:
        chunks_json: JSON string containing chunks data with content and relevance_score
        t_max: Token budget limit
    
    Returns:
        JSON string with Phase 5 results
    """
    try:
        # Parse input chunks
        chunks_data = json.loads(chunks_json)
        
        # Convert to ContextChunk objects
        chunks = []
        for i, chunk_data in enumerate(chunks_data):
            chunk = ContextChunk(
                content=chunk_data.get('content', ''),
                doc_id=chunk_data.get('doc_id', f'chunk_{i}'),
                embedding=chunk_data.get('embedding')
            )
            chunk.relevance_score = chunk_data.get('relevance_score', 0.0)
            chunks.append(chunk)
        
        config = MeVeConfig(t_max=t_max)

        # Execute Phase 5 only (suppress stdout)
        with SuppressStdout():
            final_context, final_chunks = execute_phase_5(chunks, config)
        
        result = {
            "phase": "Phase 5 - Token Budgeting",
            "token_budget": t_max,
            "results": {
                "input_chunks": len(chunks),
                "final_chunks": len(final_chunks),
                "final_context": final_context,
                "final_context_length": len(final_context),
                "token_efficiency": f"{(len(final_context.split()) / t_max * 100):.1f}%",
                "chunks_included": [
                    {
                        "doc_id": chunk.doc_id,
                        "content": chunk.content[:100] + "..." if len(chunk.content) > 100 else chunk.content,
                        "token_count": chunk.token_count,
                        "relevance_score": chunk.relevance_score
                    }
                    for chunk in final_chunks
                ]
            },
            "status": "success"
        }
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        logger.error(f"Error in execute_phase5_only: {str(e)}")
        return json.dumps({
            "error": f"Phase 5 execution failed: {str(e)}",
            "chunks_json": chunks_json[:100] + "..." if len(chunks_json) > 100 else chunks_json
        })

@mcp.tool()
async def benchmark_efficiency(queries: list, collection_name: str = "default") -> str:
    """
    Benchmark MeVe efficiency against standard RAG approach.
    
    Args:
        queries: List of test queries
        collection_name: Name of the collection to benchmark against
    
    Returns:
        JSON string with efficiency comparison metrics
    """
    try:
        await initialize_meve_data()
        
        if not server_state.vector_store:
            return json.dumps({"error": "No data available for benchmarking."})
        
        results = {
            "benchmark_type": "MeVe vs Standard RAG Efficiency",
            "collection": collection_name,
            "queries_tested": len(queries),
            "results": []
        }
        
        total_meve_chunks = 0
        total_standard_chunks = 0
        total_meve_tokens = 0
        total_standard_tokens = 0
        
        for i, query in enumerate(queries):
            query_obj = Query(text=query, vector=None)
            config = MeVeConfig()
            
            # MeVe pipeline (suppress stdout)
            try:
                with SuppressStdout():
                    engine = MeVeEngine(config, server_state.vector_store, server_state.bm25_index)
                    meve_context = engine.run(query)
                meve_chunks = len([chunk for chunk in server_state.vector_store.values()
                                 if chunk.content[:50] in meve_context])
                meve_tokens = len(meve_context.split())

            except Exception as e:
                meve_chunks = 0
                meve_tokens = 0
                logger.error(f"MeVe pipeline error for query {i}: {str(e)}")

            # Standard RAG (just top-k without filtering, suppress stdout)
            try:
                with SuppressStdout():
                    phase1_results = execute_phase_1(query_obj, config, server_state.vector_store)
                standard_chunks = len(phase1_results)
                standard_context = "\n".join([chunk.content for chunk in phase1_results])
                standard_tokens = len(standard_context.split())
                
            except Exception as e:
                standard_chunks = config.k_init
                standard_tokens = config.k_init * 100  # Rough estimate
                logger.error(f"Standard RAG error for query {i}: {str(e)}")
            
            query_result = {
                "query": query,
                "meve": {
                    "chunks": meve_chunks,
                    "tokens": meve_tokens
                },
                "standard_rag": {
                    "chunks": standard_chunks,
                    "tokens": standard_tokens
                },
                "efficiency_gain": {
                    "chunk_reduction": f"{((standard_chunks - meve_chunks) / standard_chunks * 100):.1f}%" if standard_chunks > 0 else "N/A",
                    "token_reduction": f"{((standard_tokens - meve_tokens) / standard_tokens * 100):.1f}%" if standard_tokens > 0 else "N/A"
                }
            }
            
            results["results"].append(query_result)
            
            total_meve_chunks += meve_chunks
            total_standard_chunks += standard_chunks
            total_meve_tokens += meve_tokens
            total_standard_tokens += standard_tokens
        
        # Overall statistics
        results["summary"] = {
            "avg_chunk_reduction": f"{((total_standard_chunks - total_meve_chunks) / total_standard_chunks * 100):.1f}%" if total_standard_chunks > 0 else "N/A",
            "avg_token_reduction": f"{((total_standard_tokens - total_meve_tokens) / total_standard_tokens * 100):.1f}%" if total_standard_tokens > 0 else "N/A",
            "meve_avg_chunks_per_query": total_meve_chunks / len(queries) if queries else 0,
            "standard_avg_chunks_per_query": total_standard_chunks / len(queries) if queries else 0,
            "efficiency_ratio": (total_meve_tokens / total_standard_tokens) if total_standard_tokens > 0 else 0
        }
        
        return json.dumps(results, indent=2)
        
    except Exception as e:
        logger.error(f"Error in benchmark_efficiency: {str(e)}")
        return json.dumps({
            "error": f"Benchmarking failed: {str(e)}",
            "queries": queries
        })

@mcp.tool()
async def index_documents(documents: list, collection_name: str = "custom", metadata: dict = None) -> str:
    """
    Index new documents into the MeVe system.
    
    Args:
        documents: List of document strings to index
        collection_name: Name for the new collection
        metadata: Optional metadata for the documents
    
    Returns:
        JSON string with indexing results
    """
    try:
        if not documents:
            return json.dumps({"error": "No documents provided for indexing."})
        
        # Create chunks from documents
        new_chunks = {}
        chunk_stats = {
            "total_documents": len(documents),
            "total_chunks": 0,
            "avg_chunk_length": 0,
            "total_characters": 0
        }
        
        for doc_idx, doc_content in enumerate(documents):
            if not doc_content or not doc_content.strip():
                continue
                
            # Simple chunking by paragraph or fixed size
            paragraphs = doc_content.split('\n\n')
            for para_idx, paragraph in enumerate(paragraphs):
                if len(paragraph.strip()) > 50:  # Minimum chunk size
                    chunk_id = f"{collection_name}_{doc_idx}_{para_idx}"
                    chunk = ContextChunk(
                        content=paragraph.strip(),
                        doc_id=chunk_id,
                        embedding=None
                    )
                    
                    # Add metadata if provided
                    if metadata:
                        chunk.metadata = metadata.copy()
                        chunk.metadata['document_index'] = doc_idx
                        chunk.metadata['chunk_index'] = para_idx
                    
                    new_chunks[chunk_id] = chunk
                    chunk_stats["total_chunks"] += 1
                    chunk_stats["total_characters"] += len(paragraph)
        
        if chunk_stats["total_chunks"] > 0:
            chunk_stats["avg_chunk_length"] = chunk_stats["total_characters"] // chunk_stats["total_chunks"]
        
        # Add to server state (in a real implementation, this would persist to database)
        server_state.vector_store.update(new_chunks)
        server_state.bm25_index.update(new_chunks)
        
        result = {
            "indexing_status": "success",
            "collection_name": collection_name,
            "statistics": chunk_stats,
            "new_chunk_ids": list(new_chunks.keys())[:10],  # Show first 10 IDs
            "total_chunks_in_system": len(server_state.vector_store),
            "message": f"Successfully indexed {chunk_stats['total_chunks']} chunks from {chunk_stats['total_documents']} documents"
        }
        
        logger.info(f"Indexed {chunk_stats['total_chunks']} chunks into collection '{collection_name}'")
        return json.dumps(result, indent=2)
        
    except Exception as e:
        logger.error(f"Error in index_documents: {str(e)}")
        return json.dumps({
            "error": f"Indexing failed: {str(e)}",
            "documents_count": len(documents) if documents else 0
        })

@mcp.tool()
async def connect_to_chromadb(
    server_url: str = None,
    connection_type: str = "memory",
    host: str = "localhost",
    port: int = 8000,
    persist_path: str = "./chroma_db",
    collection_name: str = "meve_collection"
) -> str:
    """
    Connect to an external ChromaDB instance.

    Args:
        server_url: Full ChromaDB server URL (e.g., "http://localhost:8000" or "https://my-chroma.com:8080")
                    If provided, this overrides host and port parameters.
        connection_type: "memory", "persistent", or "remote" (default: "memory")
        host: Host for remote ChromaDB server (default: "localhost")
        port: Port for remote ChromaDB server (default: 8000)
        persist_path: Path for persistent ChromaDB storage (default: "./chroma_db")
        collection_name: Name of the collection to use (default: "meve_collection")

    Examples:
        # Connect using URL
        connect_to_chromadb(server_url="http://localhost:8000", collection_name="my_docs")

        # Connect using Docker default
        connect_to_chromadb(server_url="http://localhost:8000")

        # Connect to remote server
        connect_to_chromadb(server_url="https://my-chroma-server.com:8080")

        # Connect to persistent local database
        connect_to_chromadb(connection_type="persistent", persist_path="/path/to/chroma_db")

    Returns:
        JSON string with connection status
    """
    try:
        import chromadb

        # Parse server URL if provided
        if server_url:
            host, port = parse_chromadb_url(server_url)
            # If URL is provided, assume remote connection
            if connection_type == "memory":
                connection_type = "remote"
            logger.info(f"Parsed ChromaDB URL '{server_url}' -> host={host}, port={port}")

        # Check for environment variable if no URL provided
        elif not server_url and os.getenv("CHROMADB_URL"):
            env_url = os.getenv("CHROMADB_URL")
            host, port = parse_chromadb_url(env_url)
            if connection_type == "memory":
                connection_type = "remote"
            logger.info(f"Using ChromaDB URL from environment: {env_url}")

        # Update ChromaDB configuration
        server_state.chroma_config = ChromaDBConfig(
            host=host,
            port=port,
            use_persistent=(connection_type == "persistent"),
            persist_path=persist_path,
            collection_name=collection_name,
            use_remote=(connection_type == "remote")
        )

        # Create client based on type
        if connection_type == "remote":
            client = chromadb.HttpClient(host=host, port=port)
            connection_info = f"Remote ChromaDB at {host}:{port}"
        elif connection_type == "persistent":
            client = chromadb.PersistentClient(path=persist_path)
            connection_info = f"Persistent ChromaDB at {persist_path}"
        else:
            client = chromadb.Client()
            connection_info = "In-memory ChromaDB"
        
        # Test connection by listing collections
        collections = client.list_collections()
        collection_names = [col.name for col in collections]
        
        # Try to get the specified collection
        try:
            collection = client.get_collection(collection_name)
            collection_count = collection.count()
            collection_exists = True
        except Exception:
            collection_count = 0
            collection_exists = False
        
        # Store the client for use in the pipeline
        server_state.external_chroma_client = client
        
        result = {
            "status": "success",
            "connection_type": connection_type,
            "connection_info": connection_info,
            "available_collections": collection_names,
            "target_collection": {
                "name": collection_name,
                "exists": collection_exists,
                "document_count": collection_count
            },
            "configuration": {
                "host": host,
                "port": port,
                "persist_path": persist_path if connection_type == "persistent" else None
            }
        }
        
        logger.info(f"Connected to ChromaDB: {connection_info}")
        return json.dumps(result, indent=2)
        
    except Exception as e:
        logger.error(f"Failed to connect to ChromaDB: {str(e)}")
        return json.dumps({
            "error": f"ChromaDB connection failed: {str(e)}",
            "connection_type": connection_type,
            "status": "failed"
        })

@mcp.tool()
async def query_external_chromadb(
    query: str,
    collection_name: str,
    n_results: int = 10,
    include_metadata: bool = True
) -> str:
    """
    Query an external ChromaDB collection directly.
    
    Args:
        query: Search query text
        collection_name: Name of the ChromaDB collection
        n_results: Number of results to return
        include_metadata: Whether to include metadata in results
    
    Returns:
        JSON string with query results
    """
    try:
        if not server_state.external_chroma_client:
            return json.dumps({
                "error": "No external ChromaDB connection. Use connect_to_chromadb first.",
                "status": "failed"
            })
        
        # Get the collection
        collection = server_state.external_chroma_client.get_collection(collection_name)
        
        # Query the collection
        results = collection.query(
            query_texts=[query],
            n_results=n_results,
            include=["documents", "metadatas", "distances"] if include_metadata else ["documents", "distances"]
        )
        
        # Format results
        formatted_results = []
        documents = results["documents"][0] if results["documents"] else []
        distances = results["distances"][0] if results["distances"] else []
        metadatas = results.get("metadatas", [[]])[0] if include_metadata else []
        
        for i, doc in enumerate(documents):
            result_item = {
                "document": doc,
                "distance": distances[i] if i < len(distances) else None,
                "similarity": 1.0 / (1.0 + distances[i]) if i < len(distances) else None
            }
            
            if include_metadata and i < len(metadatas) and metadatas[i]:
                result_item["metadata"] = metadatas[i]
            
            formatted_results.append(result_item)
        
        response = {
            "query": query,
            "collection": collection_name,
            "results_count": len(formatted_results),
            "results": formatted_results,
            "status": "success"
        }
        
        return json.dumps(response, indent=2)
        
    except Exception as e:
        logger.error(f"External ChromaDB query failed: {str(e)}")
        return json.dumps({
            "error": f"ChromaDB query failed: {str(e)}",
            "query": query,
            "collection": collection_name,
            "status": "failed"
        })

@mcp.tool()
async def list_chromadb_collections() -> str:
    """
    List all collections in the connected ChromaDB instance.
    
    Returns:
        JSON string with collection information
    """
    try:
        if not server_state.external_chroma_client:
            return json.dumps({
                "error": "No external ChromaDB connection. Use connect_to_chromadb first.",
                "status": "failed"
            })
        
        # List all collections
        collections = server_state.external_chroma_client.list_collections()
        
        collection_info = []
        for col in collections:
            try:
                count = col.count()
                # Try to get some metadata about the collection
                sample = col.peek(limit=1)
                has_embeddings = len(sample.get("embeddings", [])) > 0
                
                collection_info.append({
                    "name": col.name,
                    "document_count": count,
                    "has_embeddings": has_embeddings,
                    "metadata": col.metadata if hasattr(col, 'metadata') else None
                })
            except Exception as e:
                collection_info.append({
                    "name": col.name,
                    "error": f"Could not get details: {str(e)}"
                })
        
        result = {
            "total_collections": len(collections),
            "collections": collection_info,
            "connection_type": "remote" if server_state.chroma_config.use_remote else ("persistent" if server_state.chroma_config.use_persistent else "memory"),
            "status": "success"
        }
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        logger.error(f"Failed to list ChromaDB collections: {str(e)}")
        return json.dumps({
            "error": f"Failed to list collections: {str(e)}",
            "status": "failed"
        })

@mcp.tool()
async def use_external_collection_for_meve(
    collection_name: str,
    chunk_field: str = "document",
    metadata_fields: list = None
) -> str:
    """
    Configure MeVe to use an external ChromaDB collection.
    
    Args:
        collection_name: Name of the ChromaDB collection to use
        chunk_field: Field name containing the document text
        metadata_fields: List of metadata fields to include
    
    Returns:
        JSON string with configuration status
    """
    try:
        if not server_state.external_chroma_client:
            return json.dumps({
                "error": "No external ChromaDB connection. Use connect_to_chromadb first.",
                "status": "failed"
            })
        
        if metadata_fields is None:
            metadata_fields = []
        
        # Get the collection
        collection = server_state.external_chroma_client.get_collection(collection_name)
        
        # Get all documents from the collection
        all_docs = collection.get(include=["documents", "metadatas", "embeddings"])
        
        # Convert to MeVe format
        new_vector_store = {}
        new_bm25_index = {}
        
        documents = all_docs.get("documents", [])
        metadatas = all_docs.get("metadatas", [])
        embeddings = all_docs.get("embeddings", [])
        ids = all_docs.get("ids", [])
        
        for i, doc_id in enumerate(ids):
            if i < len(documents):
                # Extract document content
                if isinstance(documents[i], dict):
                    content = documents[i].get(chunk_field, str(documents[i]))
                else:
                    content = str(documents[i])

                # Convert embedding to list if it's a numpy array (ChromaDB returns numpy arrays)
                embedding = None
                if i < len(embeddings) and embeddings[i] is not None:
                    import numpy as np
                    if isinstance(embeddings[i], np.ndarray):
                        embedding = embeddings[i].tolist()
                    else:
                        embedding = embeddings[i]

                # Create ContextChunk
                chunk = ContextChunk(
                    content=content,
                    doc_id=doc_id,
                    embedding=embedding
                )

                # Add metadata if available
                if i < len(metadatas) and metadatas[i]:
                    chunk.metadata = metadatas[i]

                new_vector_store[doc_id] = chunk
                new_bm25_index[doc_id] = chunk
        
        # Update server state
        server_state.vector_store = new_vector_store
        server_state.bm25_index = new_bm25_index
        server_state.is_initialized = True
        
        result = {
            "status": "success",
            "collection_name": collection_name,
            "documents_loaded": len(new_vector_store),
            "chunk_field": chunk_field,
            "metadata_fields": metadata_fields,
            "has_embeddings": len([e for e in embeddings if e is not None]) > 0,
            "sample_documents": [
                {
                    "doc_id": chunk.doc_id,
                    "content_preview": chunk.content[:100] + "..." if len(chunk.content) > 100 else chunk.content,
                    "has_embedding": chunk.embedding is not None,
                    "metadata": getattr(chunk, 'metadata', None)
                }
                for chunk in list(new_vector_store.values())[:3]
            ]
        }
        
        logger.info(f"Configured MeVe to use ChromaDB collection '{collection_name}' with {len(new_vector_store)} documents")
        return json.dumps(result, indent=2)
        
    except Exception as e:
        logger.error(f"Failed to configure external collection: {str(e)}")
        return json.dumps({
            "error": f"Configuration failed: {str(e)}",
            "collection_name": collection_name,
            "status": "failed"
        })

# ==================== MCP RESOURCES ====================

@mcp.resource("meve://config")
async def get_config() -> str:
    """Get current MeVe pipeline configuration."""
    config_dict = {
        "k_init": server_state.config.k_init,
        "tau_relevance": server_state.config.tau_relevance,
        "n_min": server_state.config.n_min,
        "theta_redundancy": server_state.config.theta_redundancy,
        "t_max": server_state.config.t_max,
        "description": "MeVe pipeline configuration parameters"
    }
    return json.dumps(config_dict, indent=2)

@mcp.resource("meve://stats")
async def get_stats() -> str:
    """Get pipeline statistics and performance metrics."""
    return json.dumps(server_state.pipeline_stats, indent=2)

@mcp.resource("meve://collections")
async def get_collections() -> str:
    """Get information about available document collections."""
    await initialize_meve_data()
    
    collections = {
        "vector_store": {
            "type": "dense_vector_search",
            "chunk_count": len(server_state.vector_store),
            "description": "Vector embeddings for semantic search"
        },
        "bm25_index": {
            "type": "keyword_search",
            "chunk_count": len(server_state.bm25_index),
            "description": "BM25 index for keyword-based fallback retrieval"
        },
        "questions": {
            "type": "evaluation_data",
            "count": len(server_state.questions),
            "description": "Available test questions from HotpotQA dataset"
        }
    }
    
    return json.dumps(collections, indent=2)

# ==================== MCP PROMPTS ====================

@mcp.prompt()
async def rag_query_prompt(query: str) -> str:
    """Generate an effective RAG query prompt using MeVe pipeline."""
    return f"""You are querying a MeVe (Memory-Enhanced Vector) RAG system that uses a sophisticated 5-phase retrieval pipeline:

1. **Phase 1**: Initial kNN vector search to find semantically similar documents
2. **Phase 2**: Cross-encoder verification to filter irrelevant content
3. **Phase 3**: BM25 fallback retrieval if insufficient verified documents
4. **Phase 4**: Context prioritization to reduce redundancy
5. **Phase 5**: Token budgeting to fit within context limits

**Your query**: {query}

To get the best results:
1. Use the `query_with_meve` tool with your query
2. Adjust parameters if needed:
   - `k_init`: Number of initial candidates (default: 20)
   - `tau_relevance`: Relevance threshold (default: 0.5)
   - `n_min`: Minimum verified docs (default: 3)
   - `t_max`: Token budget (default: 512)
3. Analyze the returned context and provide a comprehensive answer
4. Cite specific document IDs from the retrieved chunks

The system is designed for high precision and context efficiency in retrieval."""

@mcp.prompt()
async def analyze_retrieval_prompt(query: str) -> str:
    """Generate a prompt for detailed retrieval analysis."""
    return f"""Analyze the retrieval quality for this query using the MeVe system:

**Query**: {query}

Use the `analyze_retrieval` tool to get a detailed breakdown of each phase:
1. How many initial candidates were found in Phase 1?
2. How many passed relevance verification in Phase 2?
3. Was fallback retrieval triggered in Phase 3?
4. How did Phase 4 prioritization affect the results?
5. What was the final token usage in Phase 5?

Provide insights on:
- Retrieval effectiveness
- Potential improvements
- Configuration adjustments
- Content quality assessment"""

# ==================== SERVER MAIN ====================

def main():
    """Initialize and run the MeVe MCP server."""
    logger.info("Starting MeVe MCP Server...")
    
    # Set server metadata
    mcp.server_info = {
        "name": "MeVe RAG Server",
        "version": "1.0.0",
        "description": "Memory-Enhanced Vector RAG Pipeline MCP Server"
    }
    
    try:
        # Run server with STDIO transport for Claude Desktop integration
        logger.info("MeVe MCP Server running on STDIO transport")
        mcp.run(transport="stdio")
        
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()