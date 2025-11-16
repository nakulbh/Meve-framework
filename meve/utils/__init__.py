"""
MeVe Framework - Utilities Module

Provides logging and other utility functions for the MeVe framework.
"""

from meve.utils.logger import get_logger, MeVeLogger, Logger, logger, log, LogLevel
from meve.utils.metrics import (
    RetrievalMetrics,
    ComparisonMetrics,
    PerformanceTracker,
)
from meve.utils.chromadb_utils import (
    query_chromadb_by_collection_id,
    load_chromadb_collection,
    query_multiple_collections,
)
from meve.utils.model_manager import (
    ModelManager,
    get_sentence_transformer,
    get_cross_encoder,
    get_tokenizer,
)

__all__ = [
    "get_logger",
    "MeVeLogger",
    "Logger",
    "logger",
    "log",
    "LogLevel",
    "RetrievalMetrics",
    "ComparisonMetrics",
    "PerformanceTracker",
    "query_chromadb_by_collection_id",
    "load_chromadb_collection",
    "query_multiple_collections",
    "ModelManager",
    "get_sentence_transformer",
    "get_cross_encoder",
    "get_tokenizer",
]
