"""
MeVe Framework - Multi-phase Efficient Vector Retrieval
A 5-phase pipeline for efficient context retrieval in RAG systems.
"""

__version__ = "0.2.0"

from meve.core.engine import MeVeEngine
from meve.core.models import ContextChunk, MeVeConfig, Query
from meve.utils.logger import get_logger, logger, log, MeVeLogger, Logger, LogLevel

__all__ = [
    "ContextChunk",
    "Query",
    "MeVeConfig",
    "MeVeEngine",
    "get_logger",
    "logger",
    "log",
    "MeVeLogger",
    "Logger",
    "LogLevel",
]
