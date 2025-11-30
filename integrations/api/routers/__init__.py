"""
Router initialization for all API endpoints.
"""

from integrations.api.routers import retrieval, health, config, data_sources

__all__ = [
    "retrieval",
    "health",
    "config",
    "data_sources",
]
