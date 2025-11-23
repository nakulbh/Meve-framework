"""
HTTP client for MeVe REST API.

Provides convenient Python interface for interacting with the API server.

Usage:
    client = MeVeClient("http://localhost:8000")

    # Health check
    health = client.health()

    # Retrieve context
    result = client.retrieve("What is photosynthesis?")
    print(result.context)
    print(result.metrics)

    # Get metrics
    metrics = client.get_metrics(result.request_id)

    # Update config
    client.update_config(k_init=10, tau_relevance=0.6)
"""

from typing import Optional, Dict, Any
import httpx
import json

from integrations.api.schemas import (
    QueryRequest,
    RetrievalResponse,
    MeVeConfigSchema,
    HealthResponse,
)


class MeVeClient:
    """HTTP client for MeVe REST API."""

    def __init__(self, base_url: str = "http://localhost:8000", timeout: float = 30.0):
        """
        Initialize client.

        Args:
            base_url: API server URL (default: localhost:8000)
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.client = httpx.Client(timeout=timeout)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """Close HTTP client."""
        self.client.close()

    def health(self) -> Dict[str, Any]:
        """
        Check API server health.

        Returns:
            Health status response
        """
        response = self.client.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()

    def status(self) -> Dict[str, Any]:
        """
        Get simplified status (alias for health).

        Returns:
            Status response
        """
        response = self.client.get(f"{self.base_url}/status")
        response.raise_for_status()
        return response.json()

    def retrieve(
        self,
        query: str,
        k_init: Optional[int] = None,
        tau_relevance: Optional[float] = None,
        n_min: Optional[int] = None,
        theta_redundancy: Optional[float] = None,
        lambda_mmr: Optional[float] = None,
        t_max: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Execute retrieval pipeline for a query.

        Args:
            query: User query string
            k_init: Optional override for Phase 1 retrieval count
            tau_relevance: Optional override for Phase 2 threshold
            n_min: Optional override for Phase 3 minimum verified
            theta_redundancy: Optional override for Phase 4 dedup threshold
            lambda_mmr: Optional override for Phase 4 MMR lambda
            t_max: Optional override for Phase 5 token budget

        Returns:
            Retrieval response with context, chunks, and metrics
        """
        request_data: Dict[str, Any] = {"query": query}

        # Add config if any override provided
        if any([k_init, tau_relevance, n_min, theta_redundancy, lambda_mmr, t_max]):
            config = {}
            if k_init is not None:
                config["k_init"] = k_init
            if tau_relevance is not None:
                config["tau_relevance"] = tau_relevance
            if n_min is not None:
                config["n_min"] = n_min
            if theta_redundancy is not None:
                config["theta_redundancy"] = theta_redundancy
            if lambda_mmr is not None:
                config["lambda_mmr"] = lambda_mmr
            if t_max is not None:
                config["t_max"] = t_max
            request_data["config"] = config

        response = self.client.post(
            f"{self.base_url}/retrieve",
            json=request_data,
        )
        response.raise_for_status()
        return response.json()

    def get_metrics(self, request_id: str) -> Dict[str, Any]:
        """
        Get metrics for a previous request.

        Args:
            request_id: Request ID to retrieve metrics for

        Returns:
            Metrics response with phase-by-phase stats
        """
        response = self.client.get(
            f"{self.base_url}/retrieve/metrics/{request_id}",
        )
        response.raise_for_status()
        return response.json()

    def get_config(self) -> Dict[str, Any]:
        """
        Get current pipeline configuration.

        Returns:
            Current MeVeConfig as dictionary
        """
        response = self.client.get(f"{self.base_url}/config")
        response.raise_for_status()
        return response.json()

    def update_config(
        self,
        k_init: Optional[int] = None,
        tau_relevance: Optional[float] = None,
        n_min: Optional[int] = None,
        theta_redundancy: Optional[float] = None,
        lambda_mmr: Optional[float] = None,
        t_max: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Update pipeline configuration.

        Args:
            k_init: Phase 1 retrieval count
            tau_relevance: Phase 2 threshold
            n_min: Phase 3 minimum verified
            theta_redundancy: Phase 4 dedup threshold
            lambda_mmr: Phase 4 MMR lambda
            t_max: Phase 5 token budget

        Returns:
            Updated configuration response
        """
        # Get current config first
        current = self.get_config()

        # Apply overrides
        config = {
            "k_init": k_init if k_init is not None else current["k_init"],
            "tau_relevance": tau_relevance
            if tau_relevance is not None
            else current["tau_relevance"],
            "n_min": n_min if n_min is not None else current["n_min"],
            "theta_redundancy": theta_redundancy
            if theta_redundancy is not None
            else current["theta_redundancy"],
            "lambda_mmr": lambda_mmr if lambda_mmr is not None else current["lambda_mmr"],
            "t_max": t_max if t_max is not None else current["t_max"],
        }

        response = self.client.post(
            f"{self.base_url}/config",
            json={"config": config},
        )
        response.raise_for_status()
        return response.json()

    def reset_config(self) -> Dict[str, Any]:
        """
        Reset configuration to defaults.

        Returns:
            Default configuration response
        """
        response = self.client.post(f"{self.base_url}/config/reset")
        response.raise_for_status()
        return response.json()

    def load_data_source(
        self,
        collection_name: str,
        is_persistent: bool = True,
        load_existing: bool = False,
    ) -> Dict[str, Any]:
        """
        Load or create a ChromaDB collection.

        Args:
            collection_name: Name of collection
            is_persistent: Whether to persist to disk
            load_existing: Load existing collection

        Returns:
            Data source load response
        """
        response = self.client.post(
            f"{self.base_url}/data-sources/load",
            json={
                "config": {
                    "collection_name": collection_name,
                    "is_persistent": is_persistent,
                    "load_existing": load_existing,
                }
            },
        )
        response.raise_for_status()
        return response.json()

    def upload_chunks(self, chunks: list) -> Dict[str, Any]:
        """
        Upload new chunks to data source.

        Args:
            chunks: List of chunk dictionaries with content, doc_id

        Returns:
            Upload response
        """
        response = self.client.post(
            f"{self.base_url}/data-sources/upload",
            json={"chunks": chunks},
        )
        response.raise_for_status()
        return response.json()

    def data_source_info(self) -> Dict[str, Any]:
        """
        Get information about active data sources.

        Returns:
            Data source info (counts, types)
        """
        response = self.client.get(f"{self.base_url}/data-sources/info")
        response.raise_for_status()
        return response.json()

    def list_chunks(
        self,
        doc_id: Optional[str] = None,
        limit: int = 100,
    ) -> Dict[str, Any]:
        """
        List chunks in data source.

        Args:
            doc_id: Optional filter by document ID
            limit: Maximum chunks to return

        Returns:
            List of chunks with metadata
        """
        params = {"limit": limit}
        if doc_id:
            params["doc_id"] = doc_id

        response = self.client.get(
            f"{self.base_url}/data-sources/list",
            params=params,
        )
        response.raise_for_status()
        return response.json()
