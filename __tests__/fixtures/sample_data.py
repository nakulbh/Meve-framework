"""Test fixtures for MeVe tests"""

from typing import Dict

import pytest

from meve import ContextChunk, MeVeConfig


@pytest.fixture
def sample_chunks() -> Dict[str, ContextChunk]:
    """Create sample chunks for testing."""
    return {
        "doc1": ContextChunk("The Eiffel Tower is in Paris, France.", "doc1"),
        "doc2": ContextChunk("Paris is the capital of France.", "doc2"),
        "doc3": ContextChunk("The Louvre Museum is in Paris.", "doc3"),
        "doc4": ContextChunk("France is in Western Europe.", "doc4"),
        "doc5": ContextChunk("The Seine River flows through Paris.", "doc5"),
    }


@pytest.fixture
def default_config() -> MeVeConfig:
    """Create default test configuration."""
    return MeVeConfig(k_init=5, tau_relevance=0.3, n_min=2, theta_redundancy=0.85, t_max=100)


@pytest.fixture
def sample_queries() -> list:
    """Sample queries for testing."""
    return [
        "Where is the Eiffel Tower?",
        "What is the capital of France?",
        "Tell me about Paris museums.",
    ]
