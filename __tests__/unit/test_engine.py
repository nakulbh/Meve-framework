"""Unit tests for MeVe engine"""

from meve import MeVeEngine


def test_engine_initialization(sample_chunks, default_config):
    """Test that engine initializes correctly."""
    engine = MeVeEngine(default_config, sample_chunks, sample_chunks)
    assert engine is not None
    assert engine.config == default_config


def test_engine_run(sample_chunks, default_config):
    """Test basic engine execution."""
    engine = MeVeEngine(default_config, sample_chunks, sample_chunks)
    result = engine.run("Where is the Eiffel Tower?")
    assert isinstance(result, str)
    assert len(result) > 0


def test_engine_with_empty_query(sample_chunks, default_config):
    """Test engine with empty query."""
    engine = MeVeEngine(default_config, sample_chunks, sample_chunks)
    result = engine.run("")
    # Should handle gracefully
    assert isinstance(result, str)
