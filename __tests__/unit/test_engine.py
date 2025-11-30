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
    assert isinstance(result, tuple)
    assert len(result) == 2
    final_context, final_chunks = result
    assert isinstance(final_context, str)
    assert isinstance(final_chunks, list)


def test_engine_with_empty_query(sample_chunks, default_config):
    """Test engine with empty query."""
    engine = MeVeEngine(default_config, sample_chunks, sample_chunks)
    result = engine.run("")
    # Should handle gracefully and return tuple
    assert isinstance(result, tuple)
    assert len(result) == 2
    final_context, final_chunks = result
    assert isinstance(final_context, str)
    assert isinstance(final_chunks, list)
