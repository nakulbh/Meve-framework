"""
Basic Usage Example for MeVe Framework
"""

from meve import ContextChunk, MeVeConfig, MeVeEngine


def simple_example():
    """Minimal example showing how to use MeVe."""

    # Create some sample chunks
    chunks = {
        "doc1": ContextChunk("The Eiffel Tower is in Paris, France.", "doc1"),
        "doc2": ContextChunk("Paris is the capital of France.", "doc2"),
        "doc3": ContextChunk("The Louvre Museum is in Paris.", "doc3"),
    }

    # Configure MeVe
    config = MeVeConfig(k_init=5, tau_relevance=0.5, n_min=2, t_max=100)

    # Initialize engine
    engine = MeVeEngine(config, chunks, chunks)

    # Run query
    result = engine.run("Where is the Eiffel Tower?")

    print("Query Result:")
    print(result)


if __name__ == "__main__":
    simple_example()
