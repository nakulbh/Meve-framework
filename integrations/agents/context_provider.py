"""
MeVe Agent Context Provider

Provides context retrieval capabilities for AI agents using the MeVe framework.
Compatible with LangChain, AutoGPT, and other agent frameworks.
"""

from typing import Dict

from meve import ContextChunk, MeVeConfig, MeVeEngine


class MeVeContextProvider:
    """
    Agent context tool for providing RAG context using MeVe pipeline.

    This can be used as a tool in agent frameworks like LangChain or AutoGPT
    to provide efficient, high-quality context retrieval.
    """

    def __init__(
        self,
        config: MeVeConfig,
        vector_store: Dict[str, ContextChunk],
        bm25_index: Dict[str, ContextChunk],
    ):
        """
        Initialize the context provider.

        Args:
            config: MeVe configuration
            vector_store: Vector database of document chunks
            bm25_index: BM25 index for fallback retrieval
        """
        self.engine = MeVeEngine(config, vector_store, bm25_index)
        self.config = config

    def retrieve_context(self, query: str) -> str:
        """
        Retrieve relevant context for an agent's query.

        Args:
            query: The agent's query

        Returns:
            Optimized context string
        """
        return self.engine.run(query)

    def as_langchain_tool(self):
        """
        Convert to LangChain tool format.

        Returns:
            LangChain Tool instance
        """
        try:
            from langchain.tools import Tool

            return Tool(
                name="MeVe Context Retrieval",
                func=self.retrieve_context,
                description=(
                    "Retrieve relevant context from the knowledge base using "
                    "MeVe's 5-phase pipeline (kNN search, verification, fallback, "
                    "prioritization, and budgeting). Use this when you need "
                    "background information or facts to answer a question."
                ),
            )
        except ImportError:
            raise ImportError("LangChain is not installed. Install it with: pip install langchain")

    def as_autogpt_tool(self) -> Dict:
        """
        Convert to AutoGPT tool format.

        Returns:
            AutoGPT tool dictionary
        """
        return {
            "name": "meve_context_retrieval",
            "description": "Retrieve relevant context using MeVe RAG pipeline",
            "parameters": {
                "query": {
                    "type": "string",
                    "description": "The query to retrieve context for",
                    "required": True,
                }
            },
            "function": self.retrieve_context,
        }

    def get_stats(self) -> Dict[str, any]:
        """
        Get statistics about the context provider.

        Returns:
            Dictionary with provider statistics
        """
        return {
            "config": {
                "k_init": self.config.k_init,
                "tau_relevance": self.config.tau_relevance,
                "n_min": self.config.n_min,
                "t_max": self.config.t_max,
            }
        }


# Example usage
if __name__ == "__main__":
    print("🤖 MeVe Agent Context Provider")
    print("=" * 50)
    print("This module provides context retrieval for AI agents.")
    print("Use MeVeContextProvider to integrate with your agent framework.")
