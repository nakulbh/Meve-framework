"""
MeVe MCP Server - Model Context Protocol Integration

This MCP server provides context retrieval capabilities using the MeVe framework.
It can be used by AI agents to retrieve relevant context for their queries.
"""

from typing import Any, Dict

from meve import ContextChunk, MeVeConfig, MeVeEngine


class MeVeContextServer:
    """
    MCP Server for providing RAG context using MeVe pipeline.

    This server exposes the MeVe framework as an MCP tool that can be
    used by AI agents for efficient context retrieval.
    """

    def __init__(
        self,
        config: MeVeConfig,
        vector_store: Dict[str, ContextChunk],
        bm25_index: Dict[str, ContextChunk],
    ):
        """
        Initialize the MeVe MCP Server.

        Args:
            config: MeVe configuration
            vector_store: Vector database of document chunks
            bm25_index: BM25 index for fallback retrieval
        """
        self.engine = MeVeEngine(config, vector_store, bm25_index)
        self.config = config

    def get_context(self, query: str) -> str:
        """
        Retrieve relevant context for a query using MeVe pipeline.

        Args:
            query: The user's query

        Returns:
            Relevant context string optimized for LLM consumption
        """
        return self.engine.run(query)

    def get_tool_definition(self) -> Dict[str, Any]:
        """
        Return MCP tool definition for this server.

        Returns:
            Tool definition dictionary compatible with MCP protocol
        """
        return {
            "name": "meve_context_retrieval",
            "description": "Retrieve relevant context using MeVe's 5-phase RAG pipeline",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The query to retrieve context for"}
                },
                "required": ["query"],
            },
        }

    def handle_request(self, tool_name: str, parameters: Dict[str, Any]) -> str:
        """
        Handle MCP tool request.

        Args:
            tool_name: Name of the tool being invoked
            parameters: Parameters for the tool

        Returns:
            Tool execution result
        """
        if tool_name == "meve_context_retrieval":
            query = parameters.get("query", "")
            return self.get_context(query)
        else:
            raise ValueError(f"Unknown tool: {tool_name}")


def create_meve_mcp_server(
    vector_store: Dict[str, ContextChunk],
    bm25_index: Dict[str, ContextChunk],
    config: MeVeConfig = None,
) -> MeVeContextServer:
    """
    Factory function to create a MeVe MCP server.

    Args:
        vector_store: Vector database of document chunks
        bm25_index: BM25 index for fallback retrieval
        config: Optional MeVe configuration (uses default if not provided)

    Returns:
        Initialized MeVe MCP server
    """
    if config is None:
        config = MeVeConfig()

    return MeVeContextServer(config, vector_store, bm25_index)


# Example usage
if __name__ == "__main__":
    print("🔧 MeVe MCP Server")
    print("=" * 50)
    print("This server provides context retrieval via MCP protocol.")
    print("Use create_meve_mcp_server() to initialize the server.")
