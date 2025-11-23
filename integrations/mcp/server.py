"""
MeVe MCP Server - Model Context Protocol Integration

This MCP server provides context retrieval capabilities using the MeVe framework.
It can be used by AI agents to retrieve relevant context for their queries.
"""

from typing import Any, Dict, List, Optional

from meve import ContextChunk, MeVeConfig, MeVeEngine
from .storage import ContextStore, FileContextStore, InMemoryContextStore, StoredContext


class MeVeContextServer:
    """
    MCP Server for providing RAG context using MeVe pipeline.

    This server exposes the MeVe framework as an MCP tool that can be
    used by AI agents for efficient context retrieval with context persistence.
    """

    def __init__(
        self,
        config: MeVeConfig,
        vector_store: Dict[str, ContextChunk] = None,
        bm25_index: Dict[str, ContextChunk] = None,
        storage: ContextStore = None,
        vector_db_client=None,
    ):
        """
        Initialize the MeVe MCP Server.

        Args:
            config: MeVe configuration
            vector_store: Vector database of document chunks (can be None if vector_db_client provided)
            bm25_index: BM25 index for fallback retrieval (can be None if vector_db_client provided)
            storage: Optional context storage backend (uses InMemoryContextStore if not provided)
            vector_db_client: Optional pre-initialized VectorDBClient for ChromaDB
        """
        # Use provided clients or fall back to dicts
        if vector_db_client is not None:
            self.engine = MeVeEngine(
                config,
                vector_store=vector_store or {},
                bm25_index=bm25_index or {},
                vector_db_client=vector_db_client,
            )
        else:
            self.engine = MeVeEngine(config, vector_store, bm25_index)

        self.config = config
        self.storage = storage or InMemoryContextStore()

    def get_context(self, query: str) -> str:
        """
        Retrieve relevant context for a query using MeVe pipeline.

        Args:
            query: The user's query

        Returns:
            Relevant context string optimized for LLM consumption
        """
        return self.engine.run(query)

    def save_context(
        self,
        query: str,
        context: str = None,
        context_id: str = None,
        metadata: Dict[str, Any] = None,
    ) -> Dict[str, str]:
        """
        Save a context retrieval result.

        If context is not provided, runs the query through the engine first.

        Args:
            query: The original query
            context: Optional pre-retrieved context string (if not provided, will call engine.run())
            context_id: Optional custom ID for the context
            metadata: Optional metadata to store with context

        Returns:
            Dictionary with 'context_id' and 'message'
        """
        # If context not provided, run query through engine
        if context is None:
            context = self.engine.run(query)

        # Save to storage
        saved_id = self.storage.save_context(query, context, context_id, metadata)

        return {"context_id": saved_id, "message": f"Context saved with ID: {saved_id}"}

    def get_saved_context(self, context_id: str) -> Dict[str, Any]:
        """
        Retrieve a previously saved context.

        Args:
            context_id: The ID of the context to retrieve

        Returns:
            Dictionary with context data or error message
        """
        stored = self.storage.get_context(context_id)

        if stored is None:
            return {"error": f"Context not found: {context_id}"}

        return {
            "context_id": stored.context_id,
            "query": stored.query,
            "context": stored.context,
            "timestamp": stored.timestamp,
            "metadata": stored.metadata,
        }

    def list_saved_contexts(self, limit: int = 10) -> Dict[str, Any]:
        """
        List all saved contexts.

        Args:
            limit: Maximum number of contexts to return

        Returns:
            Dictionary with list of context summaries
        """
        contexts = self.storage.list_contexts(limit)

        return {
            "count": len(contexts),
            "contexts": [
                {
                    "context_id": c.context_id,
                    "query": c.query,
                    "timestamp": c.timestamp,
                }
                for c in contexts
            ],
        }

    def delete_saved_context(self, context_id: str) -> Dict[str, str]:
        """
        Delete a saved context.

        Args:
            context_id: The ID of the context to delete

        Returns:
            Dictionary with success/error message
        """
        deleted = self.storage.delete_context(context_id)

        if deleted:
            return {"message": f"Context deleted: {context_id}"}
        else:
            return {"error": f"Context not found: {context_id}"}

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

    def get_all_tools(self) -> List[Dict[str, Any]]:
        """
        Return all available MCP tool definitions.

        Returns:
            List of tool definition dictionaries
        """
        return [
            {
                "name": "meve_get_context",
                "description": "Retrieve relevant context using MeVe's 5-phase RAG pipeline",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The query to retrieve context for",
                        }
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "meve_save_context",
                "description": "Save a context retrieval result for later retrieval",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The original query"},
                        "context": {
                            "type": "string",
                            "description": "Optional pre-retrieved context (will run query if not provided)",
                        },
                        "context_id": {
                            "type": "string",
                            "description": "Optional custom ID for the context",
                        },
                        "metadata": {
                            "type": "object",
                            "description": "Optional metadata to store with context",
                        },
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "meve_get_saved_context",
                "description": "Retrieve a previously saved context by ID",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "context_id": {
                            "type": "string",
                            "description": "The ID of the saved context",
                        }
                    },
                    "required": ["context_id"],
                },
            },
            {
                "name": "meve_list_saved_contexts",
                "description": "List all saved contexts",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of contexts to return",
                            "default": 10,
                        }
                    },
                },
            },
            {
                "name": "meve_delete_saved_context",
                "description": "Delete a saved context by ID",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "context_id": {
                            "type": "string",
                            "description": "The ID of the context to delete",
                        }
                    },
                    "required": ["context_id"],
                },
            },
        ]

    def handle_request(self, tool_name: str, parameters: Dict[str, Any]) -> Any:
        """
        Handle MCP tool request.

        Args:
            tool_name: Name of the tool being invoked
            parameters: Parameters for the tool

        Returns:
            Tool execution result
        """
        if tool_name == "meve_context_retrieval" or tool_name == "meve_get_context":
            query = parameters.get("query", "")
            return self.get_context(query)
        elif tool_name == "meve_save_context":
            query = parameters.get("query", "")
            context = parameters.get("context")
            context_id = parameters.get("context_id")
            metadata = parameters.get("metadata")
            return self.save_context(query, context, context_id, metadata)
        elif tool_name == "meve_get_saved_context":
            context_id = parameters.get("context_id", "")
            return self.get_saved_context(context_id)
        elif tool_name == "meve_list_saved_contexts":
            limit = parameters.get("limit", 10)
            return self.list_saved_contexts(limit)
        elif tool_name == "meve_delete_saved_context":
            context_id = parameters.get("context_id", "")
            return self.delete_saved_context(context_id)
        else:
            raise ValueError(f"Unknown tool: {tool_name}")


def create_meve_mcp_server(
    vector_store: Dict[str, ContextChunk] = None,
    bm25_index: Dict[str, ContextChunk] = None,
    config: MeVeConfig = None,
    storage: ContextStore = None,
    use_file_storage: bool = False,
    storage_dir: str = "result/mcp_contexts",
    vector_db_client=None,
) -> MeVeContextServer:
    """
    Factory function to create a MeVe MCP server.

    Args:
        vector_store: Vector database of document chunks (optional if vector_db_client provided)
        bm25_index: BM25 index for fallback retrieval (optional if vector_db_client provided)
        config: Optional MeVe configuration (uses default if not provided)
        storage: Optional pre-initialized storage backend
        use_file_storage: If True, uses FileContextStore; otherwise uses InMemoryContextStore
        storage_dir: Directory for file-based storage (only used if use_file_storage=True)
        vector_db_client: Optional pre-initialized VectorDBClient for ChromaDB

    Returns:
        Initialized MeVe MCP server
    """
    if config is None:
        config = MeVeConfig()

    # Initialize storage if not provided
    if storage is None:
        if use_file_storage:
            storage = FileContextStore(storage_dir)
        else:
            storage = InMemoryContextStore()

    return MeVeContextServer(config, vector_store, bm25_index, storage, vector_db_client)


# Example usage
if __name__ == "__main__":
    print("🔧 MeVe MCP Server")
    print("=" * 50)
    print("This server provides context retrieval via MCP protocol.")
    print("Use create_meve_mcp_server() to initialize the server.")
