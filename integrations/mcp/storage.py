"""
Context Storage Layer for MCP Server

Provides abstract interface and implementations for persisting retrieved context.
Supports multiple backends: in-memory, JSON file-based, etc.
"""

import json
import hashlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime


class StoredContext:
    """Represents a stored context entry with metadata."""

    def __init__(
        self,
        context_id: str,
        query: str,
        context: str,
        timestamp: str = None,
        metadata: Dict[str, Any] = None,
    ):
        """
        Args:
            context_id: Unique identifier for this context
            query: Original query that generated this context
            context: The context string returned by MeVe engine
            timestamp: ISO format timestamp (auto-generated if not provided)
            metadata: Additional metadata (phase stats, chunk count, token count, etc.)
        """
        self.context_id = context_id
        self.query = query
        self.context = context
        self.timestamp = timestamp or datetime.utcnow().isoformat()
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "context_id": self.context_id,
            "query": self.query,
            "context": self.context,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StoredContext":
        """Create from dictionary."""
        return cls(
            context_id=data["context_id"],
            query=data["query"],
            context=data["context"],
            timestamp=data.get("timestamp"),
            metadata=data.get("metadata"),
        )


class ContextStore(ABC):
    """Abstract base class for context storage implementations."""

    @abstractmethod
    def save_context(
        self, query: str, context: str, context_id: str = None, metadata: Dict[str, Any] = None
    ) -> str:
        """
        Save a context retrieval result.

        Args:
            query: The original query
            context: The context string from MeVe engine
            context_id: Optional custom ID; if not provided, generates hash-based ID
            metadata: Optional metadata dict to store with context

        Returns:
            The context_id (either provided or generated)
        """
        pass

    @abstractmethod
    def get_context(self, context_id: str) -> Optional[StoredContext]:
        """
        Retrieve a stored context by ID.

        Args:
            context_id: The context ID to retrieve

        Returns:
            StoredContext object if found, None otherwise
        """
        pass

    @abstractmethod
    def list_contexts(self, limit: int = 100) -> List[StoredContext]:
        """
        List all stored contexts, most recent first.

        Args:
            limit: Maximum number of contexts to return

        Returns:
            List of StoredContext objects
        """
        pass

    @abstractmethod
    def delete_context(self, context_id: str) -> bool:
        """
        Delete a stored context.

        Args:
            context_id: The context ID to delete

        Returns:
            True if deleted, False if not found
        """
        pass

    def _generate_context_id(self, query: str) -> str:
        """Generate a deterministic context ID based on query hash."""
        query_hash = hashlib.sha256(query.encode()).hexdigest()[:12]
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        return f"{timestamp}_{query_hash}"


class InMemoryContextStore(ContextStore):
    """In-memory implementation - good for testing and development."""

    def __init__(self):
        """Initialize in-memory storage."""
        self.store: Dict[str, StoredContext] = {}

    def save_context(
        self, query: str, context: str, context_id: str = None, metadata: Dict[str, Any] = None
    ) -> str:
        """Save context to memory."""
        if context_id is None:
            context_id = self._generate_context_id(query)

        stored = StoredContext(context_id, query, context, metadata=metadata)
        self.store[context_id] = stored
        return context_id

    def get_context(self, context_id: str) -> Optional[StoredContext]:
        """Retrieve context from memory."""
        return self.store.get(context_id)

    def list_contexts(self, limit: int = 100) -> List[StoredContext]:
        """List all contexts, sorted by timestamp descending."""
        sorted_contexts = sorted(self.store.values(), key=lambda x: x.timestamp, reverse=True)
        return sorted_contexts[:limit]

    def delete_context(self, context_id: str) -> bool:
        """Delete context from memory."""
        if context_id in self.store:
            del self.store[context_id]
            return True
        return False


class FileContextStore(ContextStore):
    """File-based implementation using JSON for persistence."""

    def __init__(self, storage_dir: str = "result/mcp_contexts"):
        """
        Initialize file-based storage.

        Args:
            storage_dir: Directory to store context JSON files
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.storage_dir / "index.json"
        self.index = self._load_index()

    def _load_index(self) -> Dict[str, str]:
        """Load the index of all stored contexts."""
        if self.index_file.exists():
            try:
                with open(self.index_file, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {}
        return {}

    def _save_index(self):
        """Persist the index to disk."""
        with open(self.index_file, "w") as f:
            json.dump(self.index, f, indent=2)

    def _get_context_file(self, context_id: str) -> Path:
        """Get the file path for a context ID."""
        # Sanitize context_id for filesystem safety
        safe_id = context_id.replace("/", "_").replace("\\", "_")
        return self.storage_dir / f"{safe_id}.json"

    def save_context(
        self, query: str, context: str, context_id: str = None, metadata: Dict[str, Any] = None
    ) -> str:
        """Save context to JSON file."""
        if context_id is None:
            context_id = self._generate_context_id(query)

        stored = StoredContext(context_id, query, context, metadata=metadata)
        file_path = self._get_context_file(context_id)

        with open(file_path, "w") as f:
            json.dump(stored.to_dict(), f, indent=2)

        # Update index
        self.index[context_id] = str(file_path)
        self._save_index()

        return context_id

    def get_context(self, context_id: str) -> Optional[StoredContext]:
        """Retrieve context from JSON file."""
        file_path = self._get_context_file(context_id)

        if not file_path.exists():
            return None

        try:
            with open(file_path, "r") as f:
                data = json.load(f)
                return StoredContext.from_dict(data)
        except (json.JSONDecodeError, IOError):
            return None

    def list_contexts(self, limit: int = 100) -> List[StoredContext]:
        """List all contexts from disk, sorted by timestamp descending."""
        contexts = []

        for context_id in self.index.keys():
            stored = self.get_context(context_id)
            if stored:
                contexts.append(stored)

        # Sort by timestamp descending
        sorted_contexts = sorted(contexts, key=lambda x: x.timestamp, reverse=True)
        return sorted_contexts[:limit]

    def delete_context(self, context_id: str) -> bool:
        """Delete context file."""
        file_path = self._get_context_file(context_id)

        if file_path.exists():
            try:
                file_path.unlink()
                if context_id in self.index:
                    del self.index[context_id]
                    self._save_index()
                return True
            except IOError:
                return False

        return False
