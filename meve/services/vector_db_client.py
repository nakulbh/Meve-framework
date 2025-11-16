"""
Vector DB Client Implementation using ChromaDB.
Replaces FAISS operations with ChromaDB for optimized vector similarity search.
Supports both local and cloud deployments.
"""

from typing import List, Tuple, Optional

import chromadb
from chromadb.config import Settings

from meve.core.models import ContextChunk
from meve.utils import get_sentence_transformer, get_logger

logger = get_logger(__name__)


class VectorDBClient:
    """
    ChromaDB-powered Vector DB client that replaces FAISS operations.
    Uses ChromaDB for optimized vector similarity search.
    Supports both local and cloud deployments.
    """

    def __init__(
        self,
        chunks: Optional[List[ContextChunk]] = None,
        is_persistent: bool = False,
        collection_name: str = "meve_chunks",
        use_cloud: bool = False,
        cloud_config: Optional[dict] = None,
        load_existing: bool = False,
    ):
        """Initialize with chunks and create/load ChromaDB collection."""
        self.model = get_sentence_transformer()
        self.collection_name = collection_name
        self.is_persistent = is_persistent
        self.use_cloud = use_cloud
        self.cloud_config = cloud_config or {}
        self.load_existing = load_existing

        self._setup_chromadb()

        if load_existing:
            self._load_existing_collection()
        else:
            if chunks is None:
                raise ValueError("chunks must be provided when not loading existing collection")
            self.chunks = chunks
            self._populate_collection()

    def _setup_chromadb(self):
        """Initialize ChromaDB client and collection."""
        if self.use_cloud:
            # Use ChromaDB Cloud
            self.chroma_client = chromadb.CloudClient(
                api_key=self.cloud_config.get("api_key"),
                tenant=self.cloud_config.get("tenant"),
                database=self.cloud_config.get("database"),
            )
        else:
            # Use local ChromaDB client
            self.chroma_client = chromadb.Client(
                Settings(is_persistent=self.is_persistent, anonymized_telemetry=False)
            )

        # Create or get collection with sentence transformer embedding function
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            ),
        )

    def _load_existing_collection(self):
        """Load chunks from existing ChromaDB collection."""
        try:
            # Get all documents from the collection
            results = self.collection.get(include=["documents", "metadatas"])

            self.chunks = []
            for i, (doc_id, document, metadata) in enumerate(
                zip(
                    results.get("ids", []),
                    results.get("documents", []),
                    results.get("metadatas", []),
                )
            ):
                # Create ContextChunk from stored data
                chunk = ContextChunk(
                    content=document,
                    doc_id=metadata.get("doc_id", f"doc_{i}") if metadata else f"doc_{i}",
                )
                self.chunks.append(chunk)

        except Exception as e:
            raise ValueError(f"Failed to load existing collection '{self.collection_name}': {e}")

    def _populate_collection(self):
        """Add all chunks to ChromaDB collection in batches."""
        try:
            batch_size = 5000  # ChromaDB has a limit around 5461, so use smaller batches

            for start_idx in range(0, len(self.chunks), batch_size):
                end_idx = min(start_idx + batch_size, len(self.chunks))
                batch_chunks = self.chunks[start_idx:end_idx]

                documents = []
                metadatas = []
                ids = []

                for i, chunk in enumerate(batch_chunks):
                    documents.append(chunk.content)
                    metadatas.append({"doc_id": chunk.doc_id, "chunk_index": start_idx + i})
                    ids.append(f"chunk_{start_idx + i}")

                # Add batch to collection
                if documents:
                    self.collection.add(documents=documents, metadatas=metadatas, ids=ids)
                    logger.debug(f"Added batch {start_idx // batch_size + 1} with {len(documents)} documents")

            if not self.chunks:
                raise ValueError("No documents to add to collection")

        except Exception as e:
            raise ValueError(f"Failed to populate collection '{self.collection_name}': {e}")

    def query(self, query_vector: List[float], k: int) -> Tuple[List[float], List[int]]:
        """
        ChromaDB-powered .query() method that replaces FAISS index.search().

        Args:
            query_vector: Query embedding vector
            k: Number of similar chunks to retrieve

        Returns:
            Tuple of (similarities, indices) matching FAISS interface

        Raises:
            ValueError: If query fails or no chunks available
        """
        try:
            if not self.chunks:
                raise ValueError("No chunks available in collection")

            # Limit k to available chunks
            k = min(k, len(self.chunks))

            # Query ChromaDB using pre-computed embedding
            results = self.collection.query(query_embeddings=[query_vector], n_results=k)

            # Extract results
            distances = results["distances"][0]  # ChromaDB returns nested lists
            metadatas = results["metadatas"][0]

            # Convert distances to similarities (ChromaDB returns L2 distances)
            # Convert L2 distance to cosine similarity approximation
            similarities = [1.0 / (1.0 + dist) for dist in distances]

            # Extract chunk indices from metadata
            indices = [meta["chunk_index"] for meta in metadatas]

            return similarities, indices

        except Exception as e:
            raise ValueError(f"Query failed: {e}")
