"""
Vector DB Client Implementation using ChromaDB.
Replaces FAISS operations with ChromaDB for optimized vector similarity search.
"""

from typing import List, Tuple

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from meve.core.models import ContextChunk


class VectorDBClient:
    """
    ChromaDB-powered Vector DB client that replaces FAISS operations.
    Uses ChromaDB for optimized vector similarity search.
    """

    def __init__(self, chunks: List[ContextChunk]):
        """Initialize with chunks and create ChromaDB collection."""
        self.chunks = chunks
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self._setup_chromadb()
        self._populate_collection()

    def _setup_chromadb(self):
        """Initialize ChromaDB client and collection."""
        # Create in-memory ChromaDB client for fast access
        self.chroma_client = chromadb.Client(
            Settings(is_persistent=False, anonymized_telemetry=False)  # In-memory for speed
        )

        # Create or get collection with sentence transformer embedding function
        self.collection = self.chroma_client.get_or_create_collection(
            name="meve_chunks",
            embedding_function=chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            ),
        )

    def _populate_collection(self):
        """Add all chunks to ChromaDB collection."""
        documents = []
        metadatas = []
        ids = []

        for i, chunk in enumerate(self.chunks):
            documents.append(chunk.content)
            metadatas.append({"doc_id": chunk.doc_id, "chunk_index": i})
            ids.append(f"chunk_{i}")

        # Add all documents to collection in batch
        self.collection.add(documents=documents, metadatas=metadatas, ids=ids)

    def query(self, query_vector: List[float], k: int) -> Tuple[List[float], List[int]]:
        """
        ChromaDB-powered .query() method that replaces FAISS index.search().

        Args:
            query_vector: Query embedding vector
            k: Number of similar chunks to retrieve

        Returns:
            Tuple of (similarities, indices) matching FAISS interface
        """
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
