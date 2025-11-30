"""
Vector DB Client Implementation using ChromaDB.
Replaces FAISS operations with ChromaDB for optimized vector similarity search.
Supports both local and cloud deployments.
"""

import os
from typing import List, Optional, Tuple

import chromadb
from chromadb.config import Settings

from meve.core.models import ContextChunk
from meve.utils import get_logger, get_sentence_transformer

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
        embedding_model: str = "text-embedding-3-small",
        use_http: bool = False,
        http_host: str = "localhost",
        http_port: int = 8000,
    ):
        """Initialize with chunks and create/load ChromaDB collection.
        
        Args:
            chunks: List of ContextChunk objects to index
            is_persistent: Whether to persist the collection to disk
            collection_name: Name of the ChromaDB collection
            use_cloud: Whether to use ChromaDB Cloud
            cloud_config: Configuration for ChromaDB Cloud
            load_existing: Whether to load an existing collection
            embedding_model: Name of the embedding model to use (default: text-embedding-3-small)
            use_http: Whether to use HTTP client for remote ChromaDB server
            http_host: Host of the remote ChromaDB server (default: localhost)
            http_port: Port of the remote ChromaDB server (default: 8000)
        """
        self.model = get_sentence_transformer(embedding_model)
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        self.is_persistent = is_persistent
        self.use_cloud = use_cloud
        self.cloud_config = cloud_config or {}
        self.load_existing = load_existing
        self.use_http = use_http
        self.http_host = http_host
        self.http_port = http_port

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
            # Use ChromaDB Cloud - support both passed config and environment variables
            api_key = self.cloud_config.get("api_key") or os.getenv("CHROMA_API_KEY")
            tenant = self.cloud_config.get("tenant") or os.getenv("CHROMA_TENANT")
            database = self.cloud_config.get("database") or os.getenv("CHROMA_DATABASE")

            # Validate all required credentials are present
            if not all([api_key, tenant, database]):
                missing = []
                if not api_key:
                    missing.append("api_key (CHROMA_API_KEY)")
                if not tenant:
                    missing.append("tenant (CHROMA_TENANT)")
                if not database:
                    missing.append("database (CHROMA_DATABASE)")
                raise ValueError(
                    f"Missing ChromaDB Cloud credentials: {', '.join(missing)}. "
                    f"Please provide in cloud_config or set environment variables."
                )

            logger.info(f"Connecting to ChromaDB Cloud (tenant: {tenant}, database: {database})")
            self.chroma_client = chromadb.CloudClient(
                api_key=api_key,
                tenant=tenant,
                database=database,
            )
        elif self.use_http:
            # Use HTTP client for remote ChromaDB server
            logger.info(f"Connecting to remote ChromaDB server at {self.http_host}:{self.http_port}")
            self.chroma_client = chromadb.HttpClient(
                host=self.http_host,
                port=self.http_port,
            )
        else:
            # Use local ChromaDB client
            logger.info(
                f"Using local ChromaDB client (persistent: {self.is_persistent})"
            )
            self.chroma_client = chromadb.Client(
                Settings(is_persistent=self.is_persistent, anonymized_telemetry=False)
            )

        # Create or get collection
        logger.info(f"Creating/getting collection: {self.collection_name}")
        
        if self.load_existing:
            # When loading existing collection, don't specify embedding function
            # ChromaDB will use the collection's existing embedding function
            logger.info(f"Loading existing collection (using its persisted embedding function)")
            self.collection = self.chroma_client.get_collection(
                name=self.collection_name
            )
        else:
            # When creating new collection, use the configured embedding model
            embedding_function = chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.embedding_model
            )
            logger.info(f"Creating new collection with {self.embedding_model} embedding function")
            self.collection = self.chroma_client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=embedding_function,
            )

    def _load_existing_collection(self):
        """Load chunks from existing ChromaDB collection."""
        try:
            # Get all documents from the collection
            results = self.collection.get(include=["documents", "metadatas"])

            self.chunks = []
            self._content_to_index = {}  # Map content to chunk index for query results
            
            for i, (document, metadata) in enumerate(
                zip(
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
                # Map content hash to chunk index for query results
                self._content_to_index[hash(document)] = i

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
                    logger.debug(
                        f"Added batch {start_idx // batch_size + 1} with {len(documents)} documents"
                    )

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
            documents = results["documents"][0] if "documents" in results else []

            # Convert distances to similarities (ChromaDB returns L2 distances)
            # Convert L2 distance to cosine similarity approximation
            similarities = [1.0 / (1.0 + dist) for dist in distances]

            # Extract chunk indices by matching documents
            indices = []
            for doc in documents:
                if hasattr(self, '_content_to_index'):
                    # Use content hash mapping for existing collections
                    doc_hash = hash(doc)
                    indices.append(self._content_to_index.get(doc_hash, 0))
                else:
                    # For new collections, find by direct comparison
                    for i, chunk in enumerate(self.chunks):
                        if chunk.content == doc:
                            indices.append(i)
                            break
                    else:
                        indices.append(0)

            return similarities, indices

        except Exception as e:
            raise ValueError(f"Query failed: {e}")

    def query_text(self, query_texts: List[str], k: int) -> Tuple[List[float], List[int]]:
        """
        Text-based similarity search using ChromaDB's built-in embedding.
        Let ChromaDB handle both embedding and search in one step.
        Supports batch queries with multiple texts.

        Args:
            query_texts: List of raw query texts (not pre-encoded).
                         Example: ["thus spake zarathustra", "the oracle speaks"]
            k: Number of similar chunks to retrieve

        Returns:
            Tuple of (similarities, indices) matching FAISS interface.
            For multiple queries, returns aggregated top-k results across all queries.

        Raises:
            ValueError: If query fails or no chunks available
        """
        try:
            if not self.chunks:
                raise ValueError("No chunks available in collection")

            # Ensure query_texts is a list
            if isinstance(query_texts, str):
                query_texts = [query_texts]

            # Limit k to available chunks
            k = min(k, len(self.chunks))

            # Query ChromaDB with raw texts - it handles embedding internally
            # Returns: {
            #   "distances": [[dist1, dist2, ...], [dist1, dist2, ...], ...],
            #   "documents": [[doc1, doc2, ...], [doc1, doc2, ...], ...],
            #   "metadatas": [[meta1, meta2, ...], [meta1, meta2, ...], ...],
            #   "ids": [[id1, id2, ...], [id1, id2, ...], ...]
            # }
            results = self.collection.query(query_texts=query_texts, n_results=k)

            # Aggregate results from all query texts, removing duplicates
            # Keep track of highest similarity for each chunk
            chunk_best_scores = {}  # chunk_index -> best_similarity

            for query_distances, query_docs in zip(results["distances"], results.get("documents", [])):
                # Convert distances to similarities (ChromaDB returns L2 distances)
                # Convert L2 distance to similarity approximation
                for dist, doc in zip(query_distances, query_docs):
                    similarity = 1.0 / (1.0 + dist)

                    # Find chunk index
                    chunk_idx = self._find_chunk_index(doc)

                    # Keep highest similarity score for this chunk
                    if chunk_idx not in chunk_best_scores or chunk_best_scores[chunk_idx] < similarity:
                        chunk_best_scores[chunk_idx] = similarity

            # Sort by similarity descending and keep top k
            sorted_items = sorted(
                chunk_best_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )[:k]

            if sorted_items:
                indices, similarities = zip(*sorted_items)
                return list(similarities), list(indices)
            else:
                return [], []

        except Exception as e:
            raise ValueError(f"Text query failed: {e}")

    def _find_chunk_index(self, content: str) -> int:
        """
        Find the index of a chunk by its content.

        Args:
            content: The chunk content to search for

        Returns:
            Index of the chunk in self.chunks, or 0 if not found
        """
        if hasattr(self, '_content_to_index'):
            # Use content hash mapping for existing collections
            doc_hash = hash(content)
            return self._content_to_index.get(doc_hash, 0)
        else:
            # For new collections, find by direct comparison
            for i, chunk in enumerate(self.chunks):
                if chunk.content == content:
                    return i
            return 0
