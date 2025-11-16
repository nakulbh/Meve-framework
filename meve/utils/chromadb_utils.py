"""
ChromaDB Utility Functions

Provides utility functions for querying and managing ChromaDB collections.
"""

from typing import List, Dict, Tuple, Optional
import chromadb
from sentence_transformers import SentenceTransformer

from meve.core.models import ContextChunk
from meve.utils.logger import get_logger

logger = get_logger(__name__)


def query_chromadb_by_collection_id(
    collection_id: str,
    query: str,
    cloud_config: Dict[str, str],
    top_k: int = 5,
    return_chunks: bool = True,
    query_embedding: Optional[List[float]] = None,
    embedding_model: str = "text-embedding-3-small"
) -> Tuple[List[ContextChunk], List[float]]:
    """
    Query a ChromaDB Cloud collection by its ID.

    Args:
        collection_id: The ChromaDB collection ID/name to query
        query: The query string to search for
        cloud_config: Dictionary with 'api_key', 'tenant', and 'database' keys
        top_k: Number of results to return (default: 5)
        return_chunks: If True, return ContextChunk objects; if False, return raw results
        query_embedding: Pre-computed query embedding vector (optional)
                        If None, will use OpenAI embedding model
        embedding_model: OpenAI embedding model to use (default: "text-embedding-3-small")
                        Only used if query_embedding is None

    Returns:
        Tuple of (chunks, scores) where:
            - chunks: List of ContextChunk objects with retrieved content
            - scores: List of relevance scores (similarity scores)

    Example:
        >>> cloud_config = {
        ...     'api_key': 'your-api-key',
        ...     'tenant': 'your-tenant-id',
        ...     'database': 'your-database'
        ... }
        >>> chunks, scores = query_chromadb_by_collection_id(
        ...     collection_id="YfIDqNjAwpUJQbQWQZtiOdElsm62",
        ...     query="What is machine learning?",
        ...     cloud_config=cloud_config,
        ...     top_k=5
        ... )
    """
    try:
        # Connect to ChromaDB Cloud
        logger.info(f"🔄 Connecting to ChromaDB Cloud collection: {collection_id}")
        chroma_client = chromadb.CloudClient(
            api_key=cloud_config.get('api_key'),
            tenant=cloud_config.get('tenant'),
            database=cloud_config.get('database')
        )

        # Get the collection without embedding function
        collection = chroma_client.get_collection(name=collection_id)

        # If no pre-computed embedding provided, compute it using OpenAI
        if query_embedding is None:
            try:
                import openai
                import os
                
                # Use OpenAI API to get embedding
                openai_key = cloud_config.get('openai_api_key') or os.environ.get('OPENAI_API_KEY')
                if not openai_key:
                    raise ValueError("OpenAI API key required. Provide via cloud_config['openai_api_key'] or OPENAI_API_KEY env var")
                
                client = openai.OpenAI(api_key=openai_key)
                response = client.embeddings.create(
                    input=query,
                    model=embedding_model
                )
                query_embedding = response.data[0].embedding
                logger.info(f"✅ Generated query embedding using {embedding_model}")
            except Exception as e:
                logger.error(f"❌ Failed to generate embedding: {e}")
                raise

        # Query using pre-computed embedding
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=['documents', 'metadatas', 'distances']
        )

        # Extract results
        documents = results.get('documents', [[]])[0]
        metadatas = results.get('metadatas', [[]])[0]
        distances = results.get('distances', [[]])[0]

        # Convert distances to similarity scores
        # ChromaDB returns L2 distances, convert to similarity (0-1 range)
        similarities = [1.0 / (1.0 + dist) for dist in distances]

        if return_chunks:
            # Create ContextChunk objects
            chunks = []
            for i, (doc, metadata, score) in enumerate(zip(documents, metadatas, similarities)):
                chunk = ContextChunk(
                    content=doc,
                    doc_id=metadata.get('doc_id', f'doc_{i}')
                )
                chunk.relevance_score = score
                chunks.append(chunk)

            logger.info(f"✅ Retrieved {len(chunks)} chunks from collection {collection_id}")
            return chunks, similarities
        else:
            # Return raw results
            return documents, similarities

    except Exception as e:
        logger.error(f"❌ Error querying ChromaDB collection {collection_id}: {e}")
        raise


def load_chromadb_collection(
    collection_id: str,
    cloud_config: Dict[str, str],
    max_chunks: Optional[int] = None
) -> List[ContextChunk]:
    """
    Load all chunks from a ChromaDB Cloud collection.

    Args:
        collection_id: The ChromaDB collection ID/name
        cloud_config: Dictionary with 'api_key', 'tenant', and 'database' keys
        max_chunks: Optional limit on number of chunks to load

    Returns:
        List of ContextChunk objects

    Example:
        >>> cloud_config = {
        ...     'api_key': 'your-api-key',
        ...     'tenant': 'your-tenant-id',
        ...     'database': 'your-database'
        ... }
        >>> chunks = load_chromadb_collection(
        ...     collection_id="YfIDqNjAwpUJQbQWQZtiOdElsm62",
        ...     cloud_config=cloud_config,
        ...     max_chunks=100
        ... )
    """
    try:
        logger.info(f"🔄 Loading ChromaDB Cloud collection: {collection_id}")
        
        # Connect to ChromaDB Cloud
        chroma_client = chromadb.CloudClient(
            api_key=cloud_config.get('api_key'),
            tenant=cloud_config.get('tenant'),
            database=cloud_config.get('database')
        )

        # Get the collection
        collection = chroma_client.get_collection(name=collection_id)

        # Get all documents from the collection
        results = collection.get(include=['documents', 'metadatas'])

        # Create ContextChunk objects
        chunks = []
        documents = results.get('documents', [])
        metadatas = results.get('metadatas', [])

        # Limit chunks if max_chunks specified
        limit = min(len(documents), max_chunks) if max_chunks else len(documents)

        for i in range(limit):
            doc = documents[i]
            metadata = metadatas[i] if i < len(metadatas) else {}
            
            chunk = ContextChunk(
                content=doc,
                doc_id=metadata.get('doc_id', f'doc_{i}')
            )
            chunks.append(chunk)

        logger.info(f"✅ Loaded {len(chunks)} chunks from collection {collection_id}")
        return chunks

    except Exception as e:
        logger.error(f"❌ Error loading ChromaDB collection {collection_id}: {e}")
        raise


def query_multiple_collections(
    collection_ids: List[str],
    query: str,
    cloud_config: Dict[str, str],
    top_k_per_collection: int = 3
) -> Dict[str, Tuple[List[ContextChunk], List[float]]]:
    """
    Query multiple ChromaDB collections with the same query.

    Args:
        collection_ids: List of collection IDs to query
        query: The query string
        cloud_config: Dictionary with 'api_key', 'tenant', and 'database' keys
        top_k_per_collection: Number of results to get from each collection

    Returns:
        Dictionary mapping collection_id to (chunks, scores) tuples

    Example:
        >>> results = query_multiple_collections(
        ...     collection_ids=["collection1", "collection2"],
        ...     query="machine learning",
        ...     cloud_config=cloud_config,
        ...     top_k_per_collection=5
        ... )
    """
    results = {}
    
    for collection_id in collection_ids:
        try:
            chunks, scores = query_chromadb_by_collection_id(
                collection_id=collection_id,
                query=query,
                cloud_config=cloud_config,
                top_k=top_k_per_collection
            )
            results[collection_id] = (chunks, scores)
        except Exception as e:
            logger.warning(f"⚠️  Failed to query collection {collection_id}: {e}")
            results[collection_id] = ([], [])
    
    return results
