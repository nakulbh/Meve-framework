#!/usr/bin/env python3
"""
Script to load Wikipedia physics data into ChromaDB in vector format.

This script:
1. Loads Wikipedia physics articles from JSON
2. Chunks the text into manageable pieces
3. Creates embeddings using sentence transformers
4. Stores everything in a persistent ChromaDB collection

Usage:
    python scripts/add_wikipedia_to_chromadb.py [--collection-name NAME] [--chunk-size SIZE]
"""

import json
import argparse
from pathlib import Path
from typing import List

from meve.core.models import ContextChunk
from meve.services.vector_db_client import VectorDBClient
from meve.utils import get_logger

logger = get_logger(__name__)


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Split text into overlapping chunks.

    Args:
        text: Text to chunk
        chunk_size: Maximum characters per chunk
        overlap: Number of overlapping characters between chunks

    Returns:
        List of text chunks
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        # Try to break at sentence boundary
        if end < len(text):
            # Look for sentence end (., !, ?) within the last 100 chars
            last_period = text.rfind(".", start, end)
            last_exclaim = text.rfind("!", start, end)
            last_question = text.rfind("?", start, end)

            sentence_end = max(last_period, last_exclaim, last_question)

            if sentence_end > start + (chunk_size * 0.5):  # Only if we found a good break point
                end = sentence_end + 1

        chunks.append(text[start:end].strip())
        start = end - overlap

    return chunks


def load_wikipedia_data(json_path: Path) -> List[dict]:
    """
    Load Wikipedia articles from JSON file.

    Args:
        json_path: Path to JSON file

    Returns:
        List of article dictionaries
    """
    logger.info(f"📖 Loading data from {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    logger.info(f"✅ Loaded {len(data)} articles")
    return data


def create_chunks_from_articles(
    articles: List[dict], chunk_size: int = 500, overlap: int = 50
) -> List[ContextChunk]:
    """
    Convert Wikipedia articles into ContextChunks.

    Args:
        articles: List of article dictionaries with 'id', 'title', 'text'
        chunk_size: Maximum characters per chunk
        overlap: Overlapping characters between chunks

    Returns:
        List of ContextChunk objects
    """
    logger.info("🔪 Chunking articles into smaller pieces...")

    all_chunks = []

    for article in articles:
        article_id = article.get("id", "unknown")
        title = article.get("title", "Unknown")
        text = article.get("text", "")

        if not text:
            logger.warn(f"Skipping empty article: {title}")
            continue

        # Add title as context to each chunk
        full_text = f"Article: {title}\n\n{text}"

        # Split into chunks
        text_chunk_list = chunk_text(full_text, chunk_size, overlap)

        # Create ContextChunk objects
        for i, chunk_content in enumerate(text_chunk_list):
            chunk = ContextChunk(content=chunk_content, doc_id=f"{article_id}_chunk_{i}")
            all_chunks.append(chunk)

        logger.debug(f"  {title}: {len(text_chunk_list)} chunks")

    logger.info(f"✅ Created {len(all_chunks)} total chunks from {len(articles)} articles")
    return all_chunks


def store_in_chromadb(
    chunks: List[ContextChunk], collection_name: str = "wik_dataset", persistent: bool = True
):
    """
    Store chunks in ChromaDB collection.

    Args:
        chunks: List of ContextChunk objects
        collection_name: Name for the ChromaDB collection
        persistent: Whether to persist the collection to disk
    """
    logger.info(f"💾 Storing {len(chunks)} chunks in ChromaDB collection '{collection_name}'...")

    try:
        # Create VectorDBClient with persistent storage
        vector_db = VectorDBClient(
            chunks=chunks,
            is_persistent=persistent,
            collection_name=collection_name,
            load_existing=False,
        )

        logger.info(f"✅ Successfully stored all chunks in collection '{collection_name}'")
        logger.info(f"   Collection is {'persistent' if persistent else 'in-memory only'}")

        # Test query to verify
        test_query = "What is physics?"
        logger.info(f"\n🔍 Testing with query: '{test_query}'")

        # Get embedding for test query
        from meve.utils import get_sentence_transformer

        model = get_sentence_transformer()
        query_embedding = model.encode([test_query])[0].tolist()

        # Query the collection
        similarities, indices = vector_db.query(query_embedding, k=3)

        logger.info(f"\n📊 Top 3 results:")
        for i, (sim, idx) in enumerate(zip(similarities, indices), 1):
            chunk = chunks[idx]
            preview = chunk.content[:100].replace("\n", " ")
            logger.info(f"  {i}. [Score: {sim:.4f}] {preview}...")

    except Exception as e:
        logger.error(f"❌ Failed to store chunks: {e}")
        raise


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Load Wikipedia physics data into ChromaDB")
    parser.add_argument(
        "--data-file",
        type=Path,
        default=Path("data/wikipedia_en_science_physics_10.json"),
        help="Path to Wikipedia JSON file",
    )
    parser.add_argument(
        "--collection-name", type=str, default="wik_dataset", help="Name for ChromaDB collection"
    )
    parser.add_argument("--chunk-size", type=int, default=500, help="Maximum characters per chunk")
    parser.add_argument(
        "--overlap", type=int, default=50, help="Overlapping characters between chunks"
    )
    parser.add_argument(
        "--no-persist",
        action="store_true",
        help="Don't persist collection to disk (in-memory only)",
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Wikipedia to ChromaDB Converter")
    logger.info("=" * 60)

    # Validate data file exists
    if not args.data_file.exists():
        logger.error(f"❌ Data file not found: {args.data_file}")
        logger.info("💡 Make sure to run: make download-data")
        return 1

    try:
        # Load articles
        articles = load_wikipedia_data(args.data_file)

        # Create chunks
        chunks = create_chunks_from_articles(
            articles, chunk_size=args.chunk_size, overlap=args.overlap
        )

        # Store in ChromaDB
        store_in_chromadb(
            chunks, collection_name=args.collection_name, persistent=not args.no_persist
        )

        logger.info("\n" + "=" * 60)
        logger.info("✅ SUCCESS! Data is now in ChromaDB")
        logger.info("=" * 60)
        logger.info(f"\n📝 Collection name: {args.collection_name}")
        logger.info(f"📊 Total chunks: {len(chunks)}")
        logger.info(f"💾 Persistent: {not args.no_persist}")

        return 0

    except Exception as e:
        logger.error(f"\n❌ Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
