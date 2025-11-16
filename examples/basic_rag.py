#!/usr/bin/env python3
"""
Basic RAG System using HotpotQA Data
Simple retrieval-augmented generation without the full MeVe pipeline
"""

import json
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from meve.services.vector_db_client import VectorDBClient
from meve.core.models import ContextChunk
from meve.utils import get_logger

logger = get_logger(__name__)


def load_hotpot_data(data_path: str, max_examples: int = 50) -> List[ContextChunk]:
    """Load HotpotQA data and convert to chunks."""
    logger.info(f"📂 Loading HotpotQA data from {data_path}...")

    with open(data_path, "r") as f:
        data = json.load(f)

    chunks = []
    chunk_id = 0

    for example in data[:max_examples]:
        context = example.get("context", {})
        titles = context.get("title", [])
        sentences_lists = context.get("sentences", [])

        for title, sentences in zip(titles, sentences_lists):
            content = " ".join(sentences)
            if content.strip():
                chunk = ContextChunk(
                    content=content, doc_id=f"doc_{chunk_id}_{title}", embedding=[]
                )
                chunks.append(chunk)
                chunk_id += 1

    logger.success(f"Loaded {len(chunks)} chunks from {max_examples} examples")
    return chunks


def simple_rag(query: str, chunks: List[ContextChunk], top_k: int = 5) -> str:
    """Simple RAG: retrieve top-k chunks and concatenate."""

    # Create vector store
    logger.info(f"\n🔍 Query: {query}")
    logger.info(f"🔎 Retrieving top {top_k} relevant chunks...")

    vector_client = VectorDBClient(chunks)

    # Encode query
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    query_embedding = encoder.encode([query])[0].tolist()

    # Search
    similarities, indices = vector_client.query(query_embedding, top_k)

    # Retrieve chunks
    retrieved = []
    logger.info(f"\n📊 Retrieved chunks:")
    for i, (idx, score) in enumerate(zip(indices, similarities), 1):
        chunk = chunks[idx]
        retrieved.append(chunk)
        logger.debug(f"   {i}. [Score: {score:.3f}] {chunk.content[:100]}...")

    # Concatenate context
    context = "\n\n".join([chunk.content for chunk in retrieved])

    return context


def main():
    """Run basic RAG with HotpotQA data."""

    logger.info("🚀 Basic RAG with HotpotQA Data")
    logger.info("=" * 60)

    # Load data
    chunks = load_hotpot_data("data/hotpot_dev_distractor_v1.json", max_examples=50)

    # Sample queries
    queries = [
        "Who directed the movie that featured the song 'Eye of the Tiger'?",
        "What is the population of the capital of France?",
        "When was the first iPhone released?",
    ]

    # Run RAG for each query
    for i, query in enumerate(queries, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Query {i}/{len(queries)}")
        logger.info(f"{'='*60}")

        context = simple_rag(query, chunks, top_k=5)

        logger.info(f"\n📝 Final Context ({len(context)} characters):")
        logger.debug(f"{context[:500]}...")

    logger.success(f"\n\n🎉 Basic RAG completed!")
    logger.info(f"💡 Retrieved context for {len(queries)} queries from {len(chunks)} chunks")


if __name__ == "__main__":
    main()
