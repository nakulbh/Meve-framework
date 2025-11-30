#!/usr/bin/env python3
import os
from typing import Optional

from datasets import load_dataset
from dotenv import load_dotenv

from meve.core.models import ContextChunk
from meve.services.vector_db_client import VectorDBClient
from meve.utils import get_logger

# Load environment variables from .env file
load_dotenv()


def chunk_text(text: str, chunk_size: int = 500) -> list:
    """Split text into chunks."""
    if not text or not text.strip():
        return []

    chunks = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i : i + chunk_size].strip()
        if chunk:  # Only add non-empty chunks
            chunks.append(chunk)
    return chunks


def main():
    """Download BioASQ dataset and add to ChromaDB."""
    logger.info("📥 Loading rag-mini-bioasq dataset...")
    dataset = load_dataset("enelpol/rag-mini-bioasq", name="text-corpus", split="test")
    total_examples = len(dataset)
    logger.info(f"   Found {total_examples} examples in dataset")

    logger.info("🔪 Creating chunks...")
    chunks = []
    skipped_passages = 0

    for idx, example in enumerate(dataset):
        try:
            # Dataset is consistent: always has 'passage' and 'id' keys
            passage_text = example.get("passage", "").strip()
            doc_id = example.get("id", idx)

            if passage_text:
                passage_chunks = chunk_text(passage_text)
                for chunk_idx, chunk_content in enumerate(passage_chunks):
                    chunks.append(
                        ContextChunk(
                            content=chunk_content,
                            doc_id=f"bioasq_{doc_id}_c{chunk_idx}",
                        )
                    )
            else:
                skipped_passages += 1

        except Exception as e:
            logger.warn(f"Error processing passage at index {idx}: {e}")
            continue

        # Progress tracking
        if (idx + 1) % 1000 == 0:
            progress_pct = ((idx + 1) / total_examples) * 100
            logger.info(
                f"  Processed {idx + 1}/{total_examples} passages ({progress_pct:.1f}%) - "
                f"{len(chunks)} chunks created"
            )

    logger.info(
        f"✅ Chunking complete: {len(chunks)} chunks from {total_examples - skipped_passages} "
        f"passages (skipped {skipped_passages} empty passages)"
    )

    chroma_config = {
        "api_key": os.getenv("CHROMA_API_KEY"),
        "tenant": os.getenv("CHROMA_TENANT"),
        "database": os.getenv("CHROMA_DATABASE"),
    }

    # Validate credentials
    if not all(chroma_config.values()):
        missing = [k for k, v in chroma_config.items() if not v]
        raise ValueError(
            f"Missing ChromaDB Cloud credentials: {', '.join(missing)}. "
            f"Please set CHROMA_API_KEY, CHROMA_TENANT, and CHROMA_DATABASE in your .env file."
        )

    logger.info(f"💾 Storing {len(chunks)} chunks in ChromaDB (batch size: 300)...")
    # Store in batches to respect ChromaDB Cloud limits (max 300 per batch)
    batch_size = 300

    # Initialize VectorDBClient once with an empty batch to connect and get collection
    client = VectorDBClient(
        chunks=[],
        is_persistent=True,
        collection_name="bioasq_dataset",
        load_existing=True,
        use_cloud=True,
        cloud_config=chroma_config,
        embedding_model="all-MiniLM-L6-v2",
    )

    # Add chunks in batches using the collection directly
    for batch_idx, i in enumerate(range(0, len(chunks), batch_size)):
        batch = chunks[i : i + batch_size]
        batch_num = batch_idx + 1
        total_batches = (len(chunks) + batch_size - 1) // batch_size
        logger.info(f"  Storing batch {batch_num}/{total_batches} ({len(batch)} chunks)...")

        try:
            # Prepare data for collection.add
            ids = [chunk.doc_id for chunk in batch]
            documents = [chunk.content for chunk in batch]
            embeddings = (
                [chunk.embedding for chunk in batch] if batch and batch[0].embedding else None
            )

            # Add directly to collection
            if embeddings:
                client.collection.add(ids=ids, documents=documents, embeddings=embeddings)
            else:
                client.collection.add(ids=ids, documents=documents)
        except Exception as e:
            logger.error(f"Error storing batch {batch_num}: {e}")
            continue

    logger.info(f"✅ Done! Stored {len(chunks)} chunks in 'bioasq_dataset' collection")


if __name__ == "__main__":
    main()
bioasq_dataset
