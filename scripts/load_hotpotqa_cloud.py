"""
Load HotpotQA data to ChromaDB Cloud.

Each *sentence* from each context document becomes its own chunk.

Chunk ID format:
    {question_idx}_{doc_idx}_{sent_idx}
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import List, Optional

import chromadb
from dotenv import load_dotenv

from meve.core.models import ContextChunk
from meve.utils import get_logger

# Load env variables
load_dotenv()

# Global logger
logger = get_logger(__name__)


# ============================================================
# Load HotpotQA JSON
# ============================================================

def load_hotpotqa_data(data_file: str) -> List[dict]:
    with open(data_file, "r") as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data)} questions from {data_file}")
    return data


# ============================================================
# Create sentence-level chunks
# ============================================================


def create_chunks_from_hotpotqa(data: List[dict]) -> List[ContextChunk]:
    """
    Convert HotpotQA data into chunks where **each context.sentences[n] is ONE document**.

    Example:
        context.sentences = [
            ["s1", "s2", "s3"],      # → 1 doc
            ["a", "b"],              # → 1 doc
            ["x", "y", "z", "q"],    # → 1 doc
        ]

    Chunk ID:
        qIdx_docNumber  (1-based)
    """

    chunks = []
    logger.info(f"Starting to process {len(data)} questions...")

    for q_idx, question_obj in enumerate(data):

        if q_idx % 20 == 0:
            logger.info(f"Processing question {q_idx}/{len(data)}...")

        context = question_obj.get("context", {})
        titles = context.get("title", [])
        documents = context.get("sentences", [])

        if not titles or not documents:
            logger.warning(f"Question {q_idx} has no context — skipping.")
            continue

        supporting_titles = question_obj.get("supporting_facts", {}).get("title", [])

        # Loop each document EXACTLY as context.sentences[n]
        for doc_idx, (title, sentence_list) in enumerate(zip(titles, documents)):

            # Combine all sentences into a SINGLE document
            combined_text = "\n".join(sentence_list)

            # 1-based doc index for ID
            doc_number = doc_idx + 1
            chunk_id = f"{q_idx}_{doc_number}"

            # Supporting-doc detection
            is_support = title in supporting_titles

            metadata = {
                "question_idx": q_idx,
                "doc_number": doc_number,
                "title": title,
                "question": question_obj.get("question", ""),
                "answer": question_obj.get("answer", ""),
                "question_type": question_obj.get("type", ""),
                "difficulty": question_obj.get("level", ""),
                "sentence_count": len(sentence_list),
                "is_supporting_doc": is_support,
            }

            chunk = ContextChunk(
                content=combined_text,  # WHOLE block of sentences
                doc_id=chunk_id,
                title=title,
                question_id=str(q_idx),
                source_type="hotpotqa_doc",
                metadata=metadata
            )

            chunks.append(chunk)

    logger.info(f"✅ Created {len(chunks)} document-level chunks")
    return chunks

# ============================================================
# Upload to ChromaDB Cloud
# ============================================================

def load_to_chroma_cloud(
    chunks: List[ContextChunk],
    collection_name: str,
    api_key: Optional[str] = None,
    tenant: Optional[str] = None,
    database: Optional[str] = None,
    embedding_model: str = "all-MiniLM-L6-v2",
    batch_size: int = 300,
    clear_collection: bool = False,
) -> chromadb.Collection:

    # Resolve credentials
    api_key = api_key or os.getenv("CHROMA_API_KEY")
    tenant = tenant or os.getenv("CHROMA_TENANT")
    database = database or os.getenv("CHROMA_DATABASE")

    if not all([api_key, tenant, database]):
        raise ValueError("Missing Chroma Cloud credentials.")

    logger.info(f"Connecting to ChromaDB Cloud → collection '{collection_name}'")

    client = chromadb.CloudClient(
        api_key=api_key,
        tenant=tenant,
        database=database,
    )

    embedding_function = chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=embedding_model
    )

    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_function
    )

    # Optional wipe
    if clear_collection:
        logger.info("Clearing existing documents...")
        try:
            docs = collection.get()
            if docs and docs.get("ids"):
                collection.delete(ids=docs["ids"])
                logger.info(f"Cleared {len(docs['ids'])} existing documents.")
        except Exception as e:
            logger.warning(f"Could not clear collection: {e}")

    if not chunks:
        logger.warning("No chunks to upload.")
        return collection

    total = len(chunks)
    total_batches = (total + batch_size - 1) // batch_size

    logger.info(f"Uploading {total} chunks in {total_batches} batches...")

    uploaded = 0

    for i in range(0, total, batch_size):
        batch = chunks[i:i+batch_size]
        batch_num = i // batch_size + 1

        logger.info(f"  Batch {batch_num}/{total_batches}, {len(batch)} chunks")

        try:
            collection.add(
                ids=[c.doc_id for c in batch],
                documents=[c.content for c in batch],
                metadatas=[c.metadata for c in batch],
            )
            uploaded += len(batch)
        except Exception as e:
            logger.error(f"Batch {batch_num} failed: {e}")
            raise

    logger.info(f"✅ Uploaded {uploaded}/{total} chunks successfully.")
    logger.info(f"Collection now has {collection.count()} documents.")

    return collection


# ============================================================
# Main CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Load HotpotQA into ChromaDB Cloud")
    parser.add_argument("--data-file", type=str, required=True)
    parser.add_argument("--collection-name", type=str, default="hotpotqa_sentences")
    parser.add_argument("--api-key", type=str)
    parser.add_argument("--tenant", type=str)
    parser.add_argument("--database", type=str)
    parser.add_argument("--embedding-model", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--batch-size", type=int, default=300)
    parser.add_argument("--clear-collection", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    try:
        if not Path(args.data_file).exists():
            raise FileNotFoundError(f"Data file not found: {args.data_file}")

        logger.info("=== HotpotQA → ChromaDB Cloud ===")

        data = load_hotpotqa_data(args.data_file)
        chunks = create_chunks_from_hotpotqa(data)

        load_to_chroma_cloud(
            chunks=chunks,
            collection_name=args.collection_name,
            api_key=args.api_key,
            tenant=args.tenant,
            database=args.database,
            embedding_model=args.embedding_model,
            batch_size=args.batch_size,
            clear_collection=args.clear_collection,
        )

        logger.info("=== Upload Finished ===")

    except Exception as e:
        logger.error(f"Failed to upload: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
