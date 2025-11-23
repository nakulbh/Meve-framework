#!/usr/bin/env python3
"""
ChromaDB Setup Script for MeVe Framework

This script provides utilities to:
1. Initialize persistent ChromaDB storage
2. Populate ChromaDB with HotpotQA data
3. Manage ChromaDB collections
4. Test ChromaDB functionality

Usage:
    python scripts/setup_chromadb.py --mode <init|populate|test|clean>

    Modes:
        init     - Initialize persistent ChromaDB storage
        populate - Load HotpotQA data into ChromaDB
        test     - Test ChromaDB query functionality
        clean    - Remove ChromaDB persistent storage
        info     - Display ChromaDB collection information
"""

import argparse
import os
import sys
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from meve.core.engine import load_hotpotqa_data


class ChromaDBSetup:
    """Manages ChromaDB setup and operations for MeVe framework."""

    def __init__(
        self, persist_directory: str = "./data/chromadb", collection_name: str = "meve_chunks"
    ):
        """
        Initialize ChromaDB setup manager.

        Args:
            persist_directory: Directory for persistent ChromaDB storage
            collection_name: Name of the ChromaDB collection
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def init_persistent_db(self):
        """Initialize persistent ChromaDB storage."""
        print("🔧 Initializing persistent ChromaDB storage...")
        print(f"📁 Storage directory: {self.persist_directory}")

        # Create directory if it doesn't exist
        os.makedirs(self.persist_directory, exist_ok=True)

        # Initialize persistent ChromaDB client
        client = chromadb.PersistentClient(
            path=self.persist_directory, settings=Settings(anonymized_telemetry=False)
        )

        # Create or get collection
        try:
            collection = client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name="all-MiniLM-L6-v2"
                ),
            )
            print(f"✅ Collection '{self.collection_name}' initialized")
            print(f"📊 Current collection size: {collection.count()} documents")

        except Exception as e:
            print(f"❌ Error initializing collection: {e}")
            sys.exit(1)

        print("✅ ChromaDB persistent storage initialized successfully!")
        return client, collection

    def populate_from_hotpotqa(self, max_examples: int = 100, batch_size: int = 100):
        """
        Populate ChromaDB with HotpotQA data.

        Args:
            max_examples: Maximum number of HotpotQA examples to load
            batch_size: Batch size for adding documents to ChromaDB
        """
        print("\n📚 Populating ChromaDB with HotpotQA data...")
        print(f"   • Max examples: {max_examples}")
        print(f"   • Batch size: {batch_size}")

        # Load HotpotQA data
        chunks_dict, questions = load_hotpotqa_data(max_examples=max_examples)
        chunks = list(chunks_dict.values())

        if not chunks:
            print("❌ No chunks loaded from HotpotQA data")
            sys.exit(1)

        print(f"✅ Loaded {len(chunks)} chunks from HotpotQA")

        # Initialize persistent DB
        client, collection = self.init_persistent_db()

        # Clear existing data
        try:
            if collection.count() > 0:
                print(f"🗑️  Clearing existing {collection.count()} documents...")
                client.delete_collection(name=self.collection_name)
                collection = client.create_collection(
                    name=self.collection_name,
                    embedding_function=chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction(
                        model_name="all-MiniLM-L6-v2"
                    ),
                )
        except Exception as e:
            print(f"⚠️  Warning during cleanup: {e}")

        # Add chunks in batches
        print(f"\n📥 Adding {len(chunks)} chunks to ChromaDB...")

        for i in tqdm(range(0, len(chunks), batch_size), desc="Adding batches"):
            batch_chunks = chunks[i : i + batch_size]

            documents = []
            metadatas = []
            ids = []

            for idx, chunk in enumerate(batch_chunks):
                documents.append(chunk.content)
                metadatas.append({"doc_id": chunk.doc_id, "chunk_index": i + idx})
                ids.append(f"chunk_{i + idx}")

            try:
                collection.add(documents=documents, metadatas=metadatas, ids=ids)
            except Exception as e:
                print(f"\n❌ Error adding batch {i // batch_size + 1}: {e}")
                continue

        final_count = collection.count()
        print(f"\n✅ Successfully added {final_count} documents to ChromaDB")
        print(f"📊 Collection '{self.collection_name}' is ready for use")

        return client, collection

    def test_queries(self, num_test_queries: int = 3):
        """
        Test ChromaDB query functionality.

        Args:
            num_test_queries: Number of test queries to run
        """
        print("\n🔍 Testing ChromaDB query functionality...")

        # Initialize persistent DB
        client = chromadb.PersistentClient(
            path=self.persist_directory, settings=Settings(anonymized_telemetry=False)
        )

        try:
            collection = client.get_collection(
                name=self.collection_name,
                embedding_function=chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name="all-MiniLM-L6-v2"
                ),
            )
        except Exception as e:
            print(f"❌ Collection not found: {e}")
            print("💡 Run with --mode populate first to create the collection")
            sys.exit(1)

        print(f"📊 Collection size: {collection.count()} documents")

        # Test queries
        test_queries = [
            "What is the capital of France?",
            "Who invented the telephone?",
            "When did World War 2 end?",
        ]

        for i, query_text in enumerate(test_queries[:num_test_queries], 1):
            print(f"\n{'=' * 60}")
            print(f"Query {i}: {query_text}")
            print(f"{'=' * 60}")

            try:
                results = collection.query(query_texts=[query_text], n_results=3)

                print(f"✅ Found {len(results['documents'][0])} results:")
                for j, (doc, distance, metadata) in enumerate(
                    zip(results["documents"][0], results["distances"][0], results["metadatas"][0]),
                    1,
                ):
                    print(f"\n  Result {j}:")
                    print(f"    • Distance: {distance:.4f}")
                    print(f"    • Doc ID: {metadata.get('doc_id', 'N/A')}")
                    print(f"    • Content: {doc[:100]}...")

            except Exception as e:
                print(f"❌ Error querying: {e}")

        print("\n✅ Query testing completed")

    def get_collection_info(self):
        """Display information about the ChromaDB collection."""
        print("\n📊 ChromaDB Collection Information")
        print(f"{'=' * 60}")

        if not os.path.exists(self.persist_directory):
            print(f"❌ ChromaDB storage not found at: {self.persist_directory}")
            print("💡 Run with --mode init to create the storage")
            return

        try:
            client = chromadb.PersistentClient(
                path=self.persist_directory, settings=Settings(anonymized_telemetry=False)
            )

            # List all collections
            collections = client.list_collections()
            print(f"\n📚 Collections ({len(collections)}):")
            for coll in collections:
                print(f"   • {coll.name}: {coll.count()} documents")

            # Get specific collection info
            try:
                collection = client.get_collection(name=self.collection_name)
                print(f"\n🎯 Collection: '{self.collection_name}'")
                print(f"   • Total documents: {collection.count()}")
                print(f"   • Storage path: {self.persist_directory}")

                # Peek at some data
                if collection.count() > 0:
                    sample = collection.peek(limit=3)
                    print("\n📝 Sample documents:")
                    for i, (doc_id, doc) in enumerate(zip(sample["ids"], sample["documents"]), 1):
                        print(f"   {i}. [{doc_id}] {doc[:80]}...")

            except Exception as e:
                print(f"⚠️  Collection '{self.collection_name}' not found: {e}")

        except Exception as e:
            print(f"❌ Error accessing ChromaDB: {e}")

    def clean_storage(self):
        """Remove ChromaDB persistent storage."""
        print("\n🗑️  Cleaning ChromaDB storage...")
        print(f"📁 Target directory: {self.persist_directory}")

        if not os.path.exists(self.persist_directory):
            print("✅ Storage directory doesn't exist, nothing to clean")
            return

        response = input("⚠️  Are you sure you want to delete all ChromaDB data? (yes/no): ")
        if response.lower() != "yes":
            print("❌ Cleanup cancelled")
            return

        try:
            import shutil

            shutil.rmtree(self.persist_directory)
            print("✅ Successfully removed ChromaDB storage")
        except Exception as e:
            print(f"❌ Error removing storage: {e}")


def main():
    """Main entry point for ChromaDB setup script."""
    parser = argparse.ArgumentParser(
        description="ChromaDB Setup Script for MeVe Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Initialize persistent storage
  python scripts/setup_chromadb.py --mode init
  
  # Populate with HotpotQA data
  python scripts/setup_chromadb.py --mode populate --max-examples 100
  
  # Test query functionality
  python scripts/setup_chromadb.py --mode test
  
  # View collection info
  python scripts/setup_chromadb.py --mode info
  
  # Clean up storage
  python scripts/setup_chromadb.py --mode clean
        """,
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["init", "populate", "test", "clean", "info"],
        required=True,
        help="Operation mode",
    )

    parser.add_argument(
        "--persist-dir",
        type=str,
        default="./data/chromadb",
        help="Directory for persistent ChromaDB storage (default: ./data/chromadb)",
    )

    parser.add_argument(
        "--collection-name",
        type=str,
        default="meve_chunks",
        help="Name of the ChromaDB collection (default: meve_chunks)",
    )

    parser.add_argument(
        "--max-examples",
        type=int,
        default=100,
        help="Maximum number of HotpotQA examples to load (default: 100)",
    )

    parser.add_argument(
        "--batch-size", type=int, default=100, help="Batch size for adding documents (default: 100)"
    )

    parser.add_argument(
        "--num-test-queries", type=int, default=3, help="Number of test queries to run (default: 3)"
    )

    args = parser.parse_args()

    # Initialize setup manager
    setup = ChromaDBSetup(persist_directory=args.persist_dir, collection_name=args.collection_name)

    # Execute requested operation
    print(f"\n{'=' * 60}")
    print("🚀 MeVe ChromaDB Setup")
    print(f"{'=' * 60}")

    if args.mode == "init":
        setup.init_persistent_db()

    elif args.mode == "populate":
        setup.populate_from_hotpotqa(max_examples=args.max_examples, batch_size=args.batch_size)

    elif args.mode == "test":
        setup.test_queries(num_test_queries=args.num_test_queries)

    elif args.mode == "info":
        setup.get_collection_info()

    elif args.mode == "clean":
        setup.clean_storage()

    print(f"\n{'=' * 60}")
    print("✅ Operation completed")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
