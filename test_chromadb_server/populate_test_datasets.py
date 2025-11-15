#!/usr/bin/env python3
"""
Populate ChromaDB with test datasets from chroma_datasets package.

This script loads pre-built datasets with embeddings into a persistent ChromaDB instance.
"""

import chromadb
from chromadb.config import Settings
from chroma_datasets import StateOfTheUnion, PaulGrahamEssay, HubermanPodcasts
from chroma_datasets.utils import import_into_chroma
import sys
import os


def populate_chromadb(db_path: str = "./chroma_test_data"):
    """
    Populate ChromaDB with test datasets.

    Args:
        db_path: Path where ChromaDB data will be stored
    """
    print(f"\n{'='*60}")
    print("ChromaDB Test Data Population Script")
    print(f"{'='*60}\n")

    # Create persistent ChromaDB client
    print(f"📁 Creating persistent ChromaDB at: {db_path}")
    client = chromadb.PersistentClient(path=db_path)

    # Available datasets
    datasets = [
        {
            "name": "State of the Union",
            "class": StateOfTheUnion,
            "collection": "state_of_the_union",
            "description": "2022 State of the Union address (51kb, ~40 chunks)",
            "size": "Small"
        },
        {
            "name": "Paul Graham Essays",
            "class": PaulGrahamEssay,
            "collection": "paul_graham_essays",
            "description": "Paul Graham's essays (1.3mb, ~300 chunks)",
            "size": "Medium"
        },
        {
            "name": "Huberman Podcasts",
            "class": HubermanPodcasts,
            "collection": "huberman_podcasts",
            "description": "Huberman Lab podcast transcripts (4.3mb, ~1000 chunks)",
            "size": "Large"
        }
    ]

    print("\nAvailable datasets:")
    for i, dataset in enumerate(datasets, 1):
        print(f"  {i}. {dataset['name']} ({dataset['size']})")
        print(f"     {dataset['description']}")

    print("\n" + "="*60)
    choice = input("\nSelect datasets to import (comma-separated, e.g. '1,2' or 'all'): ").strip()

    if choice.lower() == 'all':
        selected_indices = list(range(len(datasets)))
    else:
        try:
            selected_indices = [int(x.strip()) - 1 for x in choice.split(',')]
        except ValueError:
            print("❌ Invalid input. Please enter numbers separated by commas.")
            return

    print("\n" + "="*60)
    print("Importing datasets...\n")

    imported_collections = []

    for idx in selected_indices:
        if 0 <= idx < len(datasets):
            dataset = datasets[idx]
            print(f"📥 Importing: {dataset['name']}")
            print(f"   Collection: {dataset['collection']}")

            try:
                # Import dataset into ChromaDB
                collection = import_into_chroma(
                    chroma_client=client,
                    dataset=dataset['class'],
                    collection_name=dataset['collection']
                )

                # Get collection info
                doc_count = collection.count()
                sample = collection.peek(limit=1)
                has_embeddings = len(sample.get('embeddings', [])) > 0

                print(f"   ✅ Success! Imported {doc_count} documents")
                print(f"   📊 Has embeddings: {has_embeddings}")

                # Get a sample document
                if sample['documents']:
                    preview = sample['documents'][0][:100] + "..." if len(sample['documents'][0]) > 100 else sample['documents'][0]
                    print(f"   📄 Sample: {preview}\n")

                imported_collections.append({
                    "name": dataset['name'],
                    "collection": dataset['collection'],
                    "count": doc_count
                })

            except Exception as e:
                print(f"   ❌ Failed: {str(e)}\n")

    print("="*60)
    print("\n✅ Import Complete!\n")

    if imported_collections:
        print("Imported collections:")
        for col in imported_collections:
            print(f"  • {col['name']}: {col['count']} documents (collection: {col['collection']})")

        print(f"\n📁 Data location: {os.path.abspath(db_path)}")
        print("\nNext steps:")
        print("  1. Run: python start_test_server.py")
        print("  2. Configure MeVe MCP server with: CHROMADB_URL=http://localhost:8000")
        print("  3. In Claude Desktop, use: connect_to_chromadb(server_url='http://localhost:8000')")
        print(f"  4. Load a collection: use_external_collection_for_meve(collection_name='state_of_the_union')")
        print("  5. Query: query_with_meve(query='your query here')")
    else:
        print("⚠️  No collections were imported.")

    print("\n" + "="*60)


def list_existing_collections(db_path: str = "./chroma_test_data"):
    """List collections in an existing ChromaDB."""
    if not os.path.exists(db_path):
        print(f"❌ Database not found at: {db_path}")
        return

    print(f"\n{'='*60}")
    print("Existing Collections")
    print(f"{'='*60}\n")

    client = chromadb.PersistentClient(path=db_path)
    collections = client.list_collections()

    if not collections:
        print("No collections found.")
        return

    for col in collections:
        count = col.count()
        print(f"📚 {col.name}")
        print(f"   Documents: {count}")

        # Get sample
        try:
            sample = col.peek(limit=1)
            if sample['documents']:
                preview = sample['documents'][0][:80] + "..." if len(sample['documents'][0]) > 80 else sample['documents'][0]
                print(f"   Sample: {preview}")
        except:
            pass
        print()

    print("="*60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Populate ChromaDB with test datasets")
    parser.add_argument(
        "--db-path",
        default="./chroma_test_data",
        help="Path for ChromaDB data storage (default: ./chroma_test_data)"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List existing collections"
    )

    args = parser.parse_args()

    if args.list:
        list_existing_collections(args.db_path)
    else:
        populate_chromadb(args.db_path)
