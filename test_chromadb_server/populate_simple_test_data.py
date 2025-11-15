#!/usr/bin/env python3
"""
Populate ChromaDB with simple test data for MeVe testing.

This is a simpler alternative to populate_test_datasets.py that doesn't rely
on chroma_datasets. It creates sample documents with embeddings for testing.
"""

import chromadb
from chromadb.utils import embedding_functions
import os
import sys


def create_test_documents():
    """Create sample test documents."""
    return {
        "tech_articles": {
            "description": "Technology and AI articles (20 documents)",
            "documents": [
                "Artificial intelligence is transforming how we interact with technology. Machine learning models can now understand natural language, recognize images, and make predictions based on data patterns.",
                "The water cycle, also known as the hydrologic cycle, describes the continuous movement of water on, above and below the surface of the Earth. It involves evaporation, condensation, precipitation, and collection.",
                "Renewable energy sources like solar and wind power are becoming increasingly cost-effective. Solar panels convert sunlight directly into electricity, while wind turbines harness kinetic energy from wind.",
                "Photosynthesis is the process by which plants use sunlight, water and carbon dioxide to create oxygen and energy in the form of sugar. This process is fundamental to life on Earth.",
                "Cloud computing enables users to access computing resources over the internet. Instead of owning physical servers, companies can rent computing power, storage, and applications from cloud providers.",
                "Neural networks are computing systems inspired by biological neural networks in animal brains. They consist of interconnected nodes (neurons) that process information in layers to learn patterns.",
                "The human brain contains approximately 86 billion neurons, each connected to thousands of other neurons. This massive network enables consciousness, memory, and complex thought processes.",
                "Climate change refers to long-term shifts in global temperatures and weather patterns. While these shifts can be natural, human activities have been the main driver since the 1800s.",
                "Blockchain technology is a decentralized ledger system that records transactions across multiple computers. Each block contains transaction data and is cryptographically linked to the previous block.",
                "Quantum computing leverages quantum mechanical phenomena like superposition and entanglement to perform calculations. This allows quantum computers to solve certain problems much faster than classical computers.",
                "The Internet of Things (IoT) describes the network of physical objects embedded with sensors, software, and connectivity to exchange data with other devices over the internet.",
                "Cryptocurrency is a digital or virtual currency that uses cryptography for security. Bitcoin, created in 2009, was the first decentralized cryptocurrency and remains the most valuable.",
                "5G networks provide faster data speeds and lower latency than previous generations. This enables new applications like autonomous vehicles, remote surgery, and enhanced virtual reality experiences.",
                "Natural language processing (NLP) enables computers to understand, interpret and generate human language. Applications include chatbots, translation services, and voice assistants.",
                "Edge computing processes data closer to where it's generated rather than in centralized data centers. This reduces latency and bandwidth usage for IoT devices and real-time applications.",
                "Cybersecurity protects computer systems and networks from digital attacks, theft, and damage. As our dependence on technology grows, robust security measures become increasingly critical.",
                "Virtual reality (VR) creates immersive computer-generated environments that users can interact with. VR headsets track head and body movements to provide a realistic sensory experience.",
                "Genetic engineering allows scientists to modify an organism's DNA. CRISPR-Cas9 technology has made gene editing more precise, affordable, and accessible than ever before.",
                "The circular economy is an economic system aimed at eliminating waste and the continual use of resources. It emphasizes reuse, sharing, repair, refurbishment, remanufacturing and recycling.",
                "Automation uses technology to perform tasks with minimal human intervention. From manufacturing robots to software bots, automation increases efficiency and reduces human error.",
            ],
            "metadatas": [
                {"topic": "AI", "category": "technology", "source": "synthetic"},
                {"topic": "science", "category": "nature", "source": "synthetic"},
                {"topic": "energy", "category": "environment", "source": "synthetic"},
                {"topic": "biology", "category": "science", "source": "synthetic"},
                {"topic": "cloud", "category": "technology", "source": "synthetic"},
                {"topic": "AI", "category": "technology", "source": "synthetic"},
                {"topic": "neuroscience", "category": "science", "source": "synthetic"},
                {"topic": "climate", "category": "environment", "source": "synthetic"},
                {"topic": "blockchain", "category": "technology", "source": "synthetic"},
                {"topic": "quantum", "category": "technology", "source": "synthetic"},
                {"topic": "IoT", "category": "technology", "source": "synthetic"},
                {"topic": "crypto", "category": "finance", "source": "synthetic"},
                {"topic": "5G", "category": "technology", "source": "synthetic"},
                {"topic": "NLP", "category": "AI", "source": "synthetic"},
                {"topic": "edge computing", "category": "technology", "source": "synthetic"},
                {"topic": "security", "category": "technology", "source": "synthetic"},
                {"topic": "VR", "category": "technology", "source": "synthetic"},
                {"topic": "genetics", "category": "science", "source": "synthetic"},
                {"topic": "sustainability", "category": "environment", "source": "synthetic"},
                {"topic": "automation", "category": "technology", "source": "synthetic"},
            ]
        },
        "business_concepts": {
            "description": "Business and startup concepts (15 documents)",
            "documents": [
                "Product-market fit occurs when a product satisfies a strong market demand. Startups should focus on achieving this before scaling, as it indicates customers truly want what you're building.",
                "A minimum viable product (MVP) is a version of a product with just enough features to be usable by early customers. This approach helps validate assumptions before investing in full development.",
                "Growth hacking combines marketing and product development to rapidly grow a user base. It focuses on creative, low-cost strategies to acquire and retain customers efficiently.",
                "The lean startup methodology emphasizes building, measuring, and learning in rapid cycles. It helps entrepreneurs test hypotheses and pivot quickly based on customer feedback.",
                "Network effects occur when a product becomes more valuable as more people use it. Social media platforms, marketplaces, and communication tools benefit significantly from network effects.",
                "Venture capital is financing provided to startups and small businesses with long-term growth potential. VCs invest in exchange for equity and often provide strategic guidance.",
                "A business model describes how a company creates, delivers, and captures value. Common models include subscription, freemium, marketplace, and advertising-based approaches.",
                "Customer acquisition cost (CAC) is the total cost of acquiring a new customer. Companies should ensure CAC is significantly lower than customer lifetime value (LTV) for profitability.",
                "Agile development is an iterative approach to software development that delivers working features in short cycles. It emphasizes collaboration, flexibility, and continuous improvement.",
                "Platform businesses create value by facilitating exchanges between different user groups. Examples include Uber, Airbnb, and Amazon, which connect buyers and sellers.",
                "Data-driven decision making uses metrics and analytics to guide strategy rather than intuition alone. Successful companies track key performance indicators (KPIs) closely.",
                "Scalability refers to a business's ability to grow without being hampered by its structure or resources. Cloud infrastructure and automation enhance scalability significantly.",
                "Strategic partnerships can accelerate growth by combining complementary strengths. Partnerships provide access to new markets, technologies, or distribution channels.",
                "Company culture encompasses the values, beliefs, and behaviors that shape how work gets done. Strong culture attracts talent and drives long-term success.",
                "Exit strategies include acquisition, IPO, or merger. Founders and investors should consider potential exit paths early in a company's lifecycle.",
            ],
            "metadatas": [
                {"topic": "product", "category": "startup", "source": "synthetic"},
                {"topic": "MVP", "category": "product", "source": "synthetic"},
                {"topic": "marketing", "category": "growth", "source": "synthetic"},
                {"topic": "lean startup", "category": "methodology", "source": "synthetic"},
                {"topic": "network effects", "category": "business", "source": "synthetic"},
                {"topic": "funding", "category": "finance", "source": "synthetic"},
                {"topic": "business model", "category": "strategy", "source": "synthetic"},
                {"topic": "metrics", "category": "analytics", "source": "synthetic"},
                {"topic": "agile", "category": "development", "source": "synthetic"},
                {"topic": "platforms", "category": "business model", "source": "synthetic"},
                {"topic": "analytics", "category": "strategy", "source": "synthetic"},
                {"topic": "scalability", "category": "growth", "source": "synthetic"},
                {"topic": "partnerships", "category": "growth", "source": "synthetic"},
                {"topic": "culture", "category": "management", "source": "synthetic"},
                {"topic": "exit", "category": "finance", "source": "synthetic"},
            ]
        }
    }


def populate_chromadb(db_path: str = "./chroma_test_data"):
    """
    Populate ChromaDB with test data.

    Args:
        db_path: Path where ChromaDB data will be stored
    """
    print(f"\n{'='*60}")
    print("Simple ChromaDB Test Data Population")
    print(f"{'='*60}\n")

    # Create persistent ChromaDB client
    print(f"📁 Creating persistent ChromaDB at: {db_path}")
    client = chromadb.PersistentClient(path=db_path)

    # Get test documents
    collections_data = create_test_documents()

    # Create embedding function
    print("🔧 Initializing embedding function (sentence-transformers)...")
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    print(f"\nAvailable collections:")
    for i, (name, data) in enumerate(collections_data.items(), 1):
        print(f"  {i}. {name}: {data['description']}")

    print("\n" + "="*60)
    choice = input("\nSelect collections to import (comma-separated, e.g. '1,2' or 'all'): ").strip()

    if choice.lower() == 'all':
        selected_names = list(collections_data.keys())
    else:
        try:
            indices = [int(x.strip()) - 1 for x in choice.split(',')]
            selected_names = [list(collections_data.keys())[i] for i in indices if 0 <= i < len(collections_data)]
        except (ValueError, IndexError):
            print("❌ Invalid input. Please try again.")
            return

    print("\n" + "="*60)
    print("Importing collections...\n")

    imported_collections = []

    for name in selected_names:
        data = collections_data[name]
        print(f"📥 Creating collection: {name}")
        print(f"   {data['description']}")

        try:
            # Create or get collection
            collection = client.get_or_create_collection(
                name=name,
                embedding_function=sentence_transformer_ef
            )

            # Add documents
            ids = [f"{name}_{i}" for i in range(len(data['documents']))]
            collection.add(
                documents=data['documents'],
                metadatas=data['metadatas'],
                ids=ids
            )

            doc_count = collection.count()
            print(f"   ✅ Success! Imported {doc_count} documents")

            # Show sample
            sample = collection.peek(limit=1)
            if sample['documents']:
                preview = sample['documents'][0][:80] + "..." if len(sample['documents'][0]) > 80 else sample['documents'][0]
                print(f"   📄 Sample: {preview}\n")

            imported_collections.append({
                "name": name,
                "count": doc_count
            })

        except Exception as e:
            print(f"   ❌ Failed: {str(e)}\n")

    print("="*60)
    print("\n✅ Import Complete!\n")

    if imported_collections:
        print("Imported collections:")
        for col in imported_collections:
            print(f"  • {col['name']}: {col['count']} documents")

        print(f"\n📁 Data location: {os.path.abspath(db_path)}")
        print("\nNext steps:")
        print("  1. Run: uv run python start_test_server.py")
        print("  2. Configure MeVe MCP with: CHROMADB_URL=http://localhost:8000")
        print("  3. In Claude Desktop:")
        print("     - 'List ChromaDB collections'")
        print("     - 'Load the tech_articles collection'")
        print("     - 'Search for information about AI'")
    else:
        print("⚠️  No collections were imported.")

    print("\n" + "="*60)


def list_collections(db_path: str = "./chroma_test_data"):
    """List existing collections."""
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

        try:
            sample = col.peek(limit=1)
            if sample['documents']:
                preview = sample['documents'][0][:60] + "..." if len(sample['documents'][0]) > 60 else sample['documents'][0]
                print(f"   Sample: {preview}")
        except:
            pass
        print()

    print("="*60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Populate ChromaDB with simple test data")
    parser.add_argument(
        "--db-path",
        default="./chroma_test_data",
        help="Path for ChromaDB data storage"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List existing collections"
    )

    args = parser.parse_args()

    if args.list:
        list_collections(args.db_path)
    else:
        populate_chromadb(args.db_path)
