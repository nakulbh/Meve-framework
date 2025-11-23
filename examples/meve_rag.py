#!/usr/bin/env python3
"""
Simple RAG System using MeVe Framework
A question-answering system that uses the MeVe 5-phase RAG pipeline
to retrieve and generate answers from ChromaDB.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import chromadb

# Import MeVe components
from meve.core.models import ContextChunk, MeVeConfig
from meve.core.engine import MeVeEngine
from meve.services.vector_db_client import VectorDBClient


class SimpleRAGSystem:
    """A simple RAG system using MeVe for question answering."""

    def __init__(self, collection_name: str = "wik_dataset"):
        self.collection_name = collection_name
        self.chunks: List[ContextChunk] = []
        self.engine: Optional[MeVeEngine] = None
        self.vector_db_client: Optional[VectorDBClient] = None

        # Load chunks from ChromaDB and initialize engine
        self.load_from_chromadb()
        self.initialize_engine()

    def load_from_chromadb(self) -> None:
        """Load chunks from ChromaDB collection."""
        print(f"Loading data from ChromaDB collection: {self.collection_name}")

        try:
            # Get ChromaDB path
            script_dir = Path(__file__).parent
            chroma_path = script_dir.parent / "chroma"

            # Initialize VectorDBClient with existing collection
            self.vector_db_client = VectorDBClient(
                chunks=None,
                is_persistent=True,
                collection_name=self.collection_name,
                load_existing=True,
            )

            # Get the chunks from the client
            self.chunks = self.vector_db_client.chunks
            print(f"✅ Loaded {len(self.chunks)} chunks from ChromaDB")

        except Exception as e:
            print(f"❌ Failed to load from ChromaDB: {e}")
            print(f"💡 Make sure you've run: make wiki-to-chromadb")
            self.chunks = []
            self.vector_db_client = None

    def initialize_engine(self) -> None:
        """Initialize the MeVe engine with ChromaDB vector client."""
        if not self.chunks or not self.vector_db_client:
            print("❌ No chunks or vector client available for engine initialization")
            return

        print("Initializing MeVe engine...")

        # Create default config (you can customize these parameters)
        config = MeVeConfig(
            k_init=20,  # Initial retrieval count
            tau_relevance=0.3,  # Relevance threshold
            n_min=2,  # Minimum verified docs
            theta_redundancy=0.8,  # Redundancy threshold
            lambda_mmr=0.5,  # MMR lambda
            t_max=400,  # Token budget
        )

        try:
            # Convert chunks list to dict for BM25 index
            bm25_index = {chunk.doc_id: chunk for chunk in self.chunks}

            # Initialize with VectorDBClient for phase 1 and dict for BM25
            self.engine = MeVeEngine(
                config=config,
                vector_db_client=self.vector_db_client,  # Use ChromaDB client
                bm25_index=bm25_index,  # Use dict for BM25
            )
            print("✅ MeVe engine initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize MeVe engine: {e}")
            import traceback

            traceback.print_exc()
            self.engine = None

    def answer_question(self, question: str) -> Dict[str, Any]:
        """
        Answer a question using the MeVe RAG pipeline.

        Args:
            question: The question to answer

        Returns:
            Dict containing answer, context, and metadata
        """
        if not self.engine:
            return {
                "answer": "RAG system not properly initialized",
                "context": [],
                "metadata": {"error": "engine_not_initialized"},
            }

        try:
            print(f"\nProcessing question: '{question}'")

            # Run MeVe pipeline
            result = self.engine.run(question)

            # Extract final answer and context
            if isinstance(result, tuple):
                final_answer, final_chunks = result
            else:
                final_answer = str(result)
                final_chunks = []

            return {
                "answer": final_answer,
                "context": [
                    {
                        "content": chunk.content,
                        "doc_id": chunk.doc_id,
                        "title": "Unknown",  # Extract from content if available
                        "source": "unknown",
                    }
                    for chunk in final_chunks
                ],
                "metadata": {
                    "total_chunks": len(self.chunks),
                    "retrieved_chunks": len(final_chunks),
                    "config": {
                        "k_init": self.engine.config.k_init,
                        "tau_relevance": self.engine.config.tau_relevance,
                        "t_max": self.engine.config.t_max,
                    },
                },
            }

        except Exception as e:
            print(f"Error processing question: {e}")
            return {
                "answer": f"Error processing question: {str(e)}",
                "context": [],
                "metadata": {"error": str(e)},
            }

    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        return {
            "collection_name": self.collection_name,
            "total_chunks": len(self.chunks),
            "engine_initialized": self.engine is not None,
            "vector_db_initialized": self.vector_db_client is not None,
        }

    def run_batch_questions(self, questions: List[str]) -> None:
        """Run multiple questions in batch mode."""
        print(f"\n🚀 Running {len(questions)} questions in batch mode...\n")

        for i, question in enumerate(questions, 1):
            print(f"Q{i}: {question}")
            result = self.answer_question(question)

            print(f"🤖 Answer: {result['answer']}")

            if result["context"]:
                print(f"📚 Retrieved {len(result['context'])} relevant chunks")
            else:
                print("📚 No relevant context found.")

            meta = result["metadata"]
            print(
                f"📈 Stats: {meta.get('retrieved_chunks', 0)} chunks retrieved from {meta.get('total_chunks', 0)} total"
            )
            print("-" * 60)


def main():
    """Interactive CLI for the RAG system."""
    print("=" * 60)
    print("🤖 MeVe RAG Question-Answering System")
    print("=" * 60)

    # Initialize RAG system
    rag = SimpleRAGSystem()

    # Show statistics
    stats = rag.get_stats()
    print(f"\n📊 System Statistics:")
    print(f"   ChromaDB Collection: {stats['collection_name']}")
    print(f"   Total chunks: {stats['total_chunks']}")
    print(f"   Engine initialized: {stats['engine_initialized']}")
    print(f"   VectorDB initialized: {stats['vector_db_initialized']}")

    if not stats["engine_initialized"]:
        print("\n❌ RAG system failed to initialize. Check ChromaDB collection.")
        print("💡 Run: make wiki-to-chromadb")
        return

    # Sample questions for testing
    sample_questions = [
        "Who is Aristotle?",
        "What is the speed of light?",
        "How does photosynthesis work?",
        "What is quantum mechanics?",
        "Who discovered penicillin?",
        "What causes gravity?",
        "How do magnets work?",
        "What is the theory of relativity?",
        "Who was Albert Einstein?",
        "What is the periodic table?",
    ]

    print("\n💡 Ask questions about history, science, physics, or general knowledge!")
    print("   Type 'batch' to run sample questions, or 'quit'/'exit' to stop.\n")

    # Interactive loop
    while True:
        try:
            question = input("❓ Your question: ").strip()

            if question.lower() in ["quit", "exit", "q"]:
                print("\n👋 Goodbye!")
                break

            if question.lower() == "batch":
                rag.run_batch_questions(sample_questions)
                continue

            if not question:
                continue

            # Get answer
            result = rag.answer_question(question)

            # Display answer
            print(f"\n🤖 Answer: {result['answer']}")

            # Show context info
            if result["context"]:
                print(f"\n📚 Retrieved {len(result['context'])} relevant chunks:")
                for i, chunk in enumerate(result["context"][:3], 1):  # Show top 3
                    title = chunk.get("title", "Unknown")
                    content_preview = (
                        chunk["content"][:100] + "..."
                        if len(chunk["content"]) > 100
                        else chunk["content"]
                    )
                    print(f"   {i}. [{title}] {content_preview}")
            else:
                print("\n📚 No relevant context found.")

            # Show metadata
            meta = result["metadata"]
            print(
                f"\n📈 Stats: {meta.get('retrieved_chunks', 0)} chunks retrieved from {meta.get('total_chunks', 0)} total"
            )

            print("\n" + "-" * 60)

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("-" * 60)


if __name__ == "__main__":
    main()
