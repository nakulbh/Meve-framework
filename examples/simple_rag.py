#!/usr/bin/env python3
"""
Simple RAG System (No MeVe)
A basic question-answering system using simple vector similarity search
and retrieval from ChromaDB.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings


class SimpleRAGSystem:
    """A simple RAG system using ChromaDB for vector similarity search."""

    def __init__(self, collection_name: str = "wik_dataset", top_k: int = 5):
        self.collection_name = collection_name
        self.top_k = top_k
        self.encoder: Optional[SentenceTransformer] = None
        self.collection = None
        self.total_chunks = 0

        # Initialize ChromaDB and load data
        self.initialize_chromadb()

    def initialize_chromadb(self) -> None:
        """Initialize ChromaDB client and load the collection."""
        print(f"Loading data from ChromaDB collection: {self.collection_name}")

        try:
            # Initialize ChromaDB client with persistent storage
            # Use the same persistence directory as the loading script
            script_dir = Path(__file__).parent
            chroma_path = script_dir.parent / "chroma"
            
            self.chroma_client = chromadb.PersistentClient(
                path=str(chroma_path)
            )

            # Get the collection (it should already exist from the script)
            # Note: We don't need to specify embedding_function when just querying
            self.collection = self.chroma_client.get_collection(
                name=self.collection_name
            )

            # Get collection count
            self.total_chunks = self.collection.count()
            print(f"✅ Loaded collection with {self.total_chunks} chunks")

            # Initialize encoder for querying
            print("Initializing sentence encoder...")
            self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
            print("✅ Encoder loaded successfully")

        except Exception as e:
            print(f"❌ Failed to initialize ChromaDB: {e}")
            print(f"💡 Make sure you've run: make wiki-to-chromadb")
            self.collection = None
            self.encoder = None

    def retrieve_context(self, query: str) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context for a query using ChromaDB vector similarity.

        Args:
            query: The query string

        Returns:
            List of top-k most similar chunks with metadata
        """
        if self.encoder is None or self.collection is None:
            print("❌ ChromaDB or encoder not initialized")
            return []

        try:
            # Encode query
            query_embedding = self.encoder.encode(query, convert_to_numpy=True).tolist()

            # Query ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(self.top_k, self.total_chunks)
            )

            # Format results
            retrieved_chunks = []
            if results['documents'] and results['documents'][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    # Convert distance to similarity (ChromaDB returns L2 distance)
                    similarity = 1.0 / (1.0 + distance)
                    
                    # Extract article title from content (format: "Article: Title\n\n...")
                    title = "Unknown"
                    if doc.startswith("Article: "):
                        title_end = doc.find("\n\n")
                        if title_end > 0:
                            title = doc[9:title_end]  # Skip "Article: "
                    
                    retrieved_chunks.append({
                        'content': doc,
                        'doc_id': metadata.get('doc_id', f'unknown_{i}'),
                        'title': title,
                        'source': 'wikipedia',
                        'similarity': similarity
                    })

            return retrieved_chunks

        except Exception as e:
            print(f"Error retrieving context: {e}")
            return []

    def answer_question(self, question: str) -> Dict[str, Any]:
        """
        Answer a question using simple RAG.

        Args:
            question: The question to answer

        Returns:
            Dict containing retrieved context and metadata
        """
        if not self.encoder:
            return {
                "answer": "RAG system not properly initialized",
                "context": [],
                "metadata": {"error": "encoder_not_initialized"}
            }

        try:
            print(f"\nProcessing question: '{question}'")

            # Retrieve relevant context
            retrieved_chunks = self.retrieve_context(question)

            # Build context string
            context_parts = []
            for i, chunk in enumerate(retrieved_chunks, 1):
                context_parts.append(f"[{i}] {chunk['content']} (similarity: {chunk['similarity']:.3f})")

            context_string = "\n\n".join(context_parts)

            # In a real system, you would pass this to an LLM
            # For this example, we just return the context
            answer = f"Retrieved {len(retrieved_chunks)} relevant passages. In a complete RAG system, these would be passed to an LLM to generate an answer."

            return {
                "answer": answer,
                "context": retrieved_chunks,
                "context_string": context_string,
                "metadata": {
                    "total_chunks": self.total_chunks,
                    "retrieved_chunks": len(retrieved_chunks),
                    "top_k": self.top_k
                }
            }

        except Exception as e:
            print(f"Error processing question: {e}")
            return {
                "answer": f"Error processing question: {str(e)}",
                "context": [],
                "metadata": {"error": str(e)}
            }

    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        return {
            "collection_name": self.collection_name,
            "total_chunks": self.total_chunks,
            "encoder_initialized": self.encoder is not None,
            "chromadb_initialized": self.collection is not None
        }

    def run_batch_questions(self, questions: List[str]) -> None:
        """Run multiple questions in batch mode."""
        print(f"\n🚀 Running {len(questions)} questions in batch mode...\n")

        for i, question in enumerate(questions, 1):
            print(f"Q{i}: {question}")
            result = self.answer_question(question)

            print(f"\n📚 Retrieved {len(result['context'])} relevant chunks:")
            for j, chunk in enumerate(result['context'], 1):
                preview = chunk['content'][:80] + "..." if len(chunk['content']) > 80 else chunk['content']
                print(f"   {j}. [{chunk['title']}] {preview}")
                print(f"      Similarity: {chunk['similarity']:.3f}")

            meta = result['metadata']
            print(f"\n📈 Stats: {meta.get('retrieved_chunks', 0)} chunks retrieved from {meta.get('total_chunks', 0)} total")
            print("-" * 60)


def main():
    """Interactive CLI for the RAG system."""
    print("=" * 60)
    print("🤖 Simple RAG Question-Answering System")
    print("=" * 60)

    # Initialize RAG system
    rag = SimpleRAGSystem(top_k=5)

    # Show statistics
    stats = rag.get_stats()
    print(f"\n📊 System Statistics:")
    print(f"   ChromaDB Collection: {stats['collection_name']}")
    print(f"   Total chunks: {stats['total_chunks']}")
    print(f"   Encoder initialized: {stats['encoder_initialized']}")
    print(f"   ChromaDB initialized: {stats['chromadb_initialized']}")

    if not stats['chromadb_initialized']:
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
        "What is the periodic table?"
    ]

    print("\n💡 Ask questions about history, science, physics, or general knowledge!")
    print("   Type 'batch' to run sample questions, or 'quit'/'exit' to stop.\n")

    # Interactive loop
    while True:
        try:
            question = input("❓ Your question: ").strip()

            if question.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break

            if question.lower() == 'batch':
                rag.run_batch_questions(sample_questions)
                continue

            if not question:
                continue

            # Get answer
            result = rag.answer_question(question)

            # Display context
            if result['context']:
                print(f"\n📚 Retrieved {len(result['context'])} relevant chunks:")
                for i, chunk in enumerate(result['context'], 1):
                    title = chunk.get('title', 'Unknown')
                    content_preview = chunk['content'][:100] + "..." if len(chunk['content']) > 100 else chunk['content']
                    similarity = chunk.get('similarity', 0.0)
                    print(f"   {i}. [{title}] {content_preview}")
                    print(f"      Similarity: {similarity:.3f}")
            else:
                print("\n📚 No relevant context found.")

            # Show metadata
            meta = result['metadata']
            print(f"\n📈 Stats: {meta.get('retrieved_chunks', 0)} chunks retrieved from {meta.get('total_chunks', 0)} total")

            print("\n💡 Note: In a complete RAG system, these chunks would be passed to an LLM to generate a natural language answer.")
            print("\n" + "-" * 60)

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("-" * 60)


if __name__ == "__main__":
    main()
