"""
RAG Comparison: Basic RAG vs MeVe RAG
This demonstrates the difference between simple vector search and MeVe's 5-phase pipeline.
"""

from typing import Dict, List, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer

# Import MeVe components
from meve.core.models import ContextChunk, MeVeConfig
from meve.core.engine import MeVeEngine
from meve.services.vector_db_client import VectorDBClient


def create_sample_corpus() -> Dict[str, ContextChunk]:
    """
    Create a sample document corpus about AI and machine learning.
    
    Returns:
        Dict mapping doc_id to ContextChunk
    """
    documents = [
        # Core ML concepts
        ("doc1", "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed."),
        ("doc2", "Deep learning uses neural networks with multiple layers to progressively extract higher-level features from raw input data."),
        ("doc3", "Supervised learning trains models on labeled data, while unsupervised learning finds patterns in unlabeled data."),
        
        # RAG and retrieval
        ("doc4", "Retrieval-Augmented Generation (RAG) combines information retrieval with language generation to provide context-aware responses."),
        ("doc5", "Vector databases enable semantic search by storing embeddings and finding similar documents based on cosine similarity."),
        ("doc6", "Cross-encoders provide more accurate relevance scoring than bi-encoders but are computationally expensive."),
        
        # Applications
        ("doc7", "Large language models like GPT-4 and Claude use transformer architectures to process and generate human-like text."),
        ("doc8", "Named entity recognition (NER) identifies and classifies entities such as person names, organizations, and locations in text."),
        ("doc9", "Sentiment analysis determines the emotional tone of text, classifying it as positive, negative, or neutral."),
        
        # Advanced topics
        ("doc10", "Multi-modal learning processes data from multiple sources like text, images, and audio to create richer representations."),
        ("doc11", "Fine-tuning adapts pre-trained models to specific tasks by training on domain-specific data with smaller learning rates."),
        ("doc12", "Prompt engineering optimizes input prompts to elicit better responses from language models without changing model weights."),
        
        # Related but less relevant
        ("doc13", "Data preprocessing includes normalization, handling missing values, and feature engineering to prepare data for modeling."),
        ("doc14", "Model evaluation metrics like precision, recall, and F1-score measure classification performance on test data."),
        ("doc15", "Hyperparameter tuning optimizes model configuration using techniques like grid search and random search."),
    ]
    
    corpus = {}
    for doc_id, content in documents:
        chunk = ContextChunk(
            content=content,
            doc_id=doc_id,
            embedding=None  # Will be set by VectorDBClient
        )
        corpus[doc_id] = chunk
    
    return corpus


class SimpleRAG:
    """
    Simple RAG implementation using semantic search.
    
    Pipeline:
    1. Embed all documents using SentenceTransformer
    2. Embed the query
    3. Find top-k most similar documents using cosine similarity
    4. Return ranked context for generation
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize RAG system.
        
        Args:
            model_name: SentenceTransformer model to use for embeddingsnow also a
        """
        print(f"Loading embedding model: {model_name}")
        self.encoder = SentenceTransformer(model_name)
        self.corpus: Dict[str, ContextChunk] = {}
        
    def embed_corpus(self, corpus: Dict[str, ContextChunk]):
        """
        Embed all documents in the corpus.
        
        Args:
            corpus: Dictionary of doc_id -> ContextChunk
        """
        print(f"\nEmbedding {len(corpus)} documents...")
        self.corpus = corpus
        
        # Extract texts in order
        doc_ids = list(corpus.keys())
        texts = [corpus[doc_id].content for doc_id in doc_ids]
        
        # Encode all at once
        embeddings = self.encoder.encode(texts, show_progress_bar=True)
        
        # Store embeddings back in chunks
        for doc_id, embedding in zip(doc_ids, embeddings):
            self.corpus[doc_id].embedding = embedding
            
        print(f"✓ Embedded {len(corpus)} documents")
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[ContextChunk, float]]:
        """
        Retrieve top-k most relevant documents for the query.
        
        Args:
            query: Search query
            top_k: Number of documents to retrieve
            
        Returns:
            List of (chunk, similarity_score) tuples, sorted by relevance
        """
        if not self.corpus:
            raise ValueError("Corpus is empty. Call embed_corpus() first.")
        
        # Encode query
        query_embedding = self.encoder.encode([query])[0]
        
        # Calculate cosine similarity with all documents
        similarities = []
        for doc_id, chunk in self.corpus.items():
            if chunk.embedding is None:
                continue
            
            # Cosine similarity
            similarity = np.dot(query_embedding, chunk.embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(chunk.embedding)
            )
            similarities.append((chunk, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k
        return similarities[:top_k]
    
    def format_context(self, retrieved_docs: List[Tuple[ContextChunk, float]]) -> str:
        """
        Format retrieved documents into a context string.
        
        Args:
            retrieved_docs: List of (chunk, score) tuples
            
        Returns:
            Formatted context string
        """
        context_parts = []
        for i, (chunk, score) in enumerate(retrieved_docs, 1):
            context_parts.append(f"[{i}] (relevance: {score:.3f}) {chunk.content}")
        
        return "\n\n".join(context_parts)
    
    def query(self, question: str, top_k: int = 5) -> str:
        """
        Complete RAG query: retrieve + format context.
        
        Args:
            question: User question
            top_k: Number of documents to retrieve
            
        Returns:
            Formatted context for LLM
        """
        print(f"\n{'='*80}")
        print(f"Query: {question}")
        print(f"{'='*80}")
        
        # Retrieve relevant documents
        retrieved = self.retrieve(question, top_k=top_k)
        
        # Format results
        print(f"\nRetrieved {len(retrieved)} documents:\n")
        context = self.format_context(retrieved)
        
        return context


def main():
    """Run RAG comparison demo with sample queries."""
    
    print("="*80)
    print("RAG COMPARISON: Basic RAG vs MeVe RAG")
    print("="*80)
    
    # Create corpus
    print("\n📚 Creating sample corpus...")
    corpus = create_sample_corpus()
    print(f"✓ Created corpus with {len(corpus)} documents\n")
    
    # Initialize Basic RAG
    print("🔧 Setting up Basic RAG...")
    basic_rag = SimpleRAG()
    basic_rag.embed_corpus(corpus)
    
    # Initialize MeVe RAG
    print("\n🔧 Setting up MeVe RAG...")
    
    # Create fresh corpus for MeVe (needs clean chunks)
    meve_corpus = create_sample_corpus()
    
    # Initialize VectorDBClient with corpus
    corpus_list = list(meve_corpus.values())
    vector_client = VectorDBClient(chunks=corpus_list, is_persistent=False)
    
    # MeVe configuration - tuned for comparison
    meve_config = MeVeConfig(
        k_init=10,           # Retrieve 10 initial candidates
        tau_relevance=0.3,   # Relevance threshold for verification
        n_min=2,             # Minimum verified chunks needed
        theta_redundancy=0.85,  # Deduplication threshold
        t_max=512            # Token budget
    )
    
    # Create MeVe engine (both vector_store and bm25_index use same corpus)
    meve_engine = MeVeEngine(
        config=meve_config,
        vector_store=meve_corpus,
        bm25_index=meve_corpus
    )
    
    print("✓ MeVe RAG ready\n")
    
    # Example queries
    queries = [
        "What is RAG and how does it work?",
        "How do neural networks learn features?",
        "What techniques are used to optimize models?",
    ]
    
    print("\n" + "="*80)
    print("RUNNING COMPARISON QUERIES")
    print("="*80)
    
    for i, query in enumerate(queries, 1):
        print(f"\n{'='*80}")
        print(f"QUERY {i}: {query}")
        print(f"{'='*80}\n")
        
        # Basic RAG
        print("🔍 BASIC RAG (Simple Vector Search)")
        print("-" * 80)
        basic_results = basic_rag.retrieve(query, top_k=3)
        for j, (chunk, score) in enumerate(basic_results, 1):
            print(f"[{j}] Score: {score:.3f}")
            print(f"    {chunk.content[:100]}...")
            print()
        
        # MeVe RAG
        print("\n🚀 MEVE RAG (5-Phase Pipeline)")
        print("-" * 80)
        meve_context = meve_engine.run(query)
        
        print("\n" + "="*80 + "\n")
        
    # Interactive comparison mode
    print("\n" + "="*80)
    print("INTERACTIVE COMPARISON MODE")
    print("Enter your questions to compare both systems (or 'quit' to exit)")
    print("="*80 + "\n")
    
    while True:
        try:
            user_query = input("\n❓ Your question: ").strip()
            if user_query.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            if not user_query:
                continue
            
            print(f"\n{'='*80}")
            print(f"Query: {user_query}")
            print(f"{'='*80}\n")
            
            # Basic RAG results
            print("🔍 BASIC RAG Results:")
            print("-" * 80)
            basic_results = basic_rag.retrieve(user_query, top_k=3)
            for j, (chunk, score) in enumerate(basic_results, 1):
                print(f"[{j}] Score: {score:.3f}")
                print(f"    {chunk.content}")
                print()
            
            # MeVe RAG results
            print("\n🚀 MEVE RAG Results:")
            print("-" * 80)
            meve_context = meve_engine.run(user_query)
            
            print("\n" + "="*80)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
