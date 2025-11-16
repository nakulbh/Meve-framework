"""
MeVe RAG Example - Complete 5-Phase Retrieval Pipeline

This example demonstrates:
1. Creating a document corpus
2. Configuring the 5-phase MeVe pipeline
3. Running queries with progressive filtering
4. Analyzing phase-by-phase results
"""

from meve.core.engine import MeVeEngine
from meve.core.models import MeVeConfig, ContextChunk
from meve.services.vector_db_client import VectorDBClient
from typing import Dict, List


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


def create_meve_config(mode: str = "balanced") -> MeVeConfig:
    """
    Create MeVe configuration for different use cases.
    
    Args:
        mode: One of "balanced", "high_precision", "high_recall", "budget_constrained"
    
    Returns:
        MeVeConfig with tuned hyperparameters
    """
    configs = {
        "balanced": MeVeConfig(
            k_init=10,           # Phase 1: Retrieve top 10 candidates
            tau_relevance=0.5,   # Phase 2: Keep chunks with score > 0.5
            n_min=3,             # Phase 3: Trigger fallback if < 3 verified
            theta_redundancy=0.85,  # Phase 4: Remove chunks with > 85% similarity
            t_max=512            # Phase 5: Stay within 512 token budget
        ),
        "high_precision": MeVeConfig(
            k_init=20,           # Cast wider net initially
            tau_relevance=0.7,   # Strict relevance threshold
            n_min=5,             # Require more verified chunks
            theta_redundancy=0.8,   # Aggressive deduplication
            t_max=384            # Tighter budget
        ),
        "high_recall": MeVeConfig(
            k_init=15,
            tau_relevance=0.3,   # Lenient threshold
            n_min=2,             # Less aggressive fallback
            theta_redundancy=0.9,   # Keep more diverse content
            t_max=768            # Larger budget
        ),
        "budget_constrained": MeVeConfig(
            k_init=8,
            tau_relevance=0.6,
            n_min=2,
            theta_redundancy=0.75,
            t_max=256            # Very strict budget
        )
    }
    
    return configs.get(mode, configs["balanced"])


def run_example_query(query: str, corpus: Dict[str, ContextChunk], config: MeVeConfig):
    """
    Run a query through the MeVe pipeline and display results.
    
    Args:
        query: User question
        corpus: Document collection
        config: MeVe configuration
    """
    print(f"\n{'='*80}")
    print(f"QUERY: {query}")
    print(f"{'='*80}\n")
    
    print("Configuration:")
    print(f"  k_init (Phase 1): {config.k_init}")
    print(f"  tau_relevance (Phase 2): {config.tau_relevance}")
    print(f"  n_min (Phase 3): {config.n_min}")
    print(f"  theta_redundancy (Phase 4): {config.theta_redundancy}")
    print(f"  t_max (Phase 5): {config.t_max} tokens")
    print()
    
    # Initialize MeVe engine
    engine = MeVeEngine(
        config=config,
        vector_store=corpus,  # Currently same as bm25_index
        bm25_index=corpus
    )
    
    # Run the 5-phase pipeline
    print("Running 5-phase MeVe pipeline...\n")
    final_context = engine.run(query)
    
    print(f"\n{'='*80}")
    print("FINAL CONTEXT")
    print(f"{'='*80}\n")
    print(final_context)
    print(f"\n{'='*80}\n")


def demonstrate_all_configurations():
    """
    Demonstrate MeVe with different configurations on the same query.
    """
    print("\n" + "="*80)
    print("MeVe RAG Pipeline - Configuration Comparison")
    print("="*80)
    
    # Create corpus once
    corpus = create_sample_corpus()
    print(f"\nCreated corpus with {len(corpus)} documents\n")
    
    # Test query
    query = "How does retrieval augmented generation work with vector databases?"
    
    # Try different configurations
    modes = ["balanced", "high_precision", "high_recall", "budget_constrained"]
    
    for mode in modes:
        print(f"\n{'#'*80}")
        print(f"# MODE: {mode.upper().replace('_', ' ')}")
        print(f"{'#'*80}")
        
        config = create_meve_config(mode)
        run_example_query(query, corpus, config)
        
        print("\nPress Enter to continue to next configuration...")
        input()


def demonstrate_single_query():
    """
    Run a single query with balanced configuration.
    """
    print("\n" + "="*80)
    print("MeVe RAG Pipeline - Single Query Demo")
    print("="*80)
    
    # Create corpus
    corpus = create_sample_corpus()
    print(f"\nCreated corpus with {len(corpus)} documents")
    
    # Use balanced configuration
    config = create_meve_config("balanced")
    
    # Example queries
    queries = [
        "How does retrieval augmented generation work with vector databases?",
        "What is the difference between supervised and unsupervised learning?",
        "Explain deep learning and neural networks",
    ]
    
    print("\nAvailable example queries:")
    for i, q in enumerate(queries, 1):
        print(f"  {i}. {q}")
    
    print("\nEnter query number (1-3) or type your own question:")
    user_input = input("> ").strip()
    
    if user_input.isdigit() and 1 <= int(user_input) <= 3:
        query = queries[int(user_input) - 1]
    elif user_input:
        query = user_input
    else:
        query = queries[0]  # Default
    
    run_example_query(query, corpus, config)


def analyze_phase_behavior():
    """
    Analyze how different phases affect the final results.
    """
    print("\n" + "="*80)
    print("MeVe RAG Pipeline - Phase Behavior Analysis")
    print("="*80)
    
    corpus = create_sample_corpus()
    
    # Query that will trigger Phase 3 (fallback)
    print("\n\nScenario 1: Testing Phase 3 Fallback Trigger")
    print("-" * 80)
    
    # High threshold config that will cause fallback
    fallback_config = MeVeConfig(
        k_init=5,
        tau_relevance=0.9,  # Very strict - will trigger fallback
        n_min=4,
        theta_redundancy=0.85,
        t_max=512
    )
    
    query = "What are quantum computing applications?"  # Not well covered in corpus
    run_example_query(query, corpus, fallback_config)
    
    print("\n\nScenario 2: Testing Token Budget Constraints")
    print("-" * 80)
    
    # Very tight budget
    budget_config = MeVeConfig(
        k_init=10,
        tau_relevance=0.3,  # Lenient to get many chunks
        n_min=2,
        theta_redundancy=0.95,  # Keep diverse chunks
        t_max=150  # Very tight budget - will force truncation
    )
    
    query = "Explain machine learning concepts"
    run_example_query(query, corpus, budget_config)


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                          MeVe RAG Example                                    ║
║                    5-Phase Retrieval Pipeline Demo                           ║
╚══════════════════════════════════════════════════════════════════════════════╝

Choose a demo:
  1. Single query with balanced config (recommended for first run)
  2. Compare all configurations on same query
  3. Analyze phase behavior (fallback, budgeting)
  
Enter choice (1-3):
""")
    
    choice = input("> ").strip()
    
    if choice == "1":
        demonstrate_single_query()
    elif choice == "2":
        demonstrate_all_configurations()
    elif choice == "3":
        analyze_phase_behavior()
    else:
        print("\nInvalid choice. Running default single query demo...\n")
        demonstrate_single_query()
    
    print("\n✅ Demo complete!")
