#!/usr/bin/env python3
"""
MeVe Framework - Main Entry Point
Run the MeVe pipeline with sample data
"""

from meve import MeVeConfig, MeVeEngine
from meve.core.engine import setup_meve_data


def main():
    """Main execution function for MeVe framework."""

    print("🚀 MeVe Framework with Real HotpotQA Data")
    print("=" * 50)

    # Setup data and configuration
    vector_store, bm25_index, questions = setup_meve_data(data_dir="data", max_examples=50)
    print(f"📊 Loaded knowledge base with {len(vector_store)} chunks")

    # Use questions from the dataset
    sample_questions = [q["question"] for q in questions[:3]]

    # Configuration based on the MeVe paper
    config = MeVeConfig(
        k_init=10,  # Initial k-search candidates (increased for real data)
        tau_relevance=0.3,  # Lower threshold for real cross-encoder scores
        n_min=3,  # Minimum verified docs to avoid fallback
        theta_redundancy=0.85,  # Redundancy threshold
        t_max=200,  # Larger token budget for real content
    )

    print("\n🔧 MeVe Configuration:")
    print(f"   • k_init: {config.k_init}")
    print(f"   • tau_relevance: {config.tau_relevance}")
    print(f"   • n_min: {config.n_min}")
    print(f"   • t_max: {config.t_max}")

    # Test with real questions
    for i, query_text in enumerate(sample_questions, 1):
        print(f"\n{'='*60}")
        print(f"🔍 QUERY {i}: {query_text}")
        print(f"{'='*60}")

        engine = MeVeEngine(config, vector_store, bm25_index)
        final_context = engine.run(query_text)

        print(f"\n📋 Summary for Query {i}:")
        print(f"   • Final context length: {len(final_context)} characters")
        print(f"   • Query: {query_text[:60]}...")

    print("\n🎉 MeVe pipeline testing completed!")
    print(
        f"💡 Successfully processed {len(sample_questions)} real HotpotQA questions through all 5 phases."
    )
    print(f"📊 Knowledge base contains {len(vector_store)} context chunks from HotpotQA dataset.")


if __name__ == "__main__":
    main()
