#!/usr/bin/env python3
"""
Answer Questions Script
Answers a list of questions using the MeVe RAG system and saves results to a file.
"""

from pathlib import Path
import json
from datetime import datetime
from typing import List, Dict, Any

# Import MeVe components
from meve.core.models import MeVeConfig
from meve.core.engine import MeVeEngine
from meve.services.vector_db_client import VectorDBClient


class QuestionAnsweringSystem:
    """A system to answer multiple questions using MeVe RAG."""

    def __init__(self, collection_name: str = "wik_dataset"):
        self.collection_name = collection_name
        self.engine = None
        self.vector_db_client = None
        
        # Initialize the system
        self.initialize_system()

    def initialize_system(self) -> None:
        """Initialize the MeVe RAG system with ChromaDB."""
        print("=" * 80)
        print("Initializing MeVe RAG System")
        print("=" * 80)
        
        try:
            # Initialize VectorDBClient with existing collection
            print(f"Loading ChromaDB collection: {self.collection_name}")
            self.vector_db_client = VectorDBClient(
                chunks=None,
                is_persistent=True,
                collection_name=self.collection_name,
                load_existing=True
            )
            
            chunks = self.vector_db_client.chunks
            print(f"✅ Loaded {len(chunks)} chunks from ChromaDB\n")

            # Create MeVe config
            config = MeVeConfig(
                k_init=20,              # Initial retrieval count
                tau_relevance=0.3,      # Relevance threshold
                n_min=2,                # Minimum verified docs
                theta_redundancy=0.8,   # Redundancy threshold
                lambda_mmr=0.5,         # MMR lambda
                t_max=500               # Token budget (increased for better answers)
            )

            # Convert chunks list to dict for BM25 index
            bm25_index = {chunk.doc_id: chunk for chunk in chunks}

            # Initialize MeVe engine
            self.engine = MeVeEngine(
                config=config,
                vector_db_client=self.vector_db_client,
                bm25_index=bm25_index
            )
            print("✅ MeVe engine initialized successfully\n")

        except Exception as e:
            print(f"❌ Failed to initialize system: {e}")
            import traceback
            traceback.print_exc()
            raise

    def answer_question(self, question: str) -> Dict[str, Any]:
        """
        Answer a single question using MeVe RAG.
        
        Args:
            question: The question to answer
            
        Returns:
            Dict containing the question, context, and metadata
        """
        if not self.engine:
            raise RuntimeError("MeVe engine not initialized")

        print(f"\n{'=' * 80}")
        print(f"Question: {question}")
        print(f"{'=' * 80}\n")

        # Run MeVe pipeline
        final_context = self.engine.run(question)
        
        # Get the chunks from the engine's last retrieval
        final_chunks = self.engine.last_retrieved_chunks

        # Extract source information
        sources = []
        for chunk in final_chunks:
            # Extract article title from content
            title = "Unknown"
            if chunk.content.startswith("Article: "):
                title_end = chunk.content.find("\n")
                if title_end > 0:
                    title = chunk.content[9:title_end].strip()
            
            sources.append({
                "doc_id": chunk.doc_id,
                "title": title,
                "relevance_score": chunk.relevance_score,
                "token_count": chunk.token_count
            })

        result = {
            "question": question,
            "context": final_context,
            "sources": sources,
            "total_chunks": len(final_chunks),
            "total_tokens": sum(chunk.token_count for chunk in final_chunks)
        }

        print(f"\n{'─' * 80}")
        print(f"Retrieved {len(final_chunks)} relevant chunks")
        print(f"Total tokens: {result['total_tokens']}")
        print(f"{'─' * 80}\n")

        return result

    def answer_questions(self, questions: List[str]) -> List[Dict[str, Any]]:
        """
        Answer multiple questions.
        
        Args:
            questions: List of questions to answer
            
        Returns:
            List of results for each question
        """
        results = []
        
        for i, question in enumerate(questions, 1):
            print(f"\n{'#' * 80}")
            print(f"Processing Question {i}/{len(questions)}")
            print(f"{'#' * 80}")
            
            try:
                result = self.answer_question(question)
                results.append(result)
            except Exception as e:
                print(f"❌ Error answering question: {e}")
                results.append({
                    "question": question,
                    "error": str(e),
                    "context": None,
                    "sources": []
                })
        
        return results

    def save_results(self, results: List[Dict[str, Any]], output_file: str) -> None:
        """
        Save results to a JSON file.
        
        Args:
            results: List of question-answer results
            output_file: Path to output file
        """
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        output_data = {
            "timestamp": datetime.now().isoformat(),
            "collection": self.collection_name,
            "total_questions": len(results),
            "results": results
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"\n{'=' * 80}")
        print(f"✅ Results saved to: {output_path}")
        print(f"{'=' * 80}\n")

    def create_readable_report(self, results: List[Dict[str, Any]], output_file: str) -> None:
        """
        Create a human-readable text report.
        
        Args:
            results: List of question-answer results
            output_file: Path to output file
        """
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("MeVe RAG System - Question Answering Report\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Collection: {self.collection_name}\n")
            f.write(f"Total Questions: {len(results)}\n\n")

            for i, result in enumerate(results, 1):
                f.write("=" * 80 + "\n")
                f.write(f"Question {i}: {result['question']}\n")
                f.write("=" * 80 + "\n\n")

                if "error" in result:
                    f.write(f"ERROR: {result['error']}\n\n")
                    continue

                f.write("RETRIEVED CONTEXT:\n")
                f.write("-" * 80 + "\n")
                f.write(result['context'] + "\n")
                f.write("-" * 80 + "\n\n")

                f.write(f"SOURCES ({result['total_chunks']} chunks, {result['total_tokens']} tokens):\n")
                for j, source in enumerate(result['sources'], 1):
                    f.write(f"  {j}. {source['title']}\n")
                    f.write(f"     - Relevance: {source['relevance_score']:.3f}\n")
                    f.write(f"     - Tokens: {source['token_count']}\n")
                    f.write(f"     - Doc ID: {source['doc_id']}\n")
                f.write("\n\n")

        print(f"✅ Readable report saved to: {output_path}\n")


def main():
    """Main function to run the question answering system."""
    
    # Define the questions
    questions = [
        "Who is Aristotle and what is he known for?",
        "What is Albert Einstein's theory of relativity?",
        "What are the main contributions of Aristotle to science?",
        "Explain the concept of anatomy and its history",
        "What is the role of an astronomer?",
        "What are the Academy Awards and when did they start?",
        "What is agricultural science and what does it study?",
        "How did Einstein influence modern physics?",
        "What were Aristotle's views on logic and philosophy?",
        "What is the history of the Academy Awards ceremony?"
    ]

    print("\n" + "=" * 80)
    print("MeVe RAG Question Answering System")
    print("=" * 80)
    print(f"\nTotal questions to answer: {len(questions)}\n")

    # Initialize the system
    qa_system = QuestionAnsweringSystem(collection_name="wik_dataset")

    # Answer all questions
    results = qa_system.answer_questions(questions)

    # Save results
    result_dir = Path(__file__).parent.parent / "result"
    
    # Save JSON results
    json_file = result_dir / "question_answers.json"
    qa_system.save_results(results, str(json_file))

    # Save readable report
    report_file = result_dir / "question_answers_report.txt"
    qa_system.create_readable_report(results, str(report_file))

    print("\n" + "=" * 80)
    print("Processing Complete!")
    print("=" * 80)
    print(f"\nFiles created:")
    print(f"  - JSON: {json_file}")
    print(f"  - Report: {report_file}")
    print("\n")


if __name__ == "__main__":
    main()
