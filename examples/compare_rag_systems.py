#!/usr/bin/env python3
"""
Compare RAG Systems
Compares Simple RAG vs MeVe RAG performance on the same set of questions.
"""

from pathlib import Path
import json
from datetime import datetime
from typing import List, Dict, Any
import time

# Import Simple RAG components
from sentence_transformers import SentenceTransformer
import chromadb

# Import MeVe components
from meve.core.models import MeVeConfig
from meve.core.engine import MeVeEngine
from meve.services.vector_db_client import VectorDBClient


class SimpleRAG:
    """Simple RAG using basic vector similarity search."""

    def __init__(self, collection_name: str = "wik_dataset", top_k: int = 5):
        self.collection_name = collection_name
        self.top_k = top_k
        self.encoder = None
        self.collection = None
        self.total_chunks = 0
        self.initialize()

    def initialize(self) -> None:
        """Initialize ChromaDB and encoder."""
        print("Initializing Simple RAG...")

        script_dir = Path(__file__).parent
        chroma_path = script_dir.parent / "chroma"

        self.chroma_client = chromadb.PersistentClient(path=str(chroma_path))
        self.collection = self.chroma_client.get_collection(name=self.collection_name)
        self.total_chunks = self.collection.count()

        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        print(f"Simple RAG loaded: {self.total_chunks} chunks")

    def retrieve(self, query: str) -> Dict[str, Any]:
        """Retrieve context using simple vector similarity."""
        start_time = time.time()

        # Encode query
        query_embedding = self.encoder.encode(query, convert_to_numpy=True).tolist()

        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding], n_results=min(self.top_k, self.total_chunks)
        )

        # Format results
        chunks = []
        context_parts = []

        if results["documents"] and results["documents"][0]:
            for i, (doc, metadata, distance) in enumerate(
                zip(results["documents"][0], results["metadatas"][0], results["distances"][0]), 1
            ):
                similarity = 1.0 / (1.0 + distance)

                # Extract title
                title = "Unknown"
                if doc.startswith("Article: "):
                    title_end = doc.find("\n")
                    if title_end > 0:
                        title = doc[9:title_end].strip()

                chunks.append(
                    {
                        "doc_id": metadata.get("doc_id", f"chunk_{i}"),
                        "title": title,
                        "similarity": similarity,
                        "distance": distance,
                        "content": doc,
                    }
                )

                context_parts.append(f"Context {i}:\n{doc}\n")

        retrieval_time = time.time() - start_time
        final_context = "\n---\n\n".join(context_parts)

        return {
            "context": final_context,
            "chunks": chunks,
            "total_chunks": len(chunks),
            "retrieval_time": retrieval_time,
        }


class MeVeRAG:
    """MeVe RAG using 5-phase pipeline."""

    def __init__(self, collection_name: str = "wik_dataset"):
        self.collection_name = collection_name
        self.engine = None
        self.vector_db_client = None
        self.initialize()

    def initialize(self) -> None:
        """Initialize MeVe engine."""
        print("Initializing MeVe RAG...")

        self.vector_db_client = VectorDBClient(
            chunks=None,
            is_persistent=True,
            collection_name=self.collection_name,
            load_existing=True,
        )

        chunks = self.vector_db_client.chunks
        print(f"MeVe RAG loaded: {len(chunks)} chunks")

        config = MeVeConfig(
            k_init=20, tau_relevance=0.3, n_min=2, theta_redundancy=0.8, lambda_mmr=0.5, t_max=500
        )

        bm25_index = {chunk.doc_id: chunk for chunk in chunks}

        self.engine = MeVeEngine(
            config=config, vector_db_client=self.vector_db_client, bm25_index=bm25_index
        )
        print("MeVe engine initialized")

    def retrieve(self, query: str) -> Dict[str, Any]:
        """Retrieve context using MeVe pipeline."""
        start_time = time.time()

        final_context = self.engine.run(query)
        final_chunks = self.engine.last_retrieved_chunks

        retrieval_time = time.time() - start_time

        chunks = []
        for chunk in final_chunks:
            title = "Unknown"
            if chunk.content.startswith("Article: "):
                title_end = chunk.content.find("\n")
                if title_end > 0:
                    title = chunk.content[9:title_end].strip()

            chunks.append(
                {
                    "doc_id": chunk.doc_id,
                    "title": title,
                    "relevance_score": chunk.relevance_score,
                    "token_count": chunk.token_count,
                    "content": chunk.content,
                }
            )

        return {
            "context": final_context,
            "chunks": chunks,
            "total_chunks": len(chunks),
            "total_tokens": sum(c["token_count"] for c in chunks),
            "retrieval_time": retrieval_time,
        }


class RAGComparison:
    """Compare Simple RAG vs MeVe RAG."""

    def __init__(self):
        print("\n" + "=" * 80)
        print("Initializing RAG Systems")
        print("=" * 80 + "\n")
        self.simple_rag = SimpleRAG(top_k=5)
        self.meve_rag = MeVeRAG()
        print()

    def compare_on_question(self, question: str) -> Dict[str, Any]:
        """Run both systems on a question and compare."""
        print(f"Question: {question}")
        print("-" * 80)

        # Run Simple RAG
        print("Running Simple RAG...", end=" ")
        simple_result = self.simple_rag.retrieve(question)
        print(
            f"Done! ({simple_result['total_chunks']} chunks, {simple_result['retrieval_time']:.3f}s)"
        )

        # Run MeVe RAG
        print("Running MeVe RAG...", end=" ")
        meve_result = self.meve_rag.retrieve(question)
        print(
            f"Done! ({meve_result['total_chunks']} chunks, {meve_result['total_tokens']} tokens, {meve_result['retrieval_time']:.3f}s)"
        )
        print()

        return {"question": question, "simple_rag": simple_result, "meve_rag": meve_result}

    def run_comparison(self, questions: List[str]) -> List[Dict[str, Any]]:
        """Run comparison on all questions."""
        results = []

        print("=" * 80)
        print("Running Comparisons")
        print("=" * 80 + "\n")

        for i, question in enumerate(questions, 1):
            print(f"[{i}/{len(questions)}]")

            try:
                result = self.compare_on_question(question)
                results.append(result)
            except Exception as e:
                print(f"Error: {e}\n")
                results.append({"question": question, "error": str(e)})

        return results

    def save_results(self, results: List[Dict[str, Any]], output_file: str) -> None:
        """Save comparison results to JSON."""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Create simplified version for JSON
        json_results = []
        for result in results:
            if "error" in result:
                json_results.append(result)
                continue

            json_result = {
                "question": result["question"],
                "simple_rag": {
                    "total_chunks": result["simple_rag"]["total_chunks"],
                    "retrieval_time": result["simple_rag"]["retrieval_time"],
                    "chunks": [
                        {
                            "doc_id": c["doc_id"],
                            "title": c["title"],
                            "similarity": c.get("similarity", 0),
                        }
                        for c in result["simple_rag"]["chunks"]
                    ],
                },
                "meve_rag": {
                    "total_chunks": result["meve_rag"]["total_chunks"],
                    "total_tokens": result["meve_rag"]["total_tokens"],
                    "retrieval_time": result["meve_rag"]["retrieval_time"],
                    "chunks": [
                        {
                            "doc_id": c["doc_id"],
                            "title": c["title"],
                            "relevance_score": c.get("relevance_score", 0),
                            "token_count": c.get("token_count", 0),
                        }
                        for c in result["meve_rag"]["chunks"]
                    ],
                },
            }
            json_results.append(json_result)

        output_data = {
            "timestamp": datetime.now().isoformat(),
            "total_questions": len(results),
            "results": json_results,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"JSON results saved: {output_path}")

    def create_comparison_report(self, results: List[Dict[str, Any]], output_file: str) -> None:
        """Create detailed comparison report."""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("RAG System Comparison Report\n")
            f.write("Simple RAG vs MeVe RAG\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Questions: {len(results)}\n\n")

            # Summary statistics
            f.write("=" * 80 + "\n")
            f.write("SUMMARY STATISTICS\n")
            f.write("=" * 80 + "\n\n")

            simple_times = []
            meve_times = []
            simple_chunks = []
            meve_chunks = []
            meve_tokens = []

            for result in results:
                if "error" not in result:
                    simple_times.append(result["simple_rag"]["retrieval_time"])
                    meve_times.append(result["meve_rag"]["retrieval_time"])
                    simple_chunks.append(result["simple_rag"]["total_chunks"])
                    meve_chunks.append(result["meve_rag"]["total_chunks"])
                    meve_tokens.append(result["meve_rag"]["total_tokens"])

            if simple_times:
                f.write(f"Simple RAG:\n")
                f.write(f"  Average retrieval time: {sum(simple_times) / len(simple_times):.3f}s\n")
                f.write(
                    f"  Average chunks retrieved: {sum(simple_chunks) / len(simple_chunks):.1f}\n"
                )
                f.write(f"  Total chunks: {sum(simple_chunks)}\n\n")

                f.write(f"MeVe RAG:\n")
                f.write(f"  Average retrieval time: {sum(meve_times) / len(meve_times):.3f}s\n")
                f.write(f"  Average chunks retrieved: {sum(meve_chunks) / len(meve_chunks):.1f}\n")
                f.write(f"  Average tokens per query: {sum(meve_tokens) / len(meve_tokens):.1f}\n")
                f.write(f"  Total chunks: {sum(meve_chunks)}\n")
                f.write(f"  Total tokens: {sum(meve_tokens)}\n\n")

                f.write(f"Comparison:\n")
                f.write(
                    f"  MeVe is {sum(meve_times) / sum(simple_times):.2f}x slower (due to 5-phase processing)\n"
                )
                f.write(f"  MeVe retrieves {sum(meve_chunks) / sum(simple_chunks):.2f}x chunks\n")
                f.write(f"  MeVe provides token-budgeted context\n\n")

            # Detailed results
            f.write("\n" + "=" * 80 + "\n")
            f.write("DETAILED COMPARISON\n")
            f.write("=" * 80 + "\n\n")

            for i, result in enumerate(results, 1):
                f.write("=" * 80 + "\n")
                f.write(f"Question {i}: {result['question']}\n")
                f.write("=" * 80 + "\n\n")

                if "error" in result:
                    f.write(f"ERROR: {result['error']}\n\n")
                    continue

                # Simple RAG results
                f.write("SIMPLE RAG:\n")
                f.write("-" * 80 + "\n")
                f.write(
                    f"Chunks: {result['simple_rag']['total_chunks']}, Time: {result['simple_rag']['retrieval_time']:.3f}s\n\n"
                )
                f.write("Sources:\n")
                for j, chunk in enumerate(result["simple_rag"]["chunks"], 1):
                    f.write(f"  {j}. {chunk['title']} (similarity: {chunk['similarity']:.3f})\n")
                f.write("\n")

                # MeVe RAG results
                f.write("MEVE RAG:\n")
                f.write("-" * 80 + "\n")
                f.write(
                    f"Chunks: {result['meve_rag']['total_chunks']}, Tokens: {result['meve_rag']['total_tokens']}, Time: {result['meve_rag']['retrieval_time']:.3f}s\n\n"
                )
                f.write("Sources:\n")
                for j, chunk in enumerate(result["meve_rag"]["chunks"], 1):
                    f.write(
                        f"  {j}. {chunk['title']} (relevance: {chunk['relevance_score']:.3f}, {chunk['token_count']} tokens)\n"
                    )
                f.write("\n")

                # Comparison
                f.write("ANALYSIS:\n")
                f.write("-" * 80 + "\n")
                time_diff = (
                    result["meve_rag"]["retrieval_time"] - result["simple_rag"]["retrieval_time"]
                )
                f.write(
                    f"Time: MeVe {time_diff:+.3f}s ({result['meve_rag']['retrieval_time'] / result['simple_rag']['retrieval_time']:.2f}x)\n"
                )
                f.write(
                    f"Chunks: Simple={result['simple_rag']['total_chunks']}, MeVe={result['meve_rag']['total_chunks']}\n"
                )

                # Check for overlapping sources
                simple_titles = {c["title"] for c in result["simple_rag"]["chunks"]}
                meve_titles = {c["title"] for c in result["meve_rag"]["chunks"]}
                overlap = simple_titles & meve_titles
                f.write(f"Overlap: {len(overlap)} common articles\n")
                if overlap:
                    f.write(f"  {', '.join(sorted(overlap))}\n")
                f.write("\n\n")

        print(f"Comparison report saved: {output_path}")


def main():
    """Main comparison function."""

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
        "What is the history of the Academy Awards ceremony?",
    ]

    print("\n" + "=" * 80)
    print("RAG System Comparison")
    print("Simple RAG vs MeVe RAG")
    print("=" * 80)
    print(f"\nQuestions: {len(questions)}\n")

    # Initialize comparison
    comparison = RAGComparison()

    # Run comparison
    results = comparison.run_comparison(questions)

    # Save results
    print("\n" + "=" * 80)
    print("Saving Results")
    print("=" * 80 + "\n")

    result_dir = Path(__file__).parent.parent / "result"

    json_file = result_dir / "rag_comparison.json"
    comparison.save_results(results, str(json_file))

    report_file = result_dir / "rag_comparison_report.txt"
    comparison.create_comparison_report(results, str(report_file))

    print("\n" + "=" * 80)
    print("Comparison Complete!")
    print("=" * 80)
    print(f"\nFiles created:")
    print(f"  - JSON: {json_file}")
    print(f"  - Report: {report_file}")
    print("\n")


if __name__ == "__main__":
    main()
