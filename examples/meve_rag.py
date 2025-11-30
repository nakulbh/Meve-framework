import json
import os
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

from meve.core.engine import MeVeEngine

# Import MeVe components
from meve.core.models import ContextChunk, MeVeConfig
from meve.services.vector_db_client import VectorDBClient
from meve.utils import get_logger


class SimpleRAGSystem:
    def __init__(self, collection_name: str = "hotpotqa_sentences"):
        self.collection_name = collection_name
        self.chunks: List[ContextChunk] = []
        self.engine: Optional[MeVeEngine] = None
        self.vector_db_client: Optional[VectorDBClient] = None
        self.openai_client: Optional[OpenAI] = None

        # Create unique instance ID for logging
        self.instance_id = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        
        # Setup logging directories
        self.log_dir = Path("logs/meve_contexts")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.session_log_dir = self.log_dir / self.instance_id
        self.session_log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize logger for this instance
        self.logger = get_logger(__name__)
        
        # Log initialization
        self._log_session_start()

        # Load chunks from ChromaDB and initialize engine
        self.load_from_chromadb()
        self.initialize_engine()
        self.initialize_openai()

    def _log_session_start(self) -> None:
        """Log session initialization details."""
        self.logger.info(f"🚀 Starting MeVe RAG Session: {self.instance_id}")
        
        session_info = {
            "session_id": self.instance_id,
            "timestamp": datetime.now().isoformat(),
            "collection_name": self.collection_name,
            "log_directory": str(self.session_log_dir),
            "status": "initialized"
        }
        
        # Save session metadata
        session_file = self.session_log_dir / "session_info.json"
        with open(session_file, "w") as f:
            json.dump(session_info, f, indent=2)
        
        self.logger.info(f"📋 Session info saved to: {session_file}")

    def load_from_chromadb(self) -> None:
        """Load chunks from ChromaDB collection."""
        self.logger.info(f"📥 Loading data from ChromaDB collection: {self.collection_name}")

        try:
            # Get ChromaDB Cloud credentials from environment variables
            chroma_config = {
                "api_key": os.getenv("CHROMA_API_KEY"),
                "tenant": os.getenv("CHROMA_TENANT"),
                "database": os.getenv("CHROMA_DATABASE"),
            }

            # Validate that all required env vars are set
            if not all(chroma_config.values()):
                missing = [k for k, v in chroma_config.items() if not v]
                raise ValueError(
                    f"Missing ChromaDB Cloud credentials: {', '.join(missing)}. "
                    f"Please set CHROMA_API_KEY, CHROMA_TENANT, and CHROMA_DATABASE environment variables in .env file."
                )

            self.vector_db_client = VectorDBClient(
                chunks=None,
                is_persistent=True,
                collection_name=self.collection_name,
                load_existing=True,
                use_cloud=True,
                cloud_config=chroma_config,
                embedding_model="all-MiniLM-L6-v2",
            )

            # Get the chunks from the client
            self.chunks = self.vector_db_client.chunks
            self.logger.info(f"✅ Loaded {len(self.chunks)} chunks from ChromaDB")
            
            # Log chunk statistics
            self._log_chunk_stats()

        except Exception as e:
            self.logger.error(f"❌ Failed to load from ChromaDB: {e}")
            self.logger.warn("💡 Make sure the ChromaDB collection exists")
            self.chunks = []
            self.vector_db_client = None

    def _log_chunk_stats(self) -> None:
        """Log statistics about loaded chunks."""
        if not self.chunks:
            return
        
        chunk_stats = {
            "total_chunks": len(self.chunks),
            "timestamp": datetime.now().isoformat(),
            "chunk_samples": [
                {
                    "doc_id": chunk.doc_id,
                    "content": chunk.content,  # Full content, no truncation
                    "content_length": len(chunk.content),
                    "has_embedding": chunk.embedding is not None,
                }
                for chunk in self.chunks[:5]  # First 5 samples
            ]
        }
        
        stats_file = self.session_log_dir / "chunk_stats.json"
        with open(stats_file, "w") as f:
            json.dump(chunk_stats, f, indent=2)
        
        self.logger.info(f"📊 Chunk statistics saved to: {stats_file}")

    def initialize_engine(self) -> None:
        """Initialize the MeVe engine with ChromaDB vector client."""
        if not self.chunks or not self.vector_db_client:
            self.logger.error("❌ No chunks or vector client available for engine initialization")
            return

        self.logger.info("⚙️  Initializing MeVe engine...")

        # Create default config (you can customize these parameters)
        config = MeVeConfig(
            k_init=100,  # Initial retrieval count
            tau_relevance=0.2,  # Relevance threshold (lowered from 0.3 for more results)
            n_min=5,  # Minimum verified docs (lowered from 50 for earlier completion)
            theta_redundancy=0.8,  # Redundancy threshold
            lambda_mmr=0.5,  # MMR lambda
            t_max=1000,  # Token budget (increased from 900 for more context)
            embedding_model="all-MiniLM-L6-v2",  # OpenAI Text Embedding 3 Small
        )

        # Log configuration
        self._log_config(config)

        try:
            # Convert chunks list to dict for BM25 index
            bm25_index = {chunk.doc_id: chunk for chunk in self.chunks}

            # Initialize with VectorDBClient for phase 1 and dict for BM25
            self.engine = MeVeEngine(
                config=config,
                vector_db_client=self.vector_db_client,  # Use ChromaDB client
                bm25_index=bm25_index,  # Use dict for BM25
            )
            self.logger.info("✅ MeVe engine initialized successfully")
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize MeVe engine: {e}", error=str(e))
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            self.engine = None

    def _log_config(self, config: MeVeConfig) -> None:
        """Log MeVe configuration."""
        config_data = {
            "timestamp": datetime.now().isoformat(),
            "k_init": config.k_init,
            "tau_relevance": config.tau_relevance,
            "n_min": config.n_min,
            "theta_redundancy": config.theta_redundancy,
            "lambda_mmr": config.lambda_mmr,
            "t_max": config.t_max,
            "embedding_model": config.embedding_model,
        }
        
        config_file = self.session_log_dir / "meve_config.json"
        with open(config_file, "w") as f:
            json.dump(config_data, f, indent=2)
        
        self.logger.info(f"⚙️  MeVe config saved to: {config_file}")

    def initialize_openai(self) -> None:
        """Initialize OpenAI client for generating natural language answers."""
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                self.logger.warn("⚠️  OPENAI_API_KEY not set - answers won't be generated")
                self.openai_client = None
                return

            self.openai_client = OpenAI(api_key=api_key)
            self.logger.info("✅ OpenAI client initialized successfully")
        except Exception as e:
            self.logger.warn(f"⚠️  Failed to initialize OpenAI client: {e}")
            self.openai_client = None

    def generate_answer_with_openai(
        self, question: str, context_chunks: List[Dict[str, Any]]
    ) -> str:
        """
        Generate a natural language answer using OpenAI based on retrieved context.

        Args:
            question: The original question
            context_chunks: List of context chunks with content and metadata

        Returns:
            Natural language answer from OpenAI or "Not enough context" message
        """
        # Check if we have sufficient context
        if not self.openai_client:
            msg = "❌ OpenAI client not available."
            self.logger.error(msg)
            return msg

        try:
            # Prepare context text from chunks
            context_text = "\n\n".join(
                [f"[Source {i + 1}] {chunk['content']}" for i, chunk in enumerate(context_chunks)]
            )

            self.logger.info(f"🤖 Generating answer with OpenAI (context chunks: {len(context_chunks)})")

            # Create a strict prompt that ONLY uses provided context
            system_prompt = """You are a helpful assistant that answers questions STRICTLY based on provided context.

IMPORTANT RULES:
1. You MUST only use information explicitly stated in the provided context.
2. If the context does not contain information needed to answer the question, respond with exactly: "Not enough context to answer this question."
3. Do not make up, infer, or use any external knowledge.
4. Cite the source number when referencing information.
5. Be clear and concise in your answers."""

            user_prompt = f"""Question: {question}

Context:
{context_text}

Answer the question ONLY using the context provided above. If the context is insufficient, state: "Not enough context to answer this question." """

            # Call OpenAI API
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",  # Using gpt-4o-mini for cost-effectiveness
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,  # Lower temperature for more deterministic, fact-based responses
                max_tokens=1000,
            )

            answer = response.choices[0].message.content
            self.logger.info(f"✅ OpenAI answer generated ({len(answer)} chars)")

            # Check if OpenAI itself says there's not enough context
            

            return answer

        except Exception as e:
            error_msg = f"❌ Error generating answer with OpenAI: {str(e)}"
            self.logger.error(error_msg, error=str(e))
            return error_msg

    def log_context(self, question: str, context_chunks: List[Dict[str, Any]], answer: str) -> None:
        """
        Log the retrieved context and answer to a JSON file.

        Args:
            question: The original question
            context_chunks: List of context chunks retrieved from MeVe
            answer: The generated answer
        """
        try:
            # Create log entry with full content (no truncation)
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "question": question,
                "context_count": len(context_chunks),
                "context_chunks": context_chunks,  # Full chunks with complete content
                "answer": answer,  # Full answer, no truncation
            }

            # Generate filename with timestamp
            filename = self.session_log_dir / f"qa_{datetime.now().strftime('%H%M%S_%f')}.json"

            # Write to file
            with open(filename, "w") as f:
                json.dump(log_entry, f, indent=2)

            self.logger.info(f"✅ Q&A logged to: {filename}")

        except Exception as e:
            self.logger.error(f"⚠️  Failed to log context: {e}", error=str(e))

    def answer_question(self, question: str) -> Dict[str, Any]:
        """
        Answer a question using the MeVe RAG pipeline and OpenAI.

        Args:
            question: The question to answer

        Returns:
            Dict containing answer, context, and metadata
        """
        if not self.engine:
            self.logger.error("❌ RAG system not properly initialized")
            return {
                "answer": "RAG system not properly initialized",
                "context": [],
                "metadata": {"error": "engine_not_initialized"},
            }

        try:
            self.logger.info(f"❓ Processing question: '{question}'")

            # Run MeVe pipeline
            final_answer, final_chunks = self.engine.run(question)
            self.logger.info(f"✅ MeVe pipeline completed - Retrieved {len(final_chunks)} chunks")
            
            # Log pipeline details
            self._log_pipeline_details(question, final_chunks, final_answer)

            # Prepare context for OpenAI
            context_list = [
                {
                    "content": chunk.content,
                    "doc_id": chunk.doc_id,
                    "title": "Unknown",  # Extract from content if available
                    "source": "unknown",
                    "relevance_score": getattr(chunk, 'relevance_score', None),
                }
                for chunk in final_chunks
            ]

            # Generate natural language answer using OpenAI
            self.logger.info("🤖 Generating natural language answer with OpenAI...")
            if context_list and self.openai_client:
                openai_answer = self.generate_answer_with_openai(question, context_list)
            elif not context_list:
                openai_answer = "❌ Not enough context available to answer this question."
                self.logger.warn("⚠️  No context retrieved from MeVe pipeline")
            else:
                openai_answer = "❌ OpenAI not configured."
                self.logger.warn("⚠️  OpenAI client not available")

            # Log the context and answer
            self.log_context(question, context_list, openai_answer)

            result = {
                "answer": openai_answer,
                "context": context_list,
                "metadata": {
                    "total_chunks": len(self.chunks),
                    "retrieved_chunks": len(final_chunks),
                    "config": {
                        "k_init": self.engine.config.k_init,
                        "tau_relevance": self.engine.config.tau_relevance,
                        "t_max": self.engine.config.t_max,
                    },
                    "meve_raw_answer": final_answer,
                },
            }
            
            self.logger.info(f"✅ Question processed successfully")
            return result

        except Exception as e:
            self.logger.error(f"❌ Error processing question: {e}", error=str(e))
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                "answer": f"Error processing question: {str(e)}",
                "context": [],
                "metadata": {"error": str(e)},
            }

    def _log_pipeline_details(self, question: str, chunks: List[ContextChunk], raw_answer: str) -> None:
        """Log detailed pipeline execution information."""
        pipeline_log = {
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "raw_answer": raw_answer,  # Full raw answer, no truncation
            "chunks_retrieved": len(chunks),
            "chunk_details": [
                {
                    "doc_id": chunk.doc_id,
                    "content": chunk.content,  # Full content, no truncation
                    "content_length": len(chunk.content),
                    "relevance_score": getattr(chunk, 'relevance_score', None),
                    "token_count": getattr(chunk, 'token_count', None),
                }
                for chunk in chunks
            ]
        }
        
        pipeline_file = self.session_log_dir / f"pipeline_{datetime.now().strftime('%H%M%S_%f')}.json"
        with open(pipeline_file, "w") as f:
            json.dump(pipeline_log, f, indent=2)
        
        self.logger.info(f"📋 Pipeline details saved to: {pipeline_file}")

    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        stats = {
            "collection_name": self.collection_name,
            "total_chunks": len(self.chunks),
            "engine_initialized": self.engine is not None,
            "vector_db_initialized": self.vector_db_client is not None,
            "openai_initialized": self.openai_client is not None,
            "session_id": self.instance_id,
            "log_directory": str(self.session_log_dir),
        }
        
        self.logger.info(f"📊 System stats: {stats}")
        return stats


def main():
    """Interactive Q&A CLI for the RAG system."""
    print("=" * 70)
    print("🤖 MeVe RAG Question-Answering System")
    print("=" * 70)

    # Initialize RAG system
    rag = SimpleRAGSystem()

    # Show statistics
    stats = rag.get_stats()
    print(f"\n📊 System Status:")
    print(f"   Session ID: {stats['session_id']}")
    print(f"   Collection: {stats['collection_name']}")
    print(f"   Total chunks: {stats['total_chunks']}")
    print(f"   Engine: {'✅ Ready' if stats['engine_initialized'] else '❌ Failed'}")
    print(f"   Logs: {stats['log_directory']}")

    if not stats["engine_initialized"]:
        print("\n❌ RAG system failed to initialize.")
        print("⚠️  Please ensure:")
        print("   • 'hotpotqa_contexts' collection exists in ChromaDB")
        print("   • Environment variables are set:")
        print("     - CHROMA_API_KEY")
        print("     - CHROMA_TENANT")
        print("     - CHROMA_DATABASE")
        return

    if not stats["openai_initialized"]:
        print("\n⚠️  OpenAI integration not available.")
        print("   Set OPENAI_API_KEY environment variable for natural language answers.")

    print("\n" + "=" * 70)
    print("💡 Ask any question and I'll search the knowledge base for answers!")
    print("   Type 'quit' or 'exit' to stop")
    print(f"   Logs saved to: {stats['log_directory']}\n")

    # Interactive Q&A loop
    question_count = 0
    while True:
        try:
            question = input("❓ Your question: ").strip()

            if not question:
                continue

            if question.lower() in ["quit", "exit", "q"]:
                print("\n👋 Thank you for using MeVe RAG! Goodbye!\n")
                break

            question_count += 1
            print("\n🔍 Searching knowledge base...")

            # Get answer
            result = rag.answer_question(question)

            # Display answer
            print("\n" + "-" * 70)
            print(f"🤖 AI-GENERATED ANSWER (from OpenAI):")
            print("-" * 70)
            print(result["answer"])
            print("-" * 70)

            # Show retrieved context
            if result["context"]:
                print(f"\n📚 RETRIEVED CONTEXT ({len(result['context'])} chunks):")
                print("-" * 70)
                for i, chunk in enumerate(result["context"], 1):
                    print(f"\n[{i}] Document: {chunk['doc_id']}")
                    if chunk.get("relevance_score"):
                        print(f"    Score: {chunk['relevance_score']:.4f}")
                    print(f"    Content:\n{chunk['content']}")  # Full content, no truncation
            else:
                print("\n📚 No relevant context found in the knowledge base.")

            # Show metadata
            meta = result["metadata"]
            print("\n" + "-" * 70)
            print("📈 RETRIEVAL STATS:")
            print(
                f"   Retrieved: {meta.get('retrieved_chunks', 0)} / {meta.get('total_chunks', 0)} chunks"
            )
            print(f"   k_init: {meta.get('config', {}).get('k_init')}")
            print(f"   tau_relevance: {meta.get('config', {}).get('tau_relevance')}")
            print(f"   t_max: {meta.get('config', {}).get('t_max')} tokens")
            print("-" * 70 + "\n")

        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Thank you for using MeVe RAG!\n")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")
            print("-" * 70 + "\n")


if __name__ == "__main__":
    main()
