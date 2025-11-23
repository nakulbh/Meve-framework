import json
import os
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

from meve.core.models import ContextChunk, MeVeConfig, Query
from meve.phases.phase1_knn import execute_phase_1
from meve.phases.phase2_verification import execute_phase_2
from meve.phases.phase3_fallback import execute_phase_3, build_corpus_stats
from meve.phases.phase4_prioritization import execute_phase_4
from meve.phases.phase5_budgeting import execute_phase_5
from meve.services.vector_db_client import VectorDBClient
from meve.utils import get_logger

logger = get_logger(__name__)


class MeVeEngine:
    """Orchestrates the five-phase MeVe pipeline."""

    def __init__(
        self,
        config: MeVeConfig,
        vector_store: Optional[Dict[str, ContextChunk]] = None,
        bm25_index: Optional[Dict[str, ContextChunk]] = None,
        vector_db_client: Optional[VectorDBClient] = None,
        vector_db_config: Optional[dict] = None,
    ):
        """
        Initialize MeVe Engine with flexible data source options.

        Args:
            config: MeVe pipeline configuration
            vector_store: Optional dict of chunks for vector search (legacy mode)
            bm25_index: Optional dict of chunks for BM25 fallback
            vector_db_client: Optional pre-initialized VectorDBClient instance
            vector_db_config: Optional config to initialize new VectorDBClient:
                - collection_name: str (default: "meve_chunks")
                - is_persistent: bool (default: False)
                - use_cloud: bool (default: False)
                - cloud_config: dict with api_key, tenant, database
                - load_existing: bool (default: False) - load from existing collection
                - chunks: List[ContextChunk] - chunks to populate (if not load_existing)

        Usage modes:
            1. Legacy: Pass vector_store and bm25_index dicts
            2. External DB: Pass vector_db_client instance
            3. Auto-connect: Pass vector_db_config with load_existing=True
            4. Auto-create: Pass vector_db_config with chunks
        """
        self.config = config
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.vector_db_client = vector_db_client
        self.last_retrieved_chunks = []
        self.bm25_corpus_stats = None  # Precomputed BM25 statistics

        if vector_db_config and not vector_db_client:
            self._init_vector_db_from_config(vector_db_config)

        # Precompute BM25 corpus statistics if bm25_index is provided
        if bm25_index:
            self._precompute_bm25_stats()

        if not vector_store and not vector_db_client:
            logger.error("No vector data source provided. Engine will require data before running.")

    def _init_vector_db_from_config(self, config: dict):
        """Initialize VectorDBClient from configuration dict."""
        try:
            self.vector_db_client = VectorDBClient(**config)

            # If loading existing collection, populate vector_store for compatibility
            if config.get("load_existing") and self.vector_db_client.chunks:
                self.vector_store = {chunk.doc_id: chunk for chunk in self.vector_db_client.chunks}
        except Exception as e:
            logger.error(f"Failed to initialize VectorDBClient: {e}")
            raise

    def _precompute_bm25_stats(self):
        """Precompute BM25 corpus statistics for better performance."""
        if self.bm25_index:
            try:
                all_chunks = list(self.bm25_index.values())
                self.bm25_corpus_stats = build_corpus_stats(all_chunks)
                logger.info(f"Precomputed BM25 stats for {len(all_chunks)} chunks")
            except Exception as e:
                logger.warning(f"Failed to precompute BM25 stats: {e}")
                self.bm25_corpus_stats = None

    def run(self, query_text: str) -> str:
        """
        Execute the MeVe pipeline on a query.

        Args:
            query_text: User's query string

        Returns:
            Final context string ready for LLM consumption
        """
        # Validate data sources
        if not self.vector_store and not self.vector_db_client:
            logger.error("No vector data source configured")
            return "Error: No vector data source configured. Please provide vector_store or vector_db_client."

        # 0. Initialize and Vectorize Query
        # Let Phase 1 handle the query vectorization with proper dimensions
        query = Query(text=query_text, vector=None)

        # --- Phase 1: Initial Retrieval (kNN Search) ---
        # Pass vector_db_client if available, otherwise use vector_store
        if self.vector_db_client:
            initial_candidates = execute_phase_1(
                query, self.config, vector_store=None, vector_db_client=self.vector_db_client
            )
        else:
            initial_candidates = execute_phase_1(query, self.config, self.vector_store)

        # --- Phase 2: Relevance Verification (Cross-Encoder) ---
        verified_chunks = execute_phase_2(query, initial_candidates, self.config)

        combined_context = verified_chunks

        # --- Conditional Phase 3: Fallback Retrieval (BM25) ---
        # Logic: If |C_ver| < N_min, trigger fallback
        if len(verified_chunks) < self.config.n_min:
            # Use bm25_index if provided, otherwise use vector_store
            fallback_source = self.bm25_index if self.bm25_index else self.vector_store
            if fallback_source:
                fallback_chunks = execute_phase_3(
                    query, fallback_source, corpus_stats=self.bm25_corpus_stats
                )
                # Combine Context: C_all = C_ver U C_fallback
                combined_context.extend(fallback_chunks)
            else:
                logger.warning("BM25 fallback unavailable - no index provided")

        if not combined_context:
            logger.warning("No relevant context retrieved for query")
            return "Error: No context could be retrieved or verified."

        # --- Phase 4: Context Prioritization (Relevance/Redundancy) ---
        prioritized_context = execute_phase_4(query, combined_context, self.config)

        # --- Phase 5: Token Budgeting (Greedy Packing) ---
        final_context_string, final_chunks = execute_phase_5(prioritized_context, self.config)

        # Store final chunks for metrics/analysis
        self.last_retrieved_chunks = final_chunks

        # Log summary
        logger.info(
            f"MeVe retrieval complete: {len(final_chunks)} chunks, {sum(c.token_count for c in final_chunks)} tokens"
        )

        return final_context_string

    def connect_to_vector_db(
        self,
        collection_name: str,
        is_persistent: bool = False,
        use_cloud: bool = False,
        cloud_config: Optional[dict] = None,
    ):
        """
        Connect to an existing vector DB collection.

        Args:
            collection_name: Name of the ChromaDB collection
            is_persistent: Whether to use persistent storage
            use_cloud: Whether to connect to ChromaDB Cloud
            cloud_config: Cloud configuration (api_key, tenant, database)
        """
        config = {
            "collection_name": collection_name,
            "is_persistent": is_persistent,
            "use_cloud": use_cloud,
            "cloud_config": cloud_config,
            "load_existing": True,
        }
        self._init_vector_db_from_config(config)

        # Also populate bm25_index from loaded chunks for fallback
        if self.vector_db_client and self.vector_db_client.chunks:
            self.bm25_index = {chunk.doc_id: chunk for chunk in self.vector_db_client.chunks}
            # Precompute BM25 stats for new index
            self._precompute_bm25_stats()

    def set_data_sources(
        self,
        vector_store: Optional[Dict[str, ContextChunk]] = None,
        bm25_index: Optional[Dict[str, ContextChunk]] = None,
        vector_db_client: Optional[VectorDBClient] = None,
    ):
        """
        Update data sources after initialization.

        Args:
            vector_store: Dict of chunks for vector search
            bm25_index: Dict of chunks for BM25 fallback
            vector_db_client: Pre-initialized VectorDBClient instance
        """
        if vector_store:
            self.vector_store = vector_store
        if bm25_index:
            self.bm25_index = bm25_index
            # Recompute BM25 stats when index changes
            self._precompute_bm25_stats()
        if vector_db_client:
            self.vector_db_client = vector_db_client


# ----------------- REAL DATA LOADING -----------------


def chunk_text(text: str, chunk_size: int = 100) -> List[str]:
    """Split text into chunks of approximately chunk_size words."""
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)

    return chunks


def load_hotpotqa_data(
    data_dir: str = "data", max_examples: int = 100
) -> Tuple[Dict[str, ContextChunk], List[Dict]]:
    """Load HotpotQA data and create context chunks."""
    # Try to load training data
    train_file = os.path.join(data_dir, "hotpot_train_v1.1.json")
    dev_file = os.path.join(data_dir, "hotpot_dev_distractor_v1.json")

    data_file = train_file if os.path.exists(train_file) else dev_file

    if not os.path.exists(data_file):
        logger.error(f"No HotpotQA data found in {data_dir}")
        raise FileNotFoundError(
            f"❌ No HotpotQA data found in {data_dir}. Please ensure you have either 'hotpot_train_v1.1.json' or 'hotpot_dev_distractor_v1.json' in the data directory."
        )

    with open(data_file, "r", encoding="utf-8") as f:
        hotpot_data = json.load(f)

    # Limit examples for faster processing
    if max_examples:
        hotpot_data = hotpot_data[:max_examples]

    chunks = {}
    questions = []

    for i, example in enumerate(tqdm(hotpot_data, desc="Processing HotpotQA")):
        # Extract question info
        questions.append(
            {
                "id": example.get("_id", f"hotpot_{i}"),
                "question": example["question"],
                "answer": example["answer"],
                "type": example.get("type", ""),
                "level": example.get("level", ""),
            }
        )

        # Process context paragraphs into chunks
        context_data = example.get("context", {})

        # Handle new format: context is a dict with 'title' and 'sentences' arrays
        if isinstance(context_data, dict):
            titles = context_data.get("title", [])
            sentences_list = context_data.get("sentences", [])

            for para_idx, (title, sentences) in enumerate(zip(titles, sentences_list)):
                # Join sentences into paragraph
                if isinstance(sentences, list):
                    paragraph_text = " ".join(sentences)
                else:
                    paragraph_text = str(sentences)

                if len(paragraph_text.strip()) > 50:
                    doc_id = f"hotpot_{i}_{para_idx}"
                    content = f"{title}: {paragraph_text}"

                    chunk = ContextChunk(content=content, doc_id=doc_id, embedding=None)
                    chunks[doc_id] = chunk

    logger.info(f"Loaded {len(chunks)} chunks and {len(questions)} questions from HotpotQA")
    return chunks, questions


def setup_meve_data(
    data_dir: str = "data", max_examples: int = 100
) -> Tuple[Dict[str, ContextChunk], Dict[str, ContextChunk], List[Dict]]:
    """Load HotpotQA data and create vector store and BM25 index for MeVe framework."""

    # Load HotpotQA data
    chunks, questions = load_hotpotqa_data(data_dir, max_examples)

    # Use the same chunks for both vector store and BM25 index
    # Both phases access the same knowledge base but with different retrieval methods
    vector_store = chunks
    bm25_index = chunks

    return vector_store, bm25_index, questions


# ----------------- EXECUTION -----------------

if __name__ == "__main__":
    # Setup data and configuration
    vector_store, bm25_index, questions = setup_meve_data(data_dir="data", max_examples=50)

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

    logger.info("\n🔧 MeVe Configuration:")
    logger.info(f"   • k_init: {config.k_init}")
    logger.info(f"   • tau_relevance: {config.tau_relevance}")
    logger.info(f"   • n_min: {config.n_min}")
    logger.info(f"   • t_max: {config.t_max}")

    # Test with real questions
    for i, query_text in enumerate(sample_questions, 1):
        engine = MeVeEngine(config, vector_store, bm25_index)
        final_context = engine.run(query_text)
