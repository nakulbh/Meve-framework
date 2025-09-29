# meve_engine.py

from meve_data import ContextChunk, Query, MeVeConfig
from phase1_knn import execute_phase_1
from phase2_verification import execute_phase_2
from phase3_fallback import execute_phase_3
from phase4_prioritization import execute_phase_4
from phase5_budgeting import execute_phase_5
from typing import Dict, Tuple
import numpy as np

class MeVeEngine:
    """Orchestrates the five-phase MeVe pipeline."""
    
    def __init__(self, config: MeVeConfig, vector_store: Dict[str, ContextChunk], bm25_index: Dict[str, ContextChunk]):
        self.config = config
        self.vector_store = vector_store
        self.bm25_index = bm25_index

    def run(self, query_text: str) -> str:
        
        # 0. Initialize and Vectorize Query
        # In a real system, the query would be vectorized here
        query_vector = np.random.rand(768).tolist() 
        query = Query(text=query_text, vector=query_vector)
        print(f"\n--- Running MeVe Pipeline for Query: '{query_text}' ---\n")

        # --- Phase 1: Initial Retrieval (kNN Search) ---
        initial_candidates = execute_phase_1(query, self.config, self.vector_store)

        # --- Phase 2: Relevance Verification (Cross-Encoder) ---
        verified_chunks = execute_phase_2(query, initial_candidates, self.config)
        
        combined_context = verified_chunks

        # --- Conditional Phase 3: Fallback Retrieval (BM25) ---
        # Logic: If |C_ver| < N_min, trigger fallback
        if len(verified_chunks) < self.config.n_min:
            print(f"\n--- Condition Met: |C_ver| ({len(verified_chunks)}) < N_min ({self.config.n_min}). Triggering Fallback ---")
            fallback_chunks = execute_phase_3(query, self.bm25_index)
            
            # Combine Context: C_all = C_ver U C_fallback
            combined_context.extend(fallback_chunks)
            print(f"Total Combined Context Chunks: {len(combined_context)}")
        else:
            print(f"\n--- Condition Not Met: |C_ver| ({len(verified_chunks)}) >= N_min ({self.config.n_min}). Skipping Fallback ---")
        
        if not combined_context:
            return "Error: No context could be retrieved or verified."

        # --- Phase 4: Context Prioritization (Relevance/Redundancy) ---
        prioritized_context = execute_phase_4(query, combined_context, self.config)

        # --- Phase 5: Token Budgeting (Greedy Packing) ---
        final_context_string, final_chunks = execute_phase_5(prioritized_context, self.config)
        
        # Final Output (The context passed to the LLM)
        print("\n================= FINAL CONTEXT FOR LLM ==================")
        print(f"Total Final Chunks: {len(final_chunks)}")
        print(f"Context Snippet: {final_context_string[:200]}...")
        # In a full RAG system, the LLM would now generate the answer using this context.
        return final_context_string

# ----------------- SIMULATION SETUP -----------------

def setup_simulation_data() -> Tuple[Dict[str, ContextChunk], Dict[str, ContextChunk]]:
    """Creates a simulated vector store and BM25 index."""
    
    # 1. Simulate Knowledge Base Chunks
    # Chunks 1-3 are highly relevant to the query "Eiffel Tower"
    # Chunks 4-6 are semantically similar (Paris/architecture) but less factually relevant.
    # Chunks 7-10 are general and will simulate high redundancy or low verification scores.
    
    chunks = {
        "doc1": ContextChunk(content="The Eiffel Tower, finished in 1889, stands 330 meters tall.", doc_id="doc1", embedding=None),
        "doc2": ContextChunk(content="It was built by Gustave Eiffel for the 1889 World's Fair in Paris, France.", doc_id="doc2", embedding=None),
        "doc3": ContextChunk(content="The tower is the most visited paid monument in the world.", doc_id="doc3", embedding=None),
        "doc4": ContextChunk(content="The Pantheon is a neo-classical church also located in Paris.", doc_id="doc4", embedding=None),
        "doc5": ContextChunk(content="Architecture trends in the late 19th century favored wrought iron construction.", doc_id="doc5", embedding=None),
        "doc6": ContextChunk(content="France is known for its museums and historic landmarks.", doc_id="doc6", embedding=None),
        "doc7": ContextChunk(content="A primary goal of MeVe is context efficiency and control.", doc_id="doc7", embedding=None),
        "doc8": ContextChunk(content="MeVe uses a modular decomposition of the RAG pipeline.", doc_id="doc8", embedding=None),
        "doc9": ContextChunk(content="The computational complexity scales linearly with the number of candidates k.", doc_id="doc9", embedding=None),
        "doc10": ContextChunk(content="Final context generation uses a greedy packing algorithm.", doc_id="doc10", embedding=None),
    }
    
    # The vector store and BM25 index are the same content but accessed differently
    vector_store = chunks
    bm25_index = chunks
    
    return vector_store, bm25_index

# ----------------- EXECUTION -----------------

if __name__ == "__main__":
    
    # Setup data and configuration
    vector_store, bm25_index = setup_simulation_data()
    
    # [cite_start]Configuration based on the paper's parameters [cite: 134]
    config = MeVeConfig(
        k_init=5,           # Initial k-search candidates
        tau_relevance=0.5, # Verification threshold (τ)
        n_min=3,             # Minimum verified docs to avoid fallback (Nmin)
        theta_redundancy=0.85, # Redundancy threshold
        t_max=100           # Small token budget for demonstration
    )
    
    # --- SCENARIO 1: Success (Minimal Fallback Needed) ---
    # The query is designed to hit the highly-scored chunks (doc1-3)
    query_a = "What is the building material and height of the Eiffel Tower?"
    engine_a = MeVeEngine(config, vector_store, bm25_index)
    final_context_a = engine_a.run(query_a)
    print("\n--------------------------------------------------------------")
    print(f"Scenario 1 (Expected: Verification successful, Fallback SKIPPED) finished.")
    print("--------------------------------------------------------------")

    # --- SCENARIO 2: Failure (Fallback REQUIRED) ---
    # The query is general and will likely get poor relevance scores (e.g., hitting doc7-10)
    # This simulates semantic drift or a low relevance core.
    query_b = "Tell me about MeVe's architectural features and complexity."
    
    # To force the fallback, we must change the Phase 2 simulation logic (in phase_2_verification.py)
    # For this demonstration, we rely on the generic scores being low for this query.
    engine_b = MeVeEngine(config, vector_store, bm25_index)
    final_context_b = engine_b.run(query_b)
    print("\n--------------------------------------------------------------")
    print(f"Scenario 2 (Expected: Verification fails, Fallback TRIGGERED) finished.")
    print("--------------------------------------------------------------")
