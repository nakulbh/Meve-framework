# meve_engine.py

from meve_data import ContextChunk, Query, MeVeConfig
from phase1_knn import execute_phase_1
from phase2_verification import execute_phase_2
from phase3_fallback import execute_phase_3
from phase4_prioritization import execute_phase_4
from phase5_budgeting import execute_phase_5
from typing import Dict, Tuple, List
import json
import os
from tqdm import tqdm
from color_utils import engine_header, success_message

class MeVeEngine:
    """Orchestrates the five-phase MeVe pipeline."""
    
    def __init__(self, config: MeVeConfig, vector_store: Dict[str, ContextChunk], bm25_index: Dict[str, ContextChunk]):
        self.config = config
        self.vector_store = vector_store
        self.bm25_index = bm25_index

    def run(self, query_text: str) -> str:
        
        print(f"\n{engine_header('====== STARTING PIPELINE ======')}")
        print(f"[MEVE ENGINE] Query: '{query_text}' ({len(query_text)} chars)")
        print(f"[MEVE ENGINE] Config: k_init={self.config.k_init}, tau={self.config.tau_relevance}, n_min={self.config.n_min}, theta={self.config.theta_redundancy}, t_max={self.config.t_max}")
        print(f"[MEVE ENGINE] Knowledge base: {len(self.vector_store)} chunks")
        
        # 0. Initialize and Vectorize Query
        # Let Phase 1 handle the query vectorization with proper dimensions
        query = Query(text=query_text, vector=None)

        # --- Phase 1: Initial Retrieval (kNN Search) ---
        print(f"\n{engine_header('EXECUTING Phase 1...')}")
        initial_candidates = execute_phase_1(query, self.config, self.vector_store)

        # --- Phase 2: Relevance Verification (Cross-Encoder) ---
        print(f"{engine_header('EXECUTING Phase 2...')}")
        verified_chunks = execute_phase_2(query, initial_candidates, self.config)
        
        combined_context = verified_chunks

        # --- Conditional Phase 3: Fallback Retrieval (BM25) ---
        # Logic: If |C_ver| < N_min, trigger fallback
        print(f"{engine_header('EVALUATING fallback condition...')}")
        if len(verified_chunks) < self.config.n_min:
            print(f"[MEVE ENGINE] CONDITION MET: |C_ver| ({len(verified_chunks)}) < N_min ({self.config.n_min}) - Triggering fallback")
            print(f"{engine_header('EXECUTING Phase 3...')}")
            fallback_chunks = execute_phase_3(query, self.bm25_index)
            
            # Combine Context: C_all = C_ver U C_fallback
            combined_context.extend(fallback_chunks)
            print(f"[MEVE ENGINE] COMBINED context: {len(verified_chunks)} verified + {len(fallback_chunks)} fallback = {len(combined_context)} total")
        else:
            print(f"[MEVE ENGINE] CONDITION NOT MET: |C_ver| ({len(verified_chunks)}) >= N_min ({self.config.n_min}) - Skipping fallback")
            print(f"[MEVE ENGINE] SKIPPING Phase 3 - Using only verified chunks")
        
        if not combined_context:
            print(f"[MEVE ENGINE] ERROR: No context could be retrieved or verified")
            return "Error: No context could be retrieved or verified."

        # --- Phase 4: Context Prioritization (Relevance/Redundancy) ---
        print(f"{engine_header('EXECUTING Phase 4...')}")
        prioritized_context = execute_phase_4(query, combined_context, self.config)

        # --- Phase 5: Token Budgeting (Greedy Packing) ---
        print(f"{engine_header('EXECUTING Phase 5...')}")
        final_context_string, final_chunks = execute_phase_5(prioritized_context, self.config)
        
        # Final Output (The context passed to the LLM)
        print(f"{success_message('[MEVE ENGINE] ====== PIPELINE COMPLETED ======')}")
        print(f"[MEVE ENGINE] Summary:")
        print(f"[MEVE ENGINE]   Phase 1: {len(initial_candidates)} initial candidates")
        print(f"[MEVE ENGINE]   Phase 2: {len(verified_chunks)} verified chunks")
        print(f"[MEVE ENGINE]   Phase 3: {'EXECUTED' if len(verified_chunks) < self.config.n_min else 'SKIPPED'}")
        print(f"[MEVE ENGINE]   Phase 4: {len(prioritized_context)} prioritized chunks")
        print(f"[MEVE ENGINE]   Phase 5: {len(final_chunks)} final chunks ({len(final_context_string)} chars)")
        print(f"\n================= FINAL CONTEXT FOR LLM ==================")
        print(f"Total Final Chunks: {len(final_chunks)}")
        print(f"Context Snippet: {final_context_string[:200]}...")
        # In a full RAG system, the LLM would now generate the answer using this context.
        return final_context_string

# ----------------- REAL DATA LOADING -----------------

def chunk_text(text: str, chunk_size: int = 100) -> List[str]:
    """Split text into chunks of approximately chunk_size words."""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
    
    return chunks

def load_hotpotqa_data(data_dir: str = "data", max_examples: int = 100) -> Tuple[Dict[str, ContextChunk], List[Dict]]:
    """Load HotpotQA data and create context chunks."""
    print(f"Loading HotpotQA data from {data_dir}...")
    
    # Try to load training data
    train_file = os.path.join(data_dir, "hotpot_train_v1.1.json")
    dev_file = os.path.join(data_dir, "hotpot_dev_distractor_v1.json")
    
    data_file = train_file if os.path.exists(train_file) else dev_file
    
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"❌ No HotpotQA data found in {data_dir}. Please ensure you have either 'hotpot_train_v1.1.json' or 'hotpot_dev_distractor_v1.json' in the data directory.")
    
    print(f"📄 Loading from {data_file}")
    
    with open(data_file, 'r', encoding='utf-8') as f:
        hotpot_data = json.load(f)
    
    # Limit examples for faster processing
    if max_examples:
        hotpot_data = hotpot_data[:max_examples]
    
    chunks = {}
    questions = []
    
    for i, example in enumerate(tqdm(hotpot_data, desc="Processing HotpotQA")):
        # Extract question info
        questions.append({
            'id': example.get('_id', f'hotpot_{i}'),
            'question': example['question'],
            'answer': example['answer'],
            'type': example.get('type', ''),
            'level': example.get('level', '')
        })
        
        # Process context paragraphs into chunks
        context_paragraphs = example.get('context', [])
        
        for para_idx, context_item in enumerate(context_paragraphs):
            if isinstance(context_item, list) and len(context_item) >= 2:
                title = context_item[0]
                sentences = context_item[1]
                
                # Join sentences into paragraph
                if isinstance(sentences, list):
                    paragraph_text = ' '.join(sentences)
                else:
                    paragraph_text = str(sentences)
                
                if len(paragraph_text.strip()) > 50:
                    doc_id = f"hotpot_{i}_{para_idx}"
                    content = f"{title}: {paragraph_text}"
                    
                    chunk = ContextChunk(
                        content=content,
                        doc_id=doc_id,
                        embedding=None
                    )
                    chunks[doc_id] = chunk
    
    print(f"✅ Loaded {len(chunks)} chunks and {len(questions)} questions from HotpotQA")
    return chunks, questions


def setup_meve_data(data_dir: str = "data", max_examples: int = 100) -> Tuple[Dict[str, ContextChunk], Dict[str, ContextChunk], List[Dict]]:
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
    
    print("🚀 MeVe Framework with Real HotpotQA Data")
    print("=" * 50)
    
    # Setup data and configuration
    vector_store, bm25_index, questions = setup_meve_data(data_dir="data", max_examples=50)
    print(f"📊 Loaded knowledge base with {len(vector_store)} chunks")
    
    # Use questions from the dataset
    sample_questions = [q['question'] for q in questions[:3]]
    
    # Configuration based on the MeVe paper
    config = MeVeConfig(
        k_init=10,           # Initial k-search candidates (increased for real data)
        tau_relevance=0.3,   # Lower threshold for real cross-encoder scores
        n_min=3,             # Minimum verified docs to avoid fallback
        theta_redundancy=0.85, # Redundancy threshold
        t_max=200           # Larger token budget for real content
    )
    
    print(f"\n🔧 MeVe Configuration:")
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
    
    print(f"\n🎉 MeVe pipeline testing completed!")
    print(f"💡 Successfully processed {len(sample_questions)} real HotpotQA questions through all 5 phases.")
    print(f"📊 Knowledge base contains {len(vector_store)} context chunks from HotpotQA dataset.")
