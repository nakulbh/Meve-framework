# phase_1_knn.py
# Phase 1: Preliminary Candidate Extraction (kNN Search)

from meve_data import ContextChunk, Query, MeVeConfig
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from services.vector_db_client import VectorDBClient
from color_utils import phase_header, success_message

def execute_phase_1(query: Query, config: MeVeConfig, vector_store: Dict[str, ContextChunk]) -> List[ContextChunk]:
    """
    Phase 1: Preliminary Candidate Extraction (kNN Search).
    Retrieves the k_init closest candidates based on dense similarity using Vector DB client.
    """
    print(f"\n{phase_header(1, 'STARTING')} - Initial Retrieval (kNN={config.k_init})")
    print(f"[PHASE 1] Query: '{query.text[:50]}...' ({len(query.text)} chars)")
    
    # Initialize sentence transformer for query encoding if needed
    if not query.vector:
        print("[PHASE 1] Encoding query using sentence transformer...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        query.vector = model.encode(query.text).tolist()
        print(f"[PHASE 1] Query encoded to vector (dim={len(query.vector)})")
    else:
        print(f"[PHASE 1] Using pre-computed query vector (dim={len(query.vector)})")
    
    # Get all chunks from vector store
    all_chunks = list(vector_store.values())
    print(f"[PHASE 1] Loaded {len(all_chunks)} chunks from vector store")
    
    # Initialize ChromaDB Vector DB client (replaces FAISS operations)
    print("[PHASE 1] Initializing ChromaDB Vector DB client...")
    vector_db = VectorDBClient(all_chunks)
    print("[PHASE 1] ChromaDB Vector DB client ready")
    
    # Perform kNN search using vector DB client
    k = min(config.k_init, len(all_chunks))  # Don't search for more than available
    print(f"[PHASE 1] Performing kNN search (k={k})...")
    similarities, indices = vector_db.query(query.vector, k)
    
    # Retrieve the top-k chunks
    initial_candidates = []
    print(f"[PHASE 1] Processing {len(indices)} search results:")
    for i, idx in enumerate(indices):
        chunk = all_chunks[idx]
        # Store similarity score
        chunk.relevance_score = float(similarities[i])
        initial_candidates.append(chunk)
        print(f"   [PHASE 1] Candidate {i+1}: {chunk.doc_id} (similarity={similarities[i]:.3f})")
    
    print(f"{success_message('[PHASE 1] COMPLETED')} - Retrieved {len(initial_candidates)} initial candidates")
    print(f"[PHASE 1] Best similarity score: {max(similarities):.3f}")
    print(f"[PHASE 1] Average similarity score: {sum(similarities)/len(similarities):.3f}")
    print(f"{phase_header(1, 'HANDING OFF')} to Phase 2 (Verification)\n")
    
    return initial_candidates