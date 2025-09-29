# phase_1_knn.py

from meve_data import ContextChunk, Query, MeVeConfig
from typing import List, Optional, Dict
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

def execute_phase_1(query: Query, config: MeVeConfig, vector_store: Dict[str, ContextChunk]) -> List[ContextChunk]:
    """
    Phase 1: Preliminary Candidate Extraction (kNN Search).
    Retrieves the k_init closest candidates based on dense similarity using FAISS.
    """
    print(f"--- Phase 1: Initial Retrieval (kNN={config.k_init}) ---")
    
    # Initialize sentence transformer for query encoding if needed
    if not query.vector:
        print("Encoding query using sentence transformer...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        query.vector = model.encode(query.text).tolist()
    
    # Prepare embeddings for FAISS search
    all_chunks = list(vector_store.values())
    
    # Ensure all chunks have embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = []
    chunk_ids = []
    
    for chunk in all_chunks:
        if chunk.embedding is None:
            # Generate embedding if not present
            chunk.embedding = model.encode(chunk.content).tolist()
        embeddings.append(chunk.embedding)
        chunk_ids.append(chunk.doc_id)
    
    # Convert to numpy array for FAISS
    embeddings_array = np.array(embeddings, dtype=np.float32)
    query_vector = np.array([query.vector], dtype=np.float32)
    
    # Build FAISS index
    dimension = embeddings_array.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity)
    
    # Normalize vectors for cosine similarity
    faiss.normalize_L2(embeddings_array)
    faiss.normalize_L2(query_vector)
    
    # Add embeddings to index
    index.add(embeddings_array)
    
    # Perform kNN search
    k = min(config.k_init, len(all_chunks))  # Don't search for more than available
    distances, indices = index.search(query_vector, k)
    
    # Retrieve the top-k chunks
    initial_candidates = []
    for i, idx in enumerate(indices[0]):
        chunk = all_chunks[idx]
        # Store similarity score (convert from distance)
        chunk.relevance_score = float(distances[0][i])
        initial_candidates.append(chunk)
    
    print(f"Retrieved {len(initial_candidates)} initial candidates using FAISS kNN search.")
    return initial_candidates