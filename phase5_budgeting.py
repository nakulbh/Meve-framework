# phase_5_budgeting.py

from meve_data import ContextChunk, Query, MeVeConfig
from typing import List, Tuple
from transformers import AutoTokenizer

# Initialize tokenizer (using GPT-2 as mentioned in the paper)
_tokenizer = None

def get_tokenizer():
    """Get or initialize the tokenizer."""
    global _tokenizer
    if _tokenizer is None:
        try:
            _tokenizer = AutoTokenizer.from_pretrained('gpt2')
        except Exception as e:
            print(f"Warning: Could not load GPT-2 tokenizer ({e}). Using fallback word-based tokenization.")
            _tokenizer = None
    return _tokenizer

def tokenize_text(text: str) -> List[str]:
    """Tokenize text using proper tokenizer or fallback to word-based."""
    tokenizer = get_tokenizer()
    
    if tokenizer is not None:
        # Use actual GPT-2 tokenizer
        tokens = tokenizer.encode(text, add_special_tokens=False)
        return [str(token) for token in tokens]
    else:
        # Fallback to word-based tokenization
        return text.split()

def count_tokens(text: str) -> int:
    """Count tokens in text using proper tokenizer."""
    tokenizer = get_tokenizer()
    
    if tokenizer is not None:
        # Use actual GPT-2 tokenizer
        return len(tokenizer.encode(text, add_special_tokens=False))
    else:
        # Fallback to word count
        return len(text.split()) 

def execute_phase_5(prioritized_context: List[ContextChunk], config: MeVeConfig) -> Tuple[str, List[ContextChunk]]:
    """
    Phase 5: Token Budgeting (Greedy Packing).
    Implements greedy packing algorithm as described in MeVe paper using proper tokenization.
    """
    print(f"--- Phase 5: Token Budgeting (Tmax={config.t_max}) ---")
    
    if not prioritized_context:
        print("No prioritized context to budget.")
        return "", []
    
    final_context_string = ""
    final_chunks: List[ContextChunk] = []
    current_token_count = 0
    
    # Greedy packing: take chunks in order until budget exhausted
    for i, chunk in enumerate(prioritized_context):
        # Count tokens in this chunk using proper tokenizer
        chunk_tokens = count_tokens(chunk.content)
        chunk.token_count = chunk_tokens
        
        # Check if adding this chunk would exceed the budget
        if current_token_count + chunk_tokens <= config.t_max:
            # Include this chunk
            final_context_string += chunk.content
            if i < len(prioritized_context) - 1:  # Add separator except for last chunk
                final_context_string += " "
            
            current_token_count += chunk_tokens
            final_chunks.append(chunk)
            
            print(f"  Including chunk {i+1}: {chunk_tokens} tokens, total={current_token_count}/{config.t_max}")
        else:
            # Would exceed budget, stop here
            print(f"  Stopping at chunk {i+1}: would exceed budget ({current_token_count + chunk_tokens} > {config.t_max})")
            break
    
    print(f"Final context size: {current_token_count} tokens (Budget={config.t_max}).")
    print(f"Included {len(final_chunks)} out of {len(prioritized_context)} prioritized chunks.")
    
    return final_context_string, final_chunks