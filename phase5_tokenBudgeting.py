# phase_5_budgeting.py

from meve_data import ContextChunk, Query, MeVeConfig
from typing import List, Tuple
# Assume a tokenizer (like gpt2 [cite: 337]) is available

def simulate_tokenize(text: str) -> List[str]:
    """Simulate tokenization and return token count."""
    # Simple simulation: 1 word = 1 token
    return text.split() 

def execute_phase_5(prioritized_context: List[ContextChunk], config: MeVeConfig) -> Tuple[str, List[ContextChunk]]:
    """
    Phase 5: Token Budgeting (Greedy Packing).
    Concatenates the highest-priority segments up to the token limit (Tmax)[cite: 106].
    """
    print(f"--- Phase 5: Token Budgeting (Tmax={config.t_max}) ---")
    
    final_context_string = ""
    final_chunks: List[ContextChunk] = []
    current_token_count = 0
    
    for chunk in prioritized_context:
        tokens = simulate_tokenize(chunk.content)
        chunk.token_count = len(tokens)
        
        if current_token_count + chunk.token_count <= config.t_max:
            # Greedily include the highest-priority chunk [cite: 106]
            final_context_string += chunk.content + " "
            current_token_count += chunk.token_count
            final_chunks.append(chunk)
        else:
            # Stop when adding the next chunk would exceed the budget
            break
            
    print(f"Final context size: {current_token_count} tokens (Budget={config.t_max})[cite: 107].")
    return final_context_string, final_chunks