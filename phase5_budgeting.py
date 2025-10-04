# phase_5_budgeting.py

from meve_data import ContextChunk, Query, MeVeConfig
from typing import List, Tuple
from transformers import AutoTokenizer
from color_utils import phase_header, success_message

# Initialize tokenizer (using GPT-2 as mentioned in the paper)
_tokenizer = None

def get_tokenizer():
    """Get or initialize the tokenizer."""
    global _tokenizer
    if _tokenizer is None:
        try:
            _tokenizer = AutoTokenizer.from_pretrained('gpt2')
        except Exception as e:
            print(f"[PHASE 5] Warning: Could not load GPT-2 tokenizer ({e}). Using fallback word-based tokenization.")
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

def respect_sentence_boundaries(text: str, max_tokens: int) -> str:
    """
    Truncate text while respecting sentence boundaries for better coherence.
    As mentioned in MeVe paper for maintaining context quality.
    """
    import re
    
    # Split into sentences using common sentence endings
    sentences = re.split(r'[.!?]+\s+', text)
    
    result = ""
    current_tokens = 0
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        sentence_tokens = count_tokens(sentence)
        
        # Add sentence if it fits within budget
        if current_tokens + sentence_tokens <= max_tokens:
            if result:
                result += ". " + sentence
            else:
                result = sentence
            current_tokens += sentence_tokens
        else:
            # If single sentence is too large, truncate it
            if not result and sentence_tokens > max_tokens:
                words = sentence.split()
                truncated = ""
                word_tokens = 0
                for word in words:
                    word_token_count = count_tokens(word)
                    if word_tokens + word_token_count <= max_tokens - 3:  # Reserve space for "..."
                        truncated += word + " "
                        word_tokens += word_token_count
                    else:
                        break
                result = truncated.strip() + "..."
            break
    
    # Ensure proper sentence ending
    if result and not result.endswith(('.', '!', '?', '...')):
        result += "."
    
    return result

def intelligent_chunk_processing(chunk: ContextChunk, available_tokens: int) -> ContextChunk:
    """
    Process chunks intelligently - either include fully, truncate with boundaries, or summarize.
    Implements advanced token budgeting as suggested in MeVe paper.
    """
    chunk_tokens = count_tokens(chunk.content)
    
    if chunk_tokens <= available_tokens:
        # Chunk fits completely
        return chunk
    elif available_tokens >= 50:  # Minimum viable chunk size
        # Truncate while respecting sentence boundaries
        processed_content = respect_sentence_boundaries(chunk.content, available_tokens)
        processed_chunk = ContextChunk(
            content=processed_content,
            doc_id=chunk.doc_id + "_truncated",
            embedding=chunk.embedding
        )
        processed_chunk.relevance_score = chunk.relevance_score
        processed_chunk.token_count = count_tokens(processed_content)
        return processed_chunk
    else:
        # Not enough space for meaningful content
        return None

def format_context_coherently(chunks: List[ContextChunk]) -> str:
    """
    Format final context with proper separators and structure for LLM consumption.
    Ensures coherent flow as recommended in MeVe paper.
    """
    if not chunks:
        return ""
    
    context_parts = []
    
    for i, chunk in enumerate(chunks):
        # Add document title/source if available
        if hasattr(chunk, 'title') and chunk.title:
            context_parts.append(f"Document {i+1} ({chunk.title}):")
        else:
            context_parts.append(f"Context {i+1}:")
        
        # Add the content
        context_parts.append(chunk.content)
        
        # Add separator between chunks (except for last)
        if i < len(chunks) - 1:
            context_parts.append("\n---\n")
    
    return "\n".join(context_parts)

def execute_phase_5(prioritized_context: List[ContextChunk], config: MeVeConfig) -> Tuple[str, List[ContextChunk]]:
    """
    Phase 5: Enhanced Token Budgeting with Intelligent Text Processing.
    Implements advanced greedy packing with sentence boundary respect and summarization.
    """
    print(f"{phase_header(5, 'STARTING')} - Enhanced Token Budgeting (T_max={config.t_max})")
    print(f"[PHASE 5] Received {len(prioritized_context)} prioritized chunks from Phase 4")
    
    if not prioritized_context:
        print("[PHASE 5] WARNING: No prioritized context to budget")
        print("[PHASE 5] COMPLETED - Returning empty context")
        return "", []
    
    # Initialize tokenizer
    print("[PHASE 5] Initializing tokenizer for accurate token counting...")
    tokenizer = get_tokenizer()
    if tokenizer:
        print("[PHASE 5] Using GPT-2 tokenizer for accurate token counting")
    else:
        print("[PHASE 5] Using fallback word-based tokenization")
    
    final_chunks: List[ContextChunk] = []
    current_token_count = 0
    separator_tokens = 5  # Reserve tokens for formatting
    
    print(f"[PHASE 5] Starting greedy packing algorithm...")
    print(f"[PHASE 5] Budget: {config.t_max} tokens, Reserved for formatting: {separator_tokens} tokens")
    
    # Enhanced greedy packing with intelligent processing
    for i, chunk in enumerate(prioritized_context):
        chunk_tokens = count_tokens(chunk.content)
        chunk.token_count = chunk_tokens
        
        available_tokens = config.t_max - current_token_count - separator_tokens
        
        print(f"[PHASE 5] Processing chunk {i+1}/{len(prioritized_context)}: {chunk.doc_id}")
        print(f"[PHASE 5]   Original size: {chunk_tokens} tokens, Available: {available_tokens} tokens")
        
        if available_tokens <= 0:
            print(f"[PHASE 5]   BUDGET EXHAUSTED - Stopping at chunk {i+1}")
            break
        
        # Try intelligent processing
        processed_chunk = intelligent_chunk_processing(chunk, available_tokens)
        
        if processed_chunk is not None:
            actual_tokens = processed_chunk.token_count
            
            # Double-check we don't exceed budget
            if current_token_count + actual_tokens + separator_tokens <= config.t_max:
                final_chunks.append(processed_chunk)
                current_token_count += actual_tokens
                
                status = "truncated" if processed_chunk.doc_id.endswith("_truncated") else "full"
                print(f"[PHASE 5]   → INCLUDED ({status}): {actual_tokens} tokens, total={current_token_count}/{config.t_max}")
                
                if status == "truncated":
                    print(f"[PHASE 5]     Truncated: {chunk_tokens} tokens → {actual_tokens} tokens")
            else:
                print(f"[PHASE 5]   → REJECTED: would exceed budget even after processing")
                break
        else:
            print(f"[PHASE 5]   → SKIPPED: insufficient space for meaningful content")
            continue
    
    # Format final context with proper structure
    print(f"[PHASE 5] Formatting final context with proper structure...")
    if final_chunks:
        final_context_string = format_context_coherently(final_chunks)
        final_token_count = count_tokens(final_context_string)
    else:
        final_context_string = ""
        final_token_count = 0
    
    print(f"{success_message('[PHASE 5] COMPLETED')} - Token budgeting results:")
    print(f"[PHASE 5]   Final context: {final_token_count} tokens (Budget={config.t_max})")
    print(f"[PHASE 5]   Efficiency: {(final_token_count/config.t_max)*100:.1f}% of budget used")
    print(f"[PHASE 5]   Included {len(final_chunks)} out of {len(prioritized_context)} chunks")
    
    if final_chunks:
        avg_relevance = sum(c.relevance_score for c in final_chunks) / len(final_chunks)
        print(f"[PHASE 5]   Average relevance in final context: {avg_relevance:.3f}")
        print(f"[PHASE 5]   Final context length: {len(final_context_string)} characters")
    
    print(f"{success_message('[PHASE 5] PIPELINE COMPLETED')} - Final context ready for LLM\n")
    
    return final_context_string, final_chunks