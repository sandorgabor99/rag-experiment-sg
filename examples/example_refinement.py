"""
Example usage of AgenticChunkRefiner

This script demonstrates how to use the agentic refiner to:
1. Merge semantically similar chunks
2. Generate titles and summaries for chunks
3. Improve semantic coherence
"""

from pathlib import Path
from agentic_refiner import AgenticChunkRefiner, refine_chunks_with_agentic
from knowledge_layer import read_knowledge_entries, write_knowledge_entries

# Example 1: Using the convenience function (simplest)
def example_simple_usage():
    """Simplest way to use the refiner."""
    # Load chunks from JSONL file
    chunks = read_knowledge_entries(Path("chunks/chunks_semantic.jsonl"))
    
    # Refine chunks using Ollama (local, free)
    refined_chunks = refine_chunks_with_agentic(
        chunks=chunks,
        provider='ollama',  # or 'openai', 'anthropic', 'custom'
        model='llama3',     # Optional: specify model
        enable_merging=True,
        enable_metadata=True
    )
    
    # Save refined chunks
    write_knowledge_entries(refined_chunks, Path("chunks/chunks_refined.jsonl"))
    print(f"Refined {len(chunks)} chunks into {len(refined_chunks)} chunks")


# Example 2: Using the class directly (more control)
def example_class_usage():
    """Using AgenticChunkRefiner class for more control."""
    # Load chunks
    chunks = read_knowledge_entries(Path("chunks/chunks.jsonl"))
    
    # Initialize refiner
    refiner = AgenticChunkRefiner(
        provider='ollama',
        model='llama3',
        temperature=0,  # Deterministic output
        enable_merging=True,
        enable_metadata=True,
        print_logging=True  # See detailed logs
    )
    
    # Refine chunks
    refined_chunks = refiner.refine_chunks(
        chunks=chunks,
        max_merge_iterations=3  # How many merge passes to try
    )
    
    # Save results
    write_knowledge_entries(refined_chunks, Path("chunks/chunks_refined.jsonl"))


# Example 3: Using OpenAI API
def example_openai_usage():
    """Using OpenAI API (requires API key)."""
    chunks = read_knowledge_entries(Path("chunks/chunks.jsonl"))
    
    refined_chunks = refine_chunks_with_agentic(
        chunks=chunks,
        provider='openai',
        api_key='your-api-key-here',  # Or set OPENAI_API_KEY env var
        model='gpt-3.5-turbo',
        enable_merging=True,
        enable_metadata=True
    )
    
    write_knowledge_entries(refined_chunks, Path("chunks/chunks_refined.jsonl"))


# Example 4: Using Anthropic Claude
def example_anthropic_usage():
    """Using Anthropic Claude API (requires API key)."""
    chunks = read_knowledge_entries(Path("chunks/chunks.jsonl"))
    
    refined_chunks = refine_chunks_with_agentic(
        chunks=chunks,
        provider='anthropic',
        api_key='your-api-key-here',  # Or set ANTHROPIC_API_KEY env var
        model='claude-3-sonnet-20240229',
        enable_merging=True,
        enable_metadata=True
    )
    
    write_knowledge_entries(refined_chunks, Path("chunks/chunks_refined.jsonl"))


# Example 5: Only metadata, no merging
def example_metadata_only():
    """Generate titles/summaries without merging chunks."""
    chunks = read_knowledge_entries(Path("chunks/chunks.jsonl"))
    
    refined_chunks = refine_chunks_with_agentic(
        chunks=chunks,
        provider='ollama',
        enable_merging=False,  # Don't merge
        enable_metadata=True   # But generate titles/summaries
    )
    
    write_knowledge_entries(refined_chunks, Path("chunks/chunks_with_metadata.jsonl"))


# Example 6: Only merging, no metadata
def example_merging_only():
    """Merge chunks without generating metadata."""
    chunks = read_knowledge_entries(Path("chunks/chunks.jsonl"))
    
    refined_chunks = refine_chunks_with_agentic(
        chunks=chunks,
        provider='ollama',
        enable_merging=True,   # Merge similar chunks
        enable_metadata=False  # Don't generate titles/summaries
    )
    
    write_knowledge_entries(refined_chunks, Path("chunks/chunks_merged.jsonl"))


if __name__ == "__main__":
    # Run the simplest example
    print("Running simple usage example...")
    example_simple_usage()
