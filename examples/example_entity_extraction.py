"""
Example usage of entity extraction.

This script demonstrates how to extract entities (people, locations, etc.)
from chunks and add them to the Knowledge Layer.
"""

from pathlib import Path
from entity_extractor import EntityExtractor, extract_entities_from_chunks
from knowledge_layer import read_knowledge_entries, write_knowledge_entries

# Example 1: Using spaCy (fast, free, recommended)
def example_spacy_extraction():
    """Extract entities using spaCy NER."""
    # Load chunks
    chunks = read_knowledge_entries(Path("chunks/chunks.jsonl"))
    
    # Extract entities
    extractor = EntityExtractor(method='spacy', model='en_core_web_sm')
    enriched_chunks = extractor.extract_from_chunks(chunks)
    
    # Save enriched chunks
    write_knowledge_entries(enriched_chunks, Path("chunks/chunks_with_entities.jsonl"))
    print(f"Extracted entities from {len(chunks)} chunks")
    
    # Show sample
    for chunk in enriched_chunks[:3]:
        print(f"\nChunk: {chunk.get('id')}")
        print(f"  People: {chunk.get('entities', {}).get('people', [])}")
        print(f"  Locations: {chunk.get('entities', {}).get('locations', [])}")


# Example 2: Using LLM (more flexible, slower, costs money)
def example_llm_extraction():
    """Extract entities using LLM."""
    from agentic_refiner import AgenticChunkRefiner
    
    chunks = read_knowledge_entries(Path("chunks/chunks.jsonl"))
    
    # Initialize LLM
    refiner = AgenticChunkRefiner(provider='ollama', model='llama3')
    
    # Extract entities
    extractor = EntityExtractor(method='llm', llm=refiner.llm)
    enriched_chunks = extractor.extract_from_chunks(chunks)
    
    write_knowledge_entries(enriched_chunks, Path("chunks/chunks_with_entities_llm.jsonl"))


# Example 3: Hybrid approach (spaCy + LLM refinement)
def example_hybrid_extraction():
    """Extract entities using spaCy, then refine with LLM."""
    from agentic_refiner import AgenticChunkRefiner
    
    chunks = read_knowledge_entries(Path("chunks/chunks.jsonl"))
    
    # Initialize LLM
    refiner = AgenticChunkRefiner(provider='ollama', model='llama3')
    
    # Extract entities (spaCy + LLM)
    extractor = EntityExtractor(method='hybrid', llm=refiner.llm)
    enriched_chunks = extractor.extract_from_chunks(chunks)
    
    write_knowledge_entries(enriched_chunks, Path("chunks/chunks_with_entities_hybrid.jsonl"))


# Example 4: Extract entities after agentic refinement
def example_extract_after_refinement():
    """Extract entities from refined chunks."""
    from agentic_refiner import refine_chunks_with_agentic
    
    # Load and refine chunks
    chunks = read_knowledge_entries(Path("chunks/chunks.jsonl"))
    refined_chunks = refine_chunks_with_agentic(
        chunks=chunks,
        provider='ollama',
        enable_merging=True,
        enable_metadata=True
    )
    
    # Extract entities from refined chunks
    extractor = EntityExtractor(method='spacy')
    enriched_chunks = extractor.extract_from_chunks(refined_chunks)
    
    write_knowledge_entries(enriched_chunks, Path("chunks/chunks_refined_with_entities.jsonl"))


if __name__ == "__main__":
    # Run spaCy extraction (recommended)
    print("Running spaCy entity extraction...")
    example_spacy_extraction()
