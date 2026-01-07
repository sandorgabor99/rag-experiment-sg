"""
Unified chunking module - main entry point for text chunking.

This module provides chunking strategies:
- Simple token-based (fast, deterministic, with sentence boundary awareness)
- Sentence-aware (splits sentences first, then groups - guarantees complete sentences)
- Hungarian-aware (automatically routes to person_chunking or document_chunking based on content)

The Hungarian-aware strategy automatically detects:
- Name lists (alapitok.txt, osszestag.txt) → person_chunking
- Descriptive documents (selmec.txt) → document_chunking
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import re

import tiktoken

from src.knowledge_layer import create_knowledge_entry, write_knowledge_entries

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# chunk_advanced and chunk_improved modules have been removed
# Only simple and sentence-aware strategies are available


def chunk_text_simple(text: str, chunk_size: int, overlap: int, encoder) -> List[str]:
    """
    Simple token-based chunking with sentence boundary awareness.
    
    This is a fast chunking method that tries to respect sentence boundaries
    when possible, but falls back to token-based splitting if needed.
    
    Args:
        text: Input text to chunk
        chunk_size: Maximum number of tokens per chunk
        overlap: Number of tokens to overlap between chunks
        encoder: Tiktoken encoder instance
    
    Returns:
        List of text chunks
    """    
    # Try to find sentence boundaries near the token limit
    tokens = encoder.encode(text)
    chunks = []
    
    start = 0
    while start < len(tokens):
        end = start + chunk_size
        
        # If we're not at the end, try to find a sentence boundary
        if end < len(tokens):
            # Decode a bit beyond the target to find sentence endings
            lookahead = min(50, len(tokens) - end)  # Look ahead up to 50 tokens
            candidate_tokens = tokens[start:end + lookahead]
            candidate_text = encoder.decode(candidate_tokens)
            
            # Find the last sentence ending before or near the target
            # Look for . ! ? followed by space or newline
            sentence_endings = list(re.finditer(r'[.!?]\s+', candidate_text))
            
            if sentence_endings:
                # Find the sentence ending closest to but before chunk_size
                target_pos = len(encoder.decode(tokens[start:end]))
                best_match = None
                
                for match in sentence_endings:
                    match_pos = match.end()
                    # Prefer endings that are before or just slightly after target
                    if match_pos <= target_pos + 20:  # Allow 20 chars flexibility
                        if best_match is None or match_pos > best_match.end():
                            best_match = match
                
                if best_match:
                    # Adjust end to the sentence boundary
                    sentence_end_text = candidate_text[:best_match.end()]
                    sentence_end_tokens = encoder.encode(sentence_end_text)
                    end = start + len(sentence_end_tokens)
        
        # Extract chunk
        chunk_tokens = tokens[start:end]
        chunk_text = encoder.decode(chunk_tokens).strip()
        
        if chunk_text:  # Only add non-empty chunks
            chunks.append(chunk_text)
        
        # Move start forward with overlap
        start += chunk_size - overlap
        
        # Ensure we make progress
        if start >= len(tokens):
            break
        if start == end - overlap:  # Prevent infinite loop
            start = end

    return chunks


def chunk_text_sentence_aware(text: str, chunk_size: int, overlap: int, encoder) -> List[str]:
    """
    Sentence-aware chunking that prioritizes complete sentences.
    
    This method splits text into sentences first, then groups sentences
    into chunks while respecting token limits. This provides better
    semantic coherence than pure token-based splitting and guarantees
    no mid-sentence breaks.
    
    Args:
        text: Input text to chunk
        chunk_size: Maximum number of tokens per chunk
        overlap: Number of tokens to overlap between chunks
        encoder: Tiktoken encoder instance
    
    Returns:
        List of text chunks
    """
    
    # Split into sentences (improved regex)
    # Pattern matches sentence endings followed by space or newline and capital letter
    sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])\n+(?=[A-Z])'
    sentences = re.split(sentence_pattern, text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if not sentences:
        return []
    
    chunks = []
    current_chunk = []
    current_tokens = 0
    
    def count_tokens(text: str) -> int:
        """Count tokens in text."""
        return len(encoder.encode(text))
    
    for sentence in sentences:
        sent_tokens = count_tokens(sentence)
        
        # If sentence is too large, split it by tokens (last resort)
        if sent_tokens > chunk_size:
            logger.warning(f"Sentence too large ({sent_tokens} tokens), splitting by tokens")
            # Finalize current chunk
            if current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_tokens = 0
            
            # Split large sentence by tokens
            sent_tokens_list = encoder.encode(sentence)
            sent_start = 0
            while sent_start < len(sent_tokens_list):
                sent_end = min(sent_start + chunk_size, len(sent_tokens_list))
                sent_chunk_tokens = sent_tokens_list[sent_start:sent_end]
                sent_chunk_text = encoder.decode(sent_chunk_tokens).strip()
                if sent_chunk_text:
                    chunks.append(sent_chunk_text)
                sent_start += chunk_size - overlap
            continue
        
        # Check if sentence fits in current chunk
        if current_tokens + sent_tokens <= chunk_size:
            current_chunk.append(sentence)
            current_tokens += sent_tokens
        else:
            # Sentence doesn't fit - finalize current chunk
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            
            # Start new chunk with overlap
            if overlap > 0 and current_chunk:
                # Include last sentence(s) for overlap
                overlap_sentences = []
                overlap_tokens = 0
                
                for prev_sent in reversed(current_chunk):
                    prev_tokens = count_tokens(prev_sent)
                    if overlap_tokens + prev_tokens <= overlap:
                        overlap_sentences.insert(0, prev_sent)
                        overlap_tokens += prev_tokens
                    else:
                        break
                
                if overlap_sentences:
                    current_chunk = overlap_sentences + [sentence]
                    current_tokens = overlap_tokens + sent_tokens
                else:
                    current_chunk = [sentence]
                    current_tokens = sent_tokens
            else:
                current_chunk = [sentence]
                current_tokens = sent_tokens
    
    # Add final chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks


# ============================================================================
# HUNGARIAN-AWARE CHUNKING (Router)
# ============================================================================
# Hungarian-aware chunking logic has been moved to:
# - person_chunking.py: for name lists (alapitok.txt, osszestag.txt)
# - document_chunking.py: for descriptive documents (selmec.txt)
# This module now acts as a router that selects the appropriate chunker.


def chunk_text_hungarian_aware(
    text: str,
    chunk_size: int,
    overlap: int,
    encoder,
    source_file: str = "unknown",
    is_json: bool = False,
    json_data: Optional[Dict[str, Any]] = None
) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Hungarian-aware chunking router.
    
    Automatically detects document type and routes to appropriate chunker:
    - JSON files with person lists (alapitok.json) → person_chunking (JSON)
    - Name lists (alapitok.txt, osszestag.txt) → person_chunking (text)
    - Descriptive documents (selmec.txt) → document_chunking
    
    Args:
        text: Hungarian text to chunk (or empty if is_json=True)
        chunk_size: Target chunk size in tokens (250-400 recommended)
        overlap: Overlap tokens (40-60 recommended)
        encoder: Tiktoken encoder instance
        source_file: Source document name for metadata
        is_json: Whether this is a JSON file (default: False)
        json_data: Parsed JSON data (required if is_json=True)
    
    Returns:
        List of (chunk_text, metadata) tuples
    """
    from src.processing.person_chunking import detect_name_list, chunk_person_list, chunk_person_json
    from src.processing.document_chunking import chunk_document
    
    # Handle JSON files
    if is_json and json_data is not None:
        # First check if this is a uniform JSON structure
        from src.processing.person_chunking import chunk_uniform_json
        uniform_keys = ['SelmeciUtodiskolak', 'egyenruhak', 'uniforms']
        is_uniform_json = any(key in json_data for key in uniform_keys)
        
        if is_uniform_json:
            logger.info(f"Detected uniform JSON structure, using uniform chunking")
            return chunk_uniform_json(json_data, source_file)
        
        # Otherwise, try person list chunking
        logger.info(f"Detected JSON file, using JSON person chunking")
        # Try to detect list key (common keys: founders, members, tags, etc.)
        possible_keys = ['founders', 'members', 'all_members', 'tags', 'people', 'persons']
        list_key = None
        for key in possible_keys:
            if key in json_data and isinstance(json_data[key], list) and len(json_data[key]) > 0:
                list_key = key
                break
        
        if list_key:
            # Infer list title from source file name
            list_title = None
            if 'alapitok' in source_file.lower():
                list_title = "Alapító tagok"
            elif 'osszestag' in source_file.lower() or 'all_members' in list_key.lower():
                list_title = "Összes aktív és passzív tag"
            elif 'tag' in source_file.lower():
                list_title = "Tagok felsorolása"
            
            return chunk_person_json(json_data, source_file, list_key=list_key, list_title=list_title)
        else:
            logger.warning(f"JSON file {source_file} does not contain a recognized person list or uniform structure. Available keys: {list(json_data.keys())}")
            return []
    
    # Check if this is a name list document (text-based)
    # Split into paragraphs first
    paragraphs = re.split(r'\n\s*\n', text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    
    # Check if any paragraph is a name list
    for para in paragraphs:
        is_name_list, list_title_detected, persons_in_list = detect_name_list(para)
        if is_name_list and persons_in_list:
            # This is a name list document - use person chunking
            logger.info(f"Detected name list document, using person chunking")
            
            # Normalize section (use list title)
            normalized_section = list_title_detected.rstrip('.,;:!?').strip()
            
            # Process all paragraphs that are name lists
            all_chunks = []
            for para in paragraphs:
                para_chunks = chunk_person_list(para, source_file, normalized_section)
                if para_chunks:
                    all_chunks.extend(para_chunks)
            
            return all_chunks if all_chunks else []
    
    # Not a name list - use document chunking
    logger.info(f"Detected descriptive document, using document chunking")
    return chunk_document(text, chunk_size, overlap, encoder, source_file)


def main():
    parser = argparse.ArgumentParser(
        description='Chunk text files into segments for embedding/LLM processing.'
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        default='clean_txt',
        help='Input directory containing cleaned text files (default: clean_txt)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='chunks',
        help='Output directory for chunk files (default: chunks)'
    )
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=400,
        help='Maximum number of tokens per chunk (default: 400)'
    )
    parser.add_argument(
        '--overlap',
        type=int,
        default=60,
        help='Number of tokens to overlap between chunks (default: 60)'
    )
    parser.add_argument(
        '--output-file',
        type=str,
        default='chunks.jsonl',
        help='Output JSONL filename (default: chunks.jsonl)'
    )
    
    # Strategy selection
    available_strategies = ['simple', 'sentence-aware']
    strategy_help = 'Chunking strategy: simple (token-based), sentence-aware (sentence boundaries first)'
    
    # Default to sentence-aware - best for agentic refiner (guarantees complete sentences)
    # Sentence-aware splits into sentences first, then groups them - prevents mid-sentence breaks
    # This is critical when using agentic refiner which can't fix broken sentences
    default_strategy = 'sentence-aware'
    
    parser.add_argument(
        '--strategy',
        type=str,
        default=default_strategy,
        choices=available_strategies,
        help=strategy_help + f' (default: {default_strategy})'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.chunk_size <= 0:
        logger.error("Chunk size must be positive")
        sys.exit(1)
    
    if args.overlap < 0:
        logger.error("Overlap must be non-negative")
        sys.exit(1)
    
    if args.overlap >= args.chunk_size:
        logger.error("Overlap must be less than chunk size")
        sys.exit(1)
    
    # Validate strategy availability
    if args.strategy not in available_strategies:
        logger.error(f"Strategy '{args.strategy}' is not available. Available strategies: {', '.join(available_strategies)}")
        sys.exit(1)
    
    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    
    # Validate input directory
    if not in_dir.exists():
        logger.error(f"Input directory does not exist: {in_dir}")
        sys.exit(1)
    
    if not in_dir.is_dir():
        logger.error(f"Input path is not a directory: {in_dir}")
        sys.exit(1)
    
    # Create output directory
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {out_dir}")
    except Exception as e:
        logger.error(f"Failed to create output directory {out_dir}: {e}")
        sys.exit(1)
    
    # Initialize encoder (needed for all strategies)
    try:
        enc = tiktoken.get_encoding("cl100k_base")
        logger.info(f"Using encoding: cl100k_base")
    except Exception as e:
        logger.error(f"Failed to initialize tiktoken encoder: {e}")
        sys.exit(1)
    
    # Initialize chunker based on strategy
    if args.strategy == 'simple':
        logger.info("Using simple token-based chunking (with sentence boundary awareness)")
    elif args.strategy == 'sentence-aware':
        logger.info("Using sentence-aware chunking (splits sentences first, then groups)")
    else:
        logger.error(f"Unknown strategy: {args.strategy}")
        sys.exit(1)
    
    # Find all text files
    txt_files = list(in_dir.glob("*.txt"))
    
    if not txt_files:
        logger.warning(f"No .txt files found in {in_dir}")
        return
    
    logger.info(f"Found {len(txt_files)} text file(s) to process")
    logger.info(f"Strategy: {args.strategy}, Chunk size: {args.chunk_size} tokens, Overlap: {args.overlap} tokens")
    
    # Process each file
    all_chunks = []
    processed = 0
    failed = 0
    
    for file in txt_files:
        try:
            logger.info(f"Processing: {file.name}")
            text = file.read_text(encoding="utf-8")
            
            # Chunk based on strategy
            if args.strategy == 'simple':
                chunks = chunk_text_simple(text, args.chunk_size, args.overlap, enc)
            elif args.strategy == 'sentence-aware':
                chunks = chunk_text_sentence_aware(text, args.chunk_size, args.overlap, enc)
            else:
                logger.error(f"Unknown strategy: {args.strategy}")
                sys.exit(1)
            
            # Create Knowledge Layer entries
            for i, c in enumerate(chunks):
                entry = create_knowledge_entry(
                    source=file.name,
                    chunk_index=i,
                    text=c
                )
                all_chunks.append(entry)
            
            processed += 1
            logger.info(f"Successfully processed: {file.name} ({len(chunks)} chunks)")
        except Exception as e:
            failed += 1
            logger.error(f"Failed to process {file.name}: {e}", exc_info=True)
    
    # Write output file
    output_file = out_dir / args.output_file
    try:
        logger.info(f"Writing {len(all_chunks)} chunks to {output_file}")
        write_knowledge_entries(all_chunks, output_file, validate=True)
        logger.info(f"Successfully wrote chunks to {output_file}")
    except Exception as e:
        logger.error(f"Failed to write output file {output_file}: {e}", exc_info=True)
        sys.exit(1)
    
    logger.info(f"Processing complete: {processed} files succeeded, {failed} failed")
    logger.info(f"Total chunks created: {len(all_chunks)}")
    
    # Print statistics
    if all_chunks:
        token_counts = [len(enc.encode(entry["text"])) for entry in all_chunks]
        logger.info(f"Chunk statistics:")
        logger.info(f"  Average tokens: {sum(token_counts) / len(token_counts):.1f}")
        logger.info(f"  Min tokens: {min(token_counts)}")
        logger.info(f"  Max tokens: {max(token_counts)}")
        logger.info(f"  Std deviation: {(sum((x - sum(token_counts)/len(token_counts))**2 for x in token_counts) / len(token_counts))**0.5:.1f}")


if __name__ == "__main__":
    main()
