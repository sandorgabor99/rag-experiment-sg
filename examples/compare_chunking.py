"""
Comparison utility for evaluating different chunking strategies.

This script helps you compare:
- Simple token-based chunking
- Advanced recursive chunking (LangChain)
- Advanced token-based chunking (LangChain)
- Hybrid chunking (LangChain)

Use this to find the best chunking strategy for your use case.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

import tiktoken

from knowledge_layer import read_knowledge_entries

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def analyze_chunks(chunks_file: Path, strategy_name: str) -> Dict:
    """
    Analyze chunk quality metrics.
    
    Args:
        chunks_file: Path to chunks JSONL file
        strategy_name: Name of the chunking strategy
    
    Returns:
        Dictionary with analysis metrics
    """
    try:
        entries = read_knowledge_entries(chunks_file, validate=True)
    except Exception as e:
        logger.error(f"Failed to read {chunks_file}: {e}")
        return None
    
    if not entries:
        return None
    
    enc = tiktoken.get_encoding("cl100k_base")
    
    # Calculate metrics
    token_counts = [len(enc.encode(entry["text"])) for entry in entries]
    char_counts = [len(entry["text"]) for entry in entries]
    
    # Check for sentence boundaries (rough heuristic)
    sentences_per_chunk = []
    incomplete_sentences = 0
    
    for entry in entries:
        text = entry["text"]
        # Count sentences (simple heuristic)
        sentence_count = text.count('. ') + text.count('! ') + text.count('? ')
        sentences_per_chunk.append(sentence_count)
        
        # Check if chunk ends mid-sentence (rough check)
        if text and text[-1] not in ['.', '!', '?', '\n']:
            # Check if last "sentence" is incomplete
            last_period = max(
                text.rfind('. '),
                text.rfind('! '),
                text.rfind('? ')
            )
            if last_period > 0 and last_period < len(text) - 10:
                incomplete_sentences += 1
    
    metrics = {
        "strategy": strategy_name,
        "total_chunks": len(entries),
        "token_stats": {
            "mean": sum(token_counts) / len(token_counts) if token_counts else 0,
            "min": min(token_counts) if token_counts else 0,
            "max": max(token_counts) if token_counts else 0,
            "std": (sum((x - sum(token_counts)/len(token_counts))**2 for x in token_counts) / len(token_counts))**0.5 if token_counts else 0
        },
        "char_stats": {
            "mean": sum(char_counts) / len(char_counts) if char_counts else 0,
            "min": min(char_counts) if char_counts else 0,
            "max": max(char_counts) if char_counts else 0,
        },
        "sentence_stats": {
            "mean_sentences_per_chunk": sum(sentences_per_chunk) / len(sentences_per_chunk) if sentences_per_chunk else 0,
            "chunks_with_incomplete_sentences": incomplete_sentences,
            "pct_incomplete": (incomplete_sentences / len(entries) * 100) if entries else 0
        }
    }
    
    return metrics


def print_comparison(metrics_list: List[Dict]):
    """Print a formatted comparison of chunking strategies."""
    print("\n" + "=" * 80)
    print("CHUNKING STRATEGY COMPARISON")
    print("=" * 80)
    
    for metrics in metrics_list:
        if metrics is None:
            continue
        
        print(f"\n{metrics['strategy'].upper()}")
        print("-" * 80)
        print(f"Total chunks: {metrics['total_chunks']}")
        print(f"\nToken Statistics:")
        print(f"  Mean: {metrics['token_stats']['mean']:.1f}")
        print(f"  Min:  {metrics['token_stats']['min']}")
        print(f"  Max:  {metrics['token_stats']['max']}")
        print(f"  Std:  {metrics['token_stats']['std']:.1f}")
        print(f"\nSentence Quality:")
        print(f"  Mean sentences per chunk: {metrics['sentence_stats']['mean_sentences_per_chunk']:.1f}")
        print(f"  Chunks with incomplete sentences: {metrics['sentence_stats']['chunks_with_incomplete_sentences']}")
        print(f"  Percentage incomplete: {metrics['sentence_stats']['pct_incomplete']:.1f}%")
    
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS:")
    print("-" * 80)
    
    # Find best strategy
    best_strategy = None
    lowest_incomplete = float('inf')
    
    for metrics in metrics_list:
        if metrics and metrics['sentence_stats']['pct_incomplete'] < lowest_incomplete:
            lowest_incomplete = metrics['sentence_stats']['pct_incomplete']
            best_strategy = metrics['strategy']
    
    if best_strategy:
        print(f"✓ Best for semantic coherence: {best_strategy} ({lowest_incomplete:.1f}% incomplete sentences)")
    
    print("\nNote: Lower incomplete sentence percentage indicates better semantic boundaries.")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Compare different chunking strategies.'
    )
    parser.add_argument(
        '--chunk-files',
        type=str,
        nargs='+',
        required=True,
        help='Paths to chunk JSONL files to compare (format: strategy_name:path)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Optional JSON output file for metrics'
    )
    
    args = parser.parse_args()
    
    # Parse chunk files
    metrics_list = []
    
    for chunk_file_spec in args.chunk_files:
        if ':' in chunk_file_spec:
            strategy_name, file_path = chunk_file_spec.split(':', 1)
        else:
            # Use filename as strategy name
            strategy_name = Path(chunk_file_spec).stem
            file_path = chunk_file_spec
        
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.warning(f"File not found: {file_path}, skipping")
            continue
        
        logger.info(f"Analyzing {strategy_name}: {file_path}")
        metrics = analyze_chunks(file_path, strategy_name)
        if metrics:
            metrics_list.append(metrics)
    
    if not metrics_list:
        logger.error("No valid chunk files to analyze")
        sys.exit(1)
    
    # Print comparison
    print_comparison(metrics_list)
    
    # Save to file if requested
    if args.output:
        output_path = Path(args.output)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metrics_list, f, indent=2, ensure_ascii=False)
        logger.info(f"Metrics saved to {output_path}")


if __name__ == "__main__":
    main()
