"""
Search module for finding similar chunks using embeddings.

This module provides functionality to:
- Load embeddings from Knowledge Layer entries
- Perform similarity search using cosine similarity or dot-product
- Return top-k most similar chunks
- Support metadata filtering
- Use persistent VectorStore for efficient search
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Callable

try:
    import numpy as np
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Error: sentence-transformers and numpy are required.")
    print("Install with: pip install sentence-transformers numpy")
    sys.exit(1)

from src.knowledge_layer import read_knowledge_entries
from src.search.vector_store import VectorStore, load_vector_store_from_embeddings, create_metadata_filter

try:
    from src.metrics.models import RetrievalMetrics
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False
    RetrievalMetrics = None

# Optional context builder
try:
    from src.search.context_builder import ContextBuilder, build_context_from_search
    CONTEXT_BUILDER_AVAILABLE = True
except ImportError:
    CONTEXT_BUILDER_AVAILABLE = False
    # logger not yet defined here, will log later if needed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class EmbeddingSearcher:
    """
    Searcher for finding similar chunks using embeddings.
    
    Uses VectorStore for efficient, persistent search with metadata filtering.
    """
    
    def __init__(
        self,
        entries: Optional[List[Dict[str, Any]]] = None,
        vector_store: Optional[VectorStore] = None,
        store_path: Optional[Path] = None
    ):
        """
        Initialize the searcher.
        
        Args:
            entries: List of Knowledge Layer entries with 'embedding' field (optional)
            vector_store: Pre-initialized VectorStore (optional)
            store_path: Path to load/save VectorStore (optional)
        """
        if vector_store is not None:
            self.store = vector_store
        elif store_path is not None and store_path.exists():
            # Load from persistent store
            self.store = VectorStore()
            self.store.load(store_path)
            logger.info(f"Loaded VectorStore from {store_path}")
        elif entries is not None:
            # Create store from entries
            if not entries:
                raise ValueError("No entries provided")
            
            # Validate entries have embeddings
            for entry in entries:
                if "embedding" not in entry:
                    raise ValueError(f"Entry {entry.get('id', 'unknown')} missing embedding")
            
            # Extract vectors and metadata
            vectors = np.array([entry["embedding"] for entry in entries], dtype=np.float32)
            metadata_list = [
                {k: v for k, v in entry.items() if k != 'embedding'}
                for entry in entries
            ]
            ids = [entry['id'] for entry in entries]
            
            # Create VectorStore
            self.store = VectorStore(
                embedding_dim=vectors.shape[1],
                normalize_embeddings=True,
                persist_path=store_path
            )
            self.store.add_batch(vectors, metadata_list, ids)
            logger.info(f"Created VectorStore with {len(entries)} entries")
        else:
            raise ValueError("Must provide either entries, vector_store, or valid store_path")
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        similarity_metric: str = 'cosine',
        metadata_filter: Optional[Callable[[Dict[str, Any]], bool]] = None
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Find top-k most similar entries to a query embedding.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            similarity_metric: 'cosine' or 'dot_product'
            metadata_filter: Optional filter function for metadata
        
        Returns:
            List of (entry, similarity_score) tuples, sorted by similarity (descending)
        """
        return self.store.search(
            query_embedding,
            top_k=top_k,
            similarity_metric=similarity_metric,
            metadata_filter=metadata_filter
        )
    
    def search_by_text(
        self,
        query_text: str,
        model: SentenceTransformer,
        top_k: int = 5,
        similarity_metric: str = 'cosine',
        metadata_filter: Optional[Callable[[Dict[str, Any]], bool]] = None,
        boost_entity_names: bool = True,
        return_metrics: bool = False
    ) -> Tuple[List[Tuple[Dict[str, Any], float]], Optional[RetrievalMetrics]]:
        """
        Search using query text (embeds the text first).
        
        Args:
            query_text: Query text string
            model: SentenceTransformer model to embed the query
            top_k: Number of results to return
            similarity_metric: 'cosine' or 'dot_product'
            metadata_filter: Optional filter function for metadata
            boost_entity_names: Whether to boost results containing entity names from query
            return_metrics: Whether to return retrieval metrics
        
        Returns:
            Tuple of (results list, optional RetrievalMetrics)
        """
        start_time = time.perf_counter()
        
        # Embed query text
        embed_start = time.perf_counter()
        query_embedding = model.encode(
            query_text,
            convert_to_numpy=True,
            normalize_embeddings=(similarity_metric == 'cosine')
        )
        embed_time_ms = (time.perf_counter() - embed_start) * 1000
        
        # Get semantic search results
        search_start = time.perf_counter()
        results = self.search(query_embedding, top_k * 2, similarity_metric, metadata_filter)  # Get more results for boosting
        search_time_ms = (time.perf_counter() - search_start) * 1000
        
        # Calculate metrics
        metrics = None
        if return_metrics and METRICS_AVAILABLE:
            # Calculate similarity statistics
            if results:
                similarities = [score for _, score in results]
                top_k_avg = sum(similarities[:top_k]) / min(len(similarities), top_k) if similarities else 0.0
                top_1 = similarities[0] if similarities else 0.0
                spread = top_1 - top_k_avg
                
                # Calculate hit density (same source vs diverse)
                sources = [r[0].get('source', '') for r in results[:top_k]]
                unique_sources = len(set(sources))
                hit_density = unique_sources / len(sources) if sources else 0.0
            else:
                top_k_avg = 0.0
                spread = 0.0
                hit_density = 0.0
            
            metrics = RetrievalMetrics(
                top_k_similarity_avg=top_k_avg,
                top_1_vs_top_k_spread=spread,
                entity_boost_applied=boost_entity_names,
                hit_density=hit_density,
                query_length=len(query_text),
                retrieval_time_ms=search_time_ms,
                results_count=len(results)
            )
        
        # Boost results that contain entity names from the query
        if boost_entity_names and results:
            import re
            # Extract potential entity names (capitalized words, Hungarian names)
            entity_pattern = r'\b([A-ZÁÉÍÓÖŐÚÜŰ][a-záéíóöőúüű]+(?:\s+[A-ZÁÉÍÓÖŐÚÜŰ][a-záéíóöőúüű]+)*)\b'
            potential_entities = re.findall(entity_pattern, query_text)
            
            if potential_entities:
                boosted_results = []
                for entry, score in results:
                    text = entry.get('text', '').lower()
                    name = entry.get('name', '').lower()
                    entity_name = entry.get('entity_name', '').lower()
                    
                    # Check if any entity from query appears in chunk
                    boost = 0.0
                    for entity in potential_entities:
                        entity_lower = entity.lower()
                        # Boost if entity name appears in text, name, or entity_name fields
                        if entity_lower in text:
                            boost += 0.15  # Significant boost for text match
                        if entity_lower in name or entity_lower in entity_name:
                            boost += 0.25  # Even more boost for exact name match
                    
                    # Apply boost
                    boosted_score = score + boost
                    boosted_results.append((entry, boosted_score))
                
                # Re-sort by boosted score
                boosted_results.sort(key=lambda x: x[1], reverse=True)
                # Return top_k
                final_results = boosted_results[:top_k]
            else:
                final_results = results[:top_k]
        else:
            final_results = results[:top_k]
        
        # Update metrics with final results if needed
        if metrics and return_metrics:
            metrics.results_count = len(final_results)
            metrics.retrieval_time_ms = (time.perf_counter() - start_time) * 1000
        
        if return_metrics:
            return final_results, metrics
        return final_results
    
    def search_by_text_diverse(
        self,
        query_text: str,
        model: SentenceTransformer,
        top_k: int = 5,
        similarity_metric: str = 'cosine',
        metadata_filter: Optional[Callable[[Dict[str, Any]], bool]] = None,
        max_per_source: Optional[int] = None,
        boost_entity_names: bool = True,
        return_metrics: bool = False
    ) -> Tuple[List[Tuple[Dict[str, Any], float]], Optional[RetrievalMetrics]]:
        """
        Search using query text with source diversity.
        
        Ensures results come from multiple sources by:
        1. Getting more candidates (top_k * 3)
        2. Limiting results per source
        3. Returning top-k diverse results
        
        Args:
            query_text: Query text string
            model: SentenceTransformer model to embed the query
            top_k: Number of results to return
            similarity_metric: 'cosine' or 'dot_product'
            metadata_filter: Optional filter function for metadata
            max_per_source: Maximum results per source (None = auto-calculate)
            boost_entity_names: Whether to boost results containing entity names
            return_metrics: Whether to return retrieval metrics
        
        Returns:
            Tuple of (results list, optional RetrievalMetrics)
        """
        start_time = time.perf_counter()
        
        # Embed query text
        embed_start = time.perf_counter()
        query_embedding = model.encode(
            query_text,
            convert_to_numpy=True,
            normalize_embeddings=(similarity_metric == 'cosine')
        )
        embed_time_ms = (time.perf_counter() - embed_start) * 1000
        
        # Get more candidates for diversity
        search_start = time.perf_counter()
        candidate_count = max(top_k * 3, 20)
        candidates = self.search(query_embedding, candidate_count, similarity_metric, metadata_filter)
        search_time_ms = (time.perf_counter() - search_start) * 1000
        
        if not candidates:
            if return_metrics and METRICS_AVAILABLE:
                metrics = RetrievalMetrics(
                    query_length=len(query_text),
                    retrieval_time_ms=search_time_ms,
                    results_count=0
                )
                return [], metrics
            return []
        
        # Boost results that contain entity names from the query
        if boost_entity_names:
            import re
            # Extract potential entity names (capitalized words, Hungarian names)
            entity_pattern = r'\b([A-ZÁÉÍÓÖŐÚÜŰ][a-záéíóöőúüű]+(?:\s+[A-ZÁÉÍÓÖŐÚÜŰ][a-záéíóöőúüű]+)*)\b'
            potential_entities = re.findall(entity_pattern, query_text)
            
            if potential_entities:
                boosted_candidates = []
                for entry, score in candidates:
                    text = entry.get('text', '').lower()
                    name = entry.get('name', '').lower()
                    entity_name = entry.get('entity_name', '').lower()
                    
                    # Check if any entity from query appears in chunk
                    boost = 0.0
                    for entity in potential_entities:
                        entity_lower = entity.lower()
                        # Boost if entity name appears in text, name, or entity_name fields
                        if entity_lower in text:
                            boost += 0.15  # Significant boost for text match
                        if entity_lower in name or entity_lower in entity_name:
                            boost += 0.25  # Even more boost for exact name match
                    
                    # Apply boost
                    boosted_score = score + boost
                    boosted_candidates.append((entry, boosted_score))
                
                # Re-sort by boosted score
                candidates = sorted(boosted_candidates, key=lambda x: x[1], reverse=True)
        
        # Apply source diversity
        if max_per_source is None:
            # Auto-calculate: ensure at least 1 from each source, but allow more from sources with many chunks
            max_per_source = max(1, top_k // 2)
        
        # Group by source and limit per source
        source_counts = {}
        diverse_results = []
        
        for entry, score in candidates:
            source = entry.get('source', 'unknown')
            count = source_counts.get(source, 0)
            
            if count < max_per_source:
                diverse_results.append((entry, score))
                source_counts[source] = count + 1
            
            # Stop if we have enough results
            if len(diverse_results) >= top_k:
                break
        
        # If we don't have enough diverse results, fill with remaining candidates
        if len(diverse_results) < top_k:
            for entry, score in candidates:
                if (entry, score) not in diverse_results:
                    diverse_results.append((entry, score))
                    if len(diverse_results) >= top_k:
                        break
        
        logger.info(f"Diverse search: {len(diverse_results)} results from {len(source_counts)} sources: {dict(source_counts)}")
        return diverse_results[:top_k]
    
    def save_store(self, path: Optional[Path] = None) -> None:
        """Save VectorStore to disk."""
        self.store.save(path)


def main():
    parser = argparse.ArgumentParser(
        description='Search Knowledge Layer chunks using embeddings.'
    )
    parser.add_argument(
        '--embeddings-file',
        type=str,
        required=True,
        help='JSONL file with Knowledge Layer entries and embeddings'
    )
    parser.add_argument(
        '--query',
        type=str,
        required=True,
        help='Search query text'
    )
    parser.add_argument(
        '--model',
        type=str,
        # default='all-MiniLM-L6-v2',
        default='baseline',  # Changed from 7B model to smaller multilingual model
        help='Model name for embedding the query (default: all-MiniLM-L6-v2)'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=5,
        help='Number of results to return (default: 5)'
    )
    parser.add_argument(
        '--similarity-metric',
        type=str,
        choices=['cosine', 'dot_product'],
        default='cosine',
        help='Similarity metric: cosine or dot_product (default: cosine)'
    )
    parser.add_argument(
        '--source-filter',
        type=str,
        default=None,
        help='Filter results by source file name'
    )
    parser.add_argument(
        '--store-path',
        type=str,
        default=None,
        help='Path to save/load VectorStore for persistence'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Optional output JSON file for results'
    )
    parser.add_argument(
        '--build-context',
        action='store_true',
        help='Build a single context block from search results'
    )
    parser.add_argument(
        '--max-context-tokens',
        type=int,
        default=2000,
        help='Maximum tokens in context block (default: 2000)'
    )
    parser.add_argument(
        '--context-order',
        type=str,
        choices=['relevance', 'chunk_index'],
        default='relevance',
        help='Order chunks in context: relevance or chunk_index (default: relevance)'
    )
    parser.add_argument(
        '--no-reduce-redundancy',
        action='store_true',
        help='Disable redundancy reduction in context building'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.top_k <= 0:
        logger.error("top-k must be positive")
        sys.exit(1)
    
    embeddings_path = Path(args.embeddings_file)
    store_path = Path(args.store_path) if args.store_path else None
    
    # Initialize searcher (try loading from store first, then from embeddings file)
    try:
        if store_path and store_path.exists():
            logger.info(f"Loading VectorStore from {store_path}")
            searcher = EmbeddingSearcher(store_path=store_path)
        else:
            logger.info(f"Loading embeddings from {embeddings_path}")
            entries = read_knowledge_entries(embeddings_path, validate=False)
            
            # Check for embeddings
            if not entries or "embedding" not in entries[0]:
                logger.error("Entries do not contain embeddings. Run embed.py first.")
                sys.exit(1)
            
            logger.info(f"Loaded {len(entries)} entries with embeddings")
            searcher = EmbeddingSearcher(entries=entries, store_path=store_path)
            
            # Save store if path provided
            if store_path:
                searcher.save_store(store_path)
                logger.info(f"Saved VectorStore to {store_path}")
    except Exception as e:
        logger.error(f"Failed to initialize searcher: {e}", exc_info=True)
        sys.exit(1)
    
    # Load model for query embedding
    try:
        logger.info(f"Loading model: {args.model}")
        model = SentenceTransformer(args.model)
    except Exception as e:
        logger.error(f"Failed to load model: {e}", exc_info=True)
        sys.exit(1)
    
    # Perform search
    try:
        logger.info(f"Searching for: '{args.query}'")
        results = searcher.search_by_text(args.query, model, top_k=args.top_k)
        
        # Display results
        print(f"\nTop {len(results)} results for query: '{args.query}'\n")
        print("-" * 80)
        
        output_results = []
        for i, (entry, score) in enumerate(results, 1):
            print(f"\n[{i}] Score: {score:.4f}")
            print(f"ID: {entry['id']}")
            print(f"Source: {entry['source']}")
            print(f"Chunk Index: {entry['chunk_index']}")
            print(f"Text: {entry['text'][:200]}...")
            print("-" * 80)
            
            output_results.append({
                "rank": i,
                "score": score,
                "id": entry["id"],
                "source": entry["source"],
                "chunk_index": entry["chunk_index"],
                "text": entry["text"]
            })
        
        # Build context block if requested
        context_block = None
        if args.build_context:
            if not CONTEXT_BUILDER_AVAILABLE:
                logger.error("Context builder not available. Install tiktoken.")
            else:
                try:
                    builder = ContextBuilder(
                        max_tokens=args.max_context_tokens,
                        order_by=args.context_order,
                        reduce_redundancy=not args.no_reduce_redundancy
                    )
                    context_block = builder.build_context(results)
                    
                    print(f"\n{'='*80}")
                    print(f"CONTEXT BLOCK ({len(builder.encoder.encode(context_block))} tokens)")
                    print(f"{'='*80}\n")
                    try:
                        print(context_block)
                    except UnicodeEncodeError:
                        # Handle Unicode encoding issues on Windows
                        print(context_block.encode('utf-8', errors='replace').decode('utf-8', errors='replace'))
                    print(f"\n{'='*80}\n")
                except Exception as e:
                    logger.error(f"Failed to build context: {e}", exc_info=True)
        
        # Write output if requested
        if args.output:
            output_path = Path(args.output)
            output_data = {
                "query": args.query,
                "top_k": args.top_k,
                "results": output_results
            }
            
            if context_block:
                output_data["context_block"] = context_block
                output_data["context_tokens"] = len(builder.encoder.encode(context_block))
            
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Results written to {output_path}")
        
    except Exception as e:
        logger.error(f"Search failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
