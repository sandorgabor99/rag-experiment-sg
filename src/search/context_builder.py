"""
Context Builder module for processing search results into a single context block.

This module processes top-k search results and creates a single context block
that can be fed into LLaMA or other LLMs.

Features:
- Max token limit enforcement
- Redundancy reduction (deduplication, similarity filtering)
- Chunk ordering (by relevance, chunk_index, or custom)
- Single context block output
"""

import logging
import re
from typing import List, Dict, Any, Tuple, Optional, Callable, TYPE_CHECKING
import tiktoken

if TYPE_CHECKING:
    from src.config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class ContextBuilder:
    """
    Builds a single context block from search results.
    
    Applies rules:
    - Max token limit
    - Redundancy reduction
    - Chunk ordering
    """
    
    def __init__(
        self,
        max_tokens: Optional[int] = None,
        encoding_name: Optional[str] = None,
        order_by: Optional[str] = None,
        reduce_redundancy: Optional[bool] = None,
        similarity_threshold: Optional[float] = None,
        clean_text: Optional[bool] = None,
        use_sliding_window: Optional[bool] = None,
        window_overlap: Optional[int] = None,
        config: Optional['Config'] = None
    ):
        """
        Initialize context builder.
        
        Args:
            max_tokens: Maximum tokens in the final context block (overrides config)
            encoding_name: Tiktoken encoding name (overrides config)
            order_by: Ordering strategy - 'relevance', 'chunk_index', or 'custom' (overrides config)
            reduce_redundancy: Whether to reduce redundant chunks (overrides config)
            similarity_threshold: Threshold for considering chunks similar (0-1) (overrides config)
            clean_text: Whether to clean chunk text (overrides config)
            use_sliding_window: Whether to use sliding window (overrides config)
            window_overlap: Token overlap between sliding windows (overrides config)
            config: Optional Config instance
        """
        # Get config values with priority: function arg > config > default
        context_config = {}
        if config:
            context_config = config.context
        
        self.max_tokens = max_tokens if max_tokens is not None else context_config.get('max_tokens_default', 8000)
        self.encoding_name = encoding_name if encoding_name else context_config.get('encoding', 'cl100k_base')
        self.order_by = order_by if order_by else context_config.get('order_by', 'relevance')
        self.reduce_redundancy = reduce_redundancy if reduce_redundancy is not None else context_config.get('reduce_redundancy', True)
        self.similarity_threshold = similarity_threshold if similarity_threshold is not None else context_config.get('similarity_threshold', 0.7)
        self.clean_text = clean_text if clean_text is not None else context_config.get('clean_text', True)
        self.use_sliding_window = use_sliding_window if use_sliding_window is not None else context_config.get('use_sliding_window', False)
        self.window_overlap = window_overlap if window_overlap is not None else context_config.get('window_overlap', 200)
        self.similarity_threshold_person = context_config.get('similarity_threshold_person', 0.9) if config else 0.9
        
        # Initialize encoder
        try:
            self.encoder = tiktoken.get_encoding(self.encoding_name)
        except Exception as e:
            logger.error(f"Failed to load encoding {encoding_name}: {e}")
            raise
        
        logger.info(
            f"Initialized ContextBuilder: max_tokens={max_tokens}, "
            f"order_by={order_by}, reduce_redundancy={reduce_redundancy}, "
            f"similarity_threshold={similarity_threshold}, clean_text={clean_text}"
        )
    
    def build_context(
        self,
        search_results: List[Tuple[Dict[str, Any], float]],
        custom_order_key: Optional[Callable[[Dict[str, Any]], Any]] = None,
        return_metrics: bool = False
    ) -> Any:  # Returns str or Tuple[str, Optional[ContextMetrics]]
        """
        Build a single context block from search results.
        
        Args:
            search_results: List of (entry, similarity_score) tuples from search
            custom_order_key: Custom function for ordering (if order_by='custom')
        
        Returns:
            Single context block string
        """
        if not search_results:
            return ""
        
        logger.info(f"Building context from {len(search_results)} search results")
        
        # Step 1: Order chunks (ensure relevance-based ordering)
        ordered_chunks = self._order_chunks(search_results, custom_order_key)
        logger.info(f"Ordered {len(ordered_chunks)} chunks by {self.order_by}")
        
        # Step 2: Clean chunk text (if enabled)
        if self.clean_text:
            ordered_chunks = self._clean_chunks(ordered_chunks)
            logger.info(f"Cleaned text for {len(ordered_chunks)} chunks")
        
        # Step 3: Reduce redundancy (if enabled)
        if self.reduce_redundancy:
            deduplicated_chunks = self._reduce_redundancy(ordered_chunks)
            logger.info(f"Reduced to {len(deduplicated_chunks)} chunks after redundancy reduction")
        else:
            deduplicated_chunks = ordered_chunks
        
        # Step 4: Apply token limit
        final_chunks = self._apply_token_limit(deduplicated_chunks)
        logger.info(f"Selected {len(final_chunks)} chunks within token limit ({self.max_tokens})")
        
        # Step 5: Build context block
        context_block = self._build_context_block(final_chunks)
        
        # Verify token count
        token_count = len(self.encoder.encode(context_block))
        logger.info(f"Final context block: {token_count} tokens")
        
        if token_count > self.max_tokens:
            logger.warning(
                f"Context block exceeds max_tokens ({token_count} > {self.max_tokens}). "
                f"Consider reducing max_tokens or number of chunks."
            )
        
        return context_block
    
    def _clean_chunks(
        self,
        chunks: List[Tuple[Dict[str, Any], float]]
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Clean chunk text by removing extra whitespace, newlines, and HTML tags.
        
        Args:
            chunks: List of (entry, score) tuples
        
        Returns:
            List with cleaned text
        """
        cleaned = []
        for entry, score in chunks:
            text = entry.get('text', '')
            if not text:
                cleaned.append((entry, score))
                continue
            
            # Create a copy to avoid modifying original
            cleaned_entry = entry.copy()
            
            # Remove HTML tags
            text = re.sub(r'<[^>]+>', '', text)
            
            # Normalize whitespace: replace multiple spaces/newlines with single space
            text = re.sub(r'\s+', ' ', text)
            
            # Remove leading/trailing whitespace
            text = text.strip()
            
            # Remove excessive newlines (keep max 2 consecutive)
            text = re.sub(r'\n{3,}', '\n\n', text)
            
            cleaned_entry['text'] = text
            cleaned.append((cleaned_entry, score))
        
        return cleaned
    
    def _order_chunks(
        self,
        search_results: List[Tuple[Dict[str, Any], float]],
        custom_order_key: Optional[Callable[[Dict[str, Any]], Any]] = None
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Order chunks based on the specified strategy.
        Ensures relevance-based ordering is truly by similarity score.
        
        Args:
            search_results: List of (entry, score) tuples
            custom_order_key: Custom ordering function
        
        Returns:
            Ordered list of (entry, score) tuples
        """
        if self.order_by == 'relevance':
            # Ensure sorted by relevance (highest similarity score first)
            # Sort in descending order by score
            return sorted(search_results, key=lambda x: x[1], reverse=True)
        
        elif self.order_by == 'chunk_index':
            # Sort by chunk_index (ascending - chronological order)
            return sorted(
                search_results,
                key=lambda x: (
                    x[0].get('source', ''),
                    x[0].get('chunk_index', 0)
                )
            )
        
        elif self.order_by == 'custom':
            if custom_order_key is None:
                logger.warning("custom_order_key not provided, using relevance order")
                return search_results
            return sorted(search_results, key=lambda x: custom_order_key(x[0]))
        
        else:
            logger.warning(f"Unknown order_by: {self.order_by}, using relevance")
            return search_results
    
    def _reduce_redundancy(
        self,
        chunks: List[Tuple[Dict[str, Any], float]]
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Reduce redundant chunks.
        
        Strategies:
        1. Exact duplicate text removal
        2. Similar text filtering (based on token overlap)
        - Less aggressive for person chunks (entity_type='személy') to preserve all people
        
        Args:
            chunks: List of (entry, score) tuples
        
        Returns:
            Deduplicated list
        """
        if not chunks:
            return chunks
        
        deduplicated = []
        seen_texts = set()
        seen_token_sets = []
        
        for entry, score in chunks:
            text = entry.get('text', '').strip()
            
            # Skip empty text
            if not text:
                continue
            
            # Strategy 1: Exact duplicate removal
            if text in seen_texts:
                logger.debug(f"Skipping exact duplicate: {entry.get('id')}")
                continue
            
            # Strategy 2: Similar text filtering (token-based)
            # For person chunks, use a higher threshold to avoid removing different people
            entity_type = entry.get('entity_type', '')
            is_person_chunk = entity_type == 'személy'
            
            # Use stricter similarity check for person chunks (they have similar structure but different names)
            if is_person_chunk:
                # Only remove if very similar (higher threshold for person chunks)
                if self._is_similar_to_existing(text, seen_token_sets, threshold=0.9):
                    logger.debug(f"Skipping very similar person chunk: {entry.get('id')}")
                    continue
            else:
                # Use normal threshold for non-person chunks
                if self._is_similar_to_existing(text, seen_token_sets):
                    logger.debug(f"Skipping similar chunk: {entry.get('id')}")
                    continue
            
            # Add to results
            deduplicated.append((entry, score))
            seen_texts.add(text)
            
            # Store token set for similarity checking
            tokens = set(self.encoder.encode(text))
            seen_token_sets.append(tokens)
        
        return deduplicated
    
    def _is_similar_to_existing(
        self,
        text: str,
        seen_token_sets: List[set],
        threshold: Optional[float] = None
    ) -> bool:
        """
        Check if text is similar to any existing chunk based on token overlap.
        
        Args:
            text: Text to check
            seen_token_sets: List of token sets from already included chunks
            threshold: Optional custom threshold (uses self.similarity_threshold if None)
        
        Returns:
            True if similar to existing chunk
        """
        if not seen_token_sets:
            return False
        
        similarity_threshold = threshold if threshold is not None else self.similarity_threshold
        
        text_tokens = set(self.encoder.encode(text))
        
        for seen_tokens in seen_token_sets:
            # Calculate Jaccard similarity (intersection / union)
            intersection = len(text_tokens & seen_tokens)
            union = len(text_tokens | seen_tokens)
            
            if union == 0:
                continue
            
            similarity = intersection / union
            
            if similarity >= similarity_threshold:
                return True
        
        return False
    
    def _apply_token_limit(
        self,
        chunks: List[Tuple[Dict[str, Any], float]]
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Select chunks that fit within the token limit.
        For list queries, prioritizes including all chunks from the same source/section.
        
        Args:
            chunks: List of (entry, score) tuples
        
        Returns:
            List of chunks that fit within max_tokens
        """
        if not chunks:
            return chunks
        
        # Detect if this is a list query (all chunks from same source with entity_type='személy' or 'lista')
        # Group chunks by source and section
        source_groups = {}
        for entry, score in chunks:
            source = entry.get('source', 'unknown')
            section = entry.get('section', '')
            entity_type = entry.get('entity_type', '')
            
            # Group by source and section
            key = (source, section)
            if key not in source_groups:
                source_groups[key] = []
            source_groups[key].append((entry, score))
        
        # If we have a group with many person chunks (likely a list), prioritize including all of them
        selected = []
        total_tokens = 0
        processed_groups = set()
        
        # First, try to include complete groups (especially person lists)
        # Sort groups by priority: person lists first, then by size
        sorted_groups = sorted(
            source_groups.items(),
            key=lambda x: (
                'személy' in [chunk[0].get('entity_type', '') for chunk in x[1]] or 
                'lista' in [chunk[0].get('entity_type', '') for chunk in x[1]],
                len(x[1])
            ),
            reverse=True
        )
        
        for (source, section), group_chunks in sorted_groups:
            entity_types = [chunk[0].get('entity_type', '') for chunk in group_chunks]
            is_person_list = 'személy' in entity_types or 'lista' in entity_types
            
            # For person lists, try to include all chunks from the group
            if is_person_list and len(group_chunks) > 3:
                group_tokens = sum(len(self.encoder.encode(chunk[0].get('text', '').strip())) 
                                  for chunk in group_chunks if chunk[0].get('text', '').strip())
                
                # If the whole group fits, add all of it
                if total_tokens + group_tokens <= self.max_tokens:
                    selected.extend(group_chunks)
                    total_tokens += group_tokens
                    processed_groups.add((source, section))
                    logger.debug(f"Included complete group: {source}/{section} ({len(group_chunks)} chunks, {group_tokens} tokens)")
                    continue
                # Otherwise, add as many as possible from this group
                else:
                    for entry, score in group_chunks:
                        text = entry.get('text', '').strip()
                        if not text:
                            continue
                        
                        chunk_tokens = len(self.encoder.encode(text))
                        if total_tokens + chunk_tokens <= self.max_tokens:
                            selected.append((entry, score))
                            total_tokens += chunk_tokens
                        else:
                            break
                    processed_groups.add((source, section))
                    logger.debug(f"Included partial group: {source}/{section} ({len([s for s in selected if s[0].get('source') == source])} chunks)")
                    continue
        
        # Add remaining chunks that weren't in processed groups
        processed_ids = {chunk[0].get('id') for chunk in selected}
        for entry, score in chunks:
            if entry.get('id') in processed_ids:
                continue
            
            # Skip if this chunk's group was already processed
            source = entry.get('source', 'unknown')
            section = entry.get('section', '')
            if (source, section) in processed_groups:
                continue
            
            text = entry.get('text', '').strip()
            if not text:
                continue
            
            chunk_tokens = len(self.encoder.encode(text))
            
            if total_tokens + chunk_tokens > self.max_tokens:
                if not selected:
                    # Truncate first chunk if it's the only one
                    truncated_text = self._truncate_to_tokens(text, self.max_tokens)
                    if truncated_text:
                        selected.append((entry, score))
                        logger.info(
                            f"Truncated first chunk to fit token limit "
                            f"({chunk_tokens} -> {len(self.encoder.encode(truncated_text))} tokens)"
                        )
                break
            
            selected.append((entry, score))
            total_tokens += chunk_tokens
        
        return selected
    
    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """
        Truncate text to fit within token limit.
        
        Args:
            text: Text to truncate
            max_tokens: Maximum tokens
        
        Returns:
            Truncated text
        """
        tokens = self.encoder.encode(text)
        if len(tokens) <= max_tokens:
            return text
        
        # Truncate tokens
        truncated_tokens = tokens[:max_tokens]
        
        # Decode back to text
        try:
            truncated_text = self.encoder.decode(truncated_tokens)
            # Try to end at a sentence boundary
            truncated_text = self._truncate_at_sentence_boundary(truncated_text)
            return truncated_text
        except Exception:
            # Fallback: decode and truncate by characters
            return text[:max_tokens * 4]  # Rough estimate
    
    def _truncate_at_sentence_boundary(self, text: str) -> str:
        """
        Truncate text at the last sentence boundary.
        
        Args:
            text: Text to truncate
        
        Returns:
            Text truncated at sentence boundary
        """
        # Find last sentence ending
        import re
        sentence_endings = re.finditer(r'[.!?]\s+', text)
        matches = list(sentence_endings)
        
        if matches:
            # Use last sentence boundary
            last_match = matches[-1]
            return text[:last_match.end()]
        
        return text
    
    def _build_context_block(
        self,
        chunks: List[Tuple[Dict[str, Any], float]]
    ) -> str:
        """
        Build a single context block from selected chunks.
        Uses sliding window if enabled and context is too long.
        
        Args:
            chunks: List of (entry, score) tuples
        
        Returns:
            Single context block string (or first window if sliding window enabled)
        """
        if not chunks:
            return ""
        
        context_parts = []
        
        for entry, score in chunks:
            text = entry.get('text', '').strip()
            if not text:
                continue
            
            # Add chunk text
            context_parts.append(text)
        
        # Join with double newlines for separation
        context_block = "\n\n".join(context_parts)
        
        # Apply sliding window if enabled and context is too long
        if self.use_sliding_window:
            token_count = len(self.encoder.encode(context_block))
            if token_count > self.max_tokens:
                logger.info(f"Context too long ({token_count} tokens), using sliding window")
                # Return first window (most relevant chunks)
                context_block = self._get_sliding_window(context_block, chunks)
        
        return context_block
    
    def _get_sliding_window(
        self,
        full_context: str,
        chunks: List[Tuple[Dict[str, Any], float]]
    ) -> str:
        """
        Get the first sliding window from context.
        For now, returns the first window. Can be extended to return multiple windows.
        
        Args:
            full_context: Full context string
            chunks: List of chunks for reference
        
        Returns:
            First window of context
        """
        # Build context from chunks until we hit token limit
        window_parts = []
        total_tokens = 0
        
        for entry, score in chunks:
            text = entry.get('text', '').strip()
            if not text:
                continue
            
            text_tokens = len(self.encoder.encode(text))
            
            if total_tokens + text_tokens > self.max_tokens:
                break
            
            window_parts.append(text)
            total_tokens += text_tokens
        
        return "\n\n".join(window_parts)
    
    def build_context_windows(
        self,
        search_results: List[Tuple[Dict[str, Any], float]],
        custom_order_key: Optional[Callable[[Dict[str, Any]], Any]] = None
    ) -> List[str]:
        """
        Build multiple context windows using sliding window approach.
        Useful for very long contexts that need to be split.
        
        Args:
            search_results: List of (entry, similarity_score) tuples from search
            custom_order_key: Custom function for ordering (if order_by='custom')
        
        Returns:
            List of context block strings (windows)
        """
        if not search_results:
            return [""]
        
        logger.info(f"Building context windows from {len(search_results)} search results")
        
        # Step 1: Order chunks
        ordered_chunks = self._order_chunks(search_results, custom_order_key)
        
        # Step 2: Clean chunk text
        if self.clean_text:
            ordered_chunks = self._clean_chunks(ordered_chunks)
        
        # Step 3: Reduce redundancy
        if self.reduce_redundancy:
            deduplicated_chunks = self._reduce_redundancy(ordered_chunks)
        else:
            deduplicated_chunks = ordered_chunks
        
        # Step 4: Build windows
        windows = []
        current_window = []
        current_tokens = 0
        
        for entry, score in deduplicated_chunks:
            text = entry.get('text', '').strip()
            if not text:
                continue
            
            text_tokens = len(self.encoder.encode(text))
            
            # Check if adding this chunk would exceed limit
            if current_tokens + text_tokens > self.max_tokens and current_window:
                # Finalize current window
                windows.append("\n\n".join(current_window))
                # Start new window with overlap
                if self.window_overlap > 0:
                    # Keep last few chunks for overlap
                    overlap_tokens = 0
                    overlap_chunks = []
                    for chunk_text in reversed(current_window):
                        chunk_tokens = len(self.encoder.encode(chunk_text))
                        if overlap_tokens + chunk_tokens <= self.window_overlap:
                            overlap_chunks.insert(0, chunk_text)
                            overlap_tokens += chunk_tokens
                        else:
                            break
                    current_window = overlap_chunks
                    current_tokens = overlap_tokens
                else:
                    current_window = []
                    current_tokens = 0
            
            # Add chunk to current window
            current_window.append(text)
            current_tokens += text_tokens
        
        # Add final window
        if current_window:
            windows.append("\n\n".join(current_window))
        
        logger.info(f"Created {len(windows)} context windows")
        return windows


def build_context_from_search(
    search_results: List[Tuple[Dict[str, Any], float]],
    max_tokens: int = 8000,
    order_by: str = 'relevance',
    reduce_redundancy: bool = True,
    similarity_threshold: float = 0.7,
    clean_text: bool = True,
    encoding_name: str = "cl100k_base"
) -> str:
    """
    Convenience function to build context from search results.
    
    Args:
        search_results: List of (entry, similarity_score) tuples from search
        max_tokens: Maximum tokens in context block (default: 4000)
        order_by: Ordering strategy - 'relevance', 'chunk_index', or 'custom'
        reduce_redundancy: Whether to reduce redundant chunks
        similarity_threshold: Threshold for similarity (default: 0.7)
        clean_text: Whether to clean chunk text (default: True)
        encoding_name: Tiktoken encoding name
    
    Returns:
        Single context block string
    """
    builder = ContextBuilder(
        max_tokens=max_tokens,
        encoding_name=encoding_name,
        order_by=order_by,
        reduce_redundancy=reduce_redundancy,
        similarity_threshold=similarity_threshold,
        clean_text=clean_text
    )
    return builder.build_context(search_results)


if __name__ == "__main__":
    # Example usage
    from src.search.searcher import EmbeddingSearcher
    from sentence_transformers import SentenceTransformer
    from pathlib import Path
    from src.knowledge_layer import read_knowledge_entries
    
    # Load embeddings and search
    entries = read_knowledge_entries(Path("embedded.jsonl"))
    searcher = EmbeddingSearcher(entries=entries)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Search
    results = searcher.search_by_text("What is Frankenstein about?", model, top_k=10)
    
    # Build context
    builder = ContextBuilder(max_tokens=2000, order_by='relevance', reduce_redundancy=True)
    context = builder.build_context(results)
    
    print(f"\nContext Block ({len(builder.encoder.encode(context))} tokens):")
    print("=" * 80)
    print(context[:500] + "..." if len(context) > 500 else context)
    print("=" * 80)

