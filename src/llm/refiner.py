"""
Agentic chunk refinement module for post-processing chunks.

This module uses LLM-powered semantic analysis to refine chunks created by
deterministic chunking methods. It can:
- Merge semantically similar chunks
- Generate titles and summaries for chunks
- Improve semantic coherence of chunk boundaries

Use this as a post-processing step after fast deterministic chunking.
"""

import logging
import os
import uuid
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.config import Config

try:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI
    from langchain_anthropic import ChatAnthropic
    try:
        from langchain_ollama import ChatOllama
    except ImportError:
        from langchain_community.chat_models import ChatOllama
    # Try correct import path for create_extraction_chain_pydantic
    try:
        from langchain.chains.openai_functions.extraction import create_extraction_chain_pydantic  # type: ignore
    except ImportError:
        try:
            from langchain.chains import create_extraction_chain_pydantic  # type: ignore
        except ImportError:
            # If both fail, set to None and handle gracefully
            create_extraction_chain_pydantic = None
    from pydantic import BaseModel
    from langchain_core.language_models import BaseChatModel
    from dotenv import load_dotenv
    AGENTIC_AVAILABLE = True
except ImportError:
    AGENTIC_AVAILABLE = False
    create_extraction_chain_pydantic = None

from src.knowledge_layer import create_knowledge_entry

# Load environment variables
try:
    load_dotenv()
except:
    pass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class ChunkID(BaseModel):
    """Pydantic model for extracting chunk IDs."""
    chunk_id: Optional[str]


class AgenticChunkRefiner:
    """
    Refines chunks using LLM-powered semantic analysis.
    
    Takes existing chunks from deterministic chunking and:
    1. Identifies semantically similar chunks
    2. Merges them intelligently
    3. Generates titles and summaries
    """
    
    def __init__(
        self,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: Optional[float] = None,
        enable_merging: Optional[bool] = None,
        enable_metadata: Optional[bool] = None,
        print_logging: bool = True,
        max_chunk_tokens: Optional[int] = None,
        max_merges_per_iteration: Optional[int] = None,
        config: Optional['Config'] = None
    ):
        """
        Initialize the refiner.
        
        Args:
            provider: LLM provider - 'openai', 'anthropic', 'ollama', or 'custom' (overrides config)
            api_key: API key for the provider (defaults to env vars)
            model: Model name (overrides config)
            base_url: Custom base URL for API (overrides config)
            temperature: Temperature for LLM (overrides config)
            enable_merging: Whether to merge similar chunks (overrides config)
            enable_metadata: Whether to generate titles/summaries (overrides config)
            print_logging: Whether to print detailed logs
            max_chunk_tokens: Maximum tokens per merged chunk (overrides config)
            max_merges_per_iteration: Maximum merges per iteration (overrides config)
            config: Optional Config instance
        """
        if not AGENTIC_AVAILABLE:
            raise ImportError(
                "Agentic refiner requires langchain packages. "
                "Install with: pip install langchain langchain-openai langchain-community python-dotenv"
            )
        
        # Get refiner config with priority: function arg > config > default
        refiner_config = {}
        if config:
            refiner_config = config.llm.get('refiner', {})
        
        provider = provider if provider else refiner_config.get('provider', 'ollama')
        model = model if model else refiner_config.get('model', 'llama3')
        temperature = temperature if temperature is not None else refiner_config.get('temperature', 0.3)
        enable_merging = enable_merging if enable_merging is not None else refiner_config.get('enable_merging', True)
        enable_metadata = enable_metadata if enable_metadata is not None else refiner_config.get('enable_metadata', True)
        max_chunk_tokens = max_chunk_tokens if max_chunk_tokens is not None else refiner_config.get('max_chunk_tokens', 600)
        max_merges_per_iteration = max_merges_per_iteration if max_merges_per_iteration is not None else refiner_config.get('max_merges_per_iteration', 5)
        base_url = base_url if base_url else refiner_config.get('base_url', None)
        
        self.enable_merging = enable_merging
        self.enable_metadata = enable_metadata
        self.print_logging = print_logging
        self.id_truncate_limit = 5
        self.max_chunk_tokens = max_chunk_tokens
        self.max_merges_per_iteration = max_merges_per_iteration
        
        # Initialize LLM based on provider
        self.llm = self._initialize_llm(provider, api_key, model, base_url, temperature)
        
        logger.info(f"Initialized AgenticChunkRefiner with {provider} (merging={enable_merging}, metadata={enable_metadata})")
    
    def _initialize_llm(
        self,
        provider: str,
        api_key: Optional[str],
        model: Optional[str],
        base_url: Optional[str],
        temperature: float
    ) -> BaseChatModel:
        """Initialize LLM based on provider."""
        
        if provider == 'ollama':
            # Local Ollama models (free, no API key needed)
            if model is None:
                model = 'llama3'  # Default Ollama model
            logger.info(f"Using Ollama with model: {model}")
            return ChatOllama(
                model=model,
                temperature=temperature,
                base_url=base_url or 'http://localhost:11434'
            )
        
        elif provider == 'openai':
            # OpenAI API
            if api_key is None:
                api_key = os.getenv("OPENAI_API_KEY")
            if api_key is None:
                raise ValueError(
                    "OpenAI API key required. Set OPENAI_API_KEY env var or pass api_key parameter."
                )
            if model is None:
                model = 'gpt-3.5-turbo'
            
            llm = ChatOpenAI(
                model=model,
                openai_api_key=api_key,
                temperature=temperature
            )
            
            # Support custom base URLs (for proxies, local OpenAI-compatible APIs)
            if base_url:
                llm.openai_api_base = base_url
            
            return llm
        
        elif provider == 'anthropic':
            # Anthropic Claude API
            if api_key is None:
                api_key = os.getenv("ANTHROPIC_API_KEY")
            if api_key is None:
                raise ValueError(
                    "Anthropic API key required. Set ANTHROPIC_API_KEY env var or pass api_key parameter."
                )
            if model is None:
                model = 'claude-3-sonnet-20240229'
            
            return ChatAnthropic(
                model=model,
                anthropic_api_key=api_key,
                temperature=temperature
            )
        
        elif provider == 'custom':
            # Custom OpenAI-compatible API (e.g., local models, proxies)
            if api_key is None:
                api_key = os.getenv("OPENAI_API_KEY", "dummy-key")  # Some APIs don't need real key
            if model is None:
                model = 'gpt-3.5-turbo'
            if base_url is None:
                raise ValueError("base_url required for custom provider")
            
            logger.info(f"Using custom API at {base_url} with model: {model}")
            return ChatOpenAI(
                model=model,
                openai_api_key=api_key,
                openai_api_base=base_url,
                temperature=temperature
            )
        
        else:
            raise ValueError(
                f"Unknown provider: {provider}. "
                "Use 'openai', 'anthropic', 'ollama', or 'custom'"
            )
    
    def refine_chunks(
        self,
        chunks: List[Dict],
        max_merge_iterations: int = 3
    ) -> List[Dict]:
        """
        Refine chunks by merging similar ones and adding metadata.
        
        Args:
            chunks: List of chunk dictionaries with 'text' field
            max_merge_iterations: Maximum number of merge passes
        
        Returns:
            Refined list of chunks with potential merges and metadata
        """
        if not chunks:
            return chunks
        
        logger.info(f"Refining {len(chunks)} chunks...")
        
        # Convert to internal format
        internal_chunks = self._chunks_to_internal_format(chunks)
        
        # Merge similar chunks if enabled
        if self.enable_merging:
            internal_chunks = self._merge_similar_chunks(
                internal_chunks,
                max_iterations=max_merge_iterations
            )
        
        # Generate metadata if enabled
        if self.enable_metadata:
            internal_chunks = self._add_metadata(internal_chunks)
        
        # Convert back to Knowledge Layer format
        refined_chunks = self._internal_to_chunks_format(internal_chunks, chunks)
        
        logger.info(f"Refinement complete: {len(chunks)} -> {len(refined_chunks)} chunks")
        
        return refined_chunks
    
    def _chunks_to_internal_format(self, chunks: List[Dict]) -> Dict[str, Dict]:
        """Convert Knowledge Layer chunks to internal format."""
        internal = {}
        for i, chunk in enumerate(chunks):
            chunk_id = str(uuid.uuid4())[:self.id_truncate_limit]
            internal[chunk_id] = {
                'chunk_id': chunk_id,
                'original_index': i,
                'source': chunk.get('source', 'unknown'),  # Preserve source to prevent cross-source merging
                'propositions': [chunk.get('text', '')],
                'title': '',
                'summary': '',
                'entity_type': chunk.get('entity_type', ''),  # Preserve entity_type for person detection
                'entity_name_normalized': chunk.get('entity_name_normalized', '')  # Preserve person name if available
            }
        return internal
    
    def _internal_to_chunks_format(
        self,
        internal_chunks: Dict[str, Dict],
        original_chunks: List[Dict]
    ) -> List[Dict]:
        """Convert internal format back to Knowledge Layer format."""
        refined = []
        for chunk_id, chunk_data in internal_chunks.items():
            # Get original chunk data
            orig_idx = chunk_data.get('original_index', 0)
            original_chunk = original_chunks[orig_idx] if orig_idx < len(original_chunks) else {}
            
            # Merge text from all propositions
            merged_text = ' '.join(chunk_data['propositions'])
            
            # Create new entry - use preserved source from chunk_data (important for merged chunks)
            entry = create_knowledge_entry(
                source=chunk_data.get('source', original_chunk.get('source', 'unknown')),
                chunk_index=len(refined),
                text=merged_text
            )
            
            # Add metadata if available
            if chunk_data.get('title'):
                entry['title'] = chunk_data['title']
            if chunk_data.get('summary'):
                entry['summary'] = chunk_data['summary']
            
            refined.append(entry)
        
        return refined
    
    def _merge_similar_chunks(
        self,
        chunks: Dict[str, Dict],
        max_iterations: int = 3
    ) -> Dict[str, Dict]:
        """Merge semantically similar chunks with constraints."""
        if len(chunks) <= 1:
            return chunks
        
        logger.info(f"Starting merge process (max {max_iterations} iterations, max {self.max_merges_per_iteration} merges/iteration, max {self.max_chunk_tokens} tokens/chunk)...")
        
        import tiktoken
        encoder = tiktoken.get_encoding("cl100k_base")
        
        def count_tokens(text: str) -> int:
            return len(encoder.encode(text))
        
        cross_source_blocks = 0  # Track how many cross-source merges were prevented
        
        for iteration in range(max_iterations):
            chunk_outline = self._get_chunk_outline(chunks)
            valid_chunk_ids = set(chunks.keys())  # Track valid chunk IDs for this iteration
            
            # Try to find chunks to merge
            merges_found = 0
            merges_this_iteration = 0
            
            # Process each chunk to see if it should merge with another
            chunks_to_remove = set()
            for chunk_id, chunk in list(chunks.items()):
                if chunk_id in chunks_to_remove:
                    continue
                
                # Limit merges per iteration
                if merges_this_iteration >= self.max_merges_per_iteration:
                    break
                
                # Get the text of this chunk
                chunk_text = ' '.join(chunk['propositions'])
                chunk_tokens = count_tokens(chunk_text)
                
                # Skip if chunk is already too large
                if chunk_tokens >= self.max_chunk_tokens:
                    continue
                
                # Find if this chunk should merge with another
                target_chunk_id = self._find_merge_target(chunk_text, chunk_outline, chunk_id, valid_chunk_ids)
                
                if target_chunk_id and target_chunk_id != chunk_id:
                    # Validate that target chunk still exists (may have been removed in this iteration)
                    if target_chunk_id not in chunks or target_chunk_id in chunks_to_remove:
                        if self.print_logging:
                            logger.debug(f"Target chunk {target_chunk_id} not found or already marked for removal, skipping merge")
                        continue
                    
                    # Prevent merging chunks from different sources
                    chunk_source = chunk.get('source', 'unknown')
                    target_chunk = chunks[target_chunk_id]
                    target_source = target_chunk.get('source', 'unknown')
                    
                    if chunk_source != target_source:
                        cross_source_blocks += 1
                        if self.print_logging:
                            logger.info(f"⚠ Skipping merge: chunks from different sources ('{chunk_source}' vs '{target_source}') - preserving source separation")
                        continue
                    
                    # Prevent merging person chunks (entity_type == 'személy')
                    chunk_entity_type = chunk.get('entity_type', '')
                    target_entity_type = target_chunk.get('entity_type', '')
                    if chunk_entity_type == 'személy' or target_entity_type == 'személy':
                        if self.print_logging:
                            logger.debug(f"Skipping merge: person chunks should not be merged (entity_type: személy)")
                        continue
                    
                    # Check if merged chunk would exceed size limit
                    target_text = ' '.join(target_chunk['propositions'])
                    target_tokens = count_tokens(target_text)
                    
                    # Only merge if combined size is within limit
                    combined_tokens = chunk_tokens + target_tokens
                    if combined_tokens <= self.max_chunk_tokens:
                        # Merge chunks
                        target_chunk['propositions'].extend(chunk['propositions'])
                        chunks_to_remove.add(chunk_id)
                        merges_found += 1
                        merges_this_iteration += 1
                        
                        if self.print_logging:
                            logger.info(f"Merged chunk {chunk_id} ({chunk_tokens} tokens) into {target_chunk_id} ({target_tokens} tokens) -> {combined_tokens} tokens")
                    else:
                        if self.print_logging:
                            logger.debug(f"Skipped merge: {chunk_id} + {target_chunk_id} would exceed {self.max_chunk_tokens} tokens ({combined_tokens} tokens)")
            
            # Remove merged chunks
            for chunk_id in chunks_to_remove:
                del chunks[chunk_id]
            
            if merges_found == 0:
                logger.info(f"No more merges found after iteration {iteration + 1}")
                break
            
            logger.info(f"Iteration {iteration + 1}: Merged {merges_found} chunks, {len(chunks)} remaining")
        
        if cross_source_blocks > 0:
            logger.info(f"✓ Preserved source separation: prevented {cross_source_blocks} cross-source merge(s) to maintain file boundaries")
        
        return chunks
    
    def _find_merge_target(
        self,
        chunk_text: str,
        chunk_outline: str,
        current_chunk_id: str,
        valid_chunk_ids: Optional[set] = None
    ) -> Optional[str]:
        """Find if a chunk should merge with another existing chunk."""
        PROMPT = ChatPromptTemplate.from_messages([
            (
                "system",
                """
                Determine whether the given chunk should be merged with any existing chunk.
                
                IMPORTANT: Be conservative! Only merge if chunks are:
                - Directly related (same specific topic, not just similar themes)
                - Sequential or adjacent in the original text
                - Would create a coherent, focused unit when combined
                
                DO NOT merge if:
                - Chunks discuss different topics, even if related
                - Chunks are from different sections of the document
                - Merging would create an overly long or unfocused chunk
                
                A chunk should be merged ONLY if it directly continues or completes another chunk's topic.
                
                If the chunk should be merged, return the target chunk ID.
                If it should remain separate, return "No merge".
                
                Only return the chunk ID or "No merge", nothing else.
                """
            ),
            ("user", "Existing chunks:\n{chunk_outline}"),
            ("user", "Chunk to evaluate:\n{chunk_text}\n\nShould this merge with an existing chunk? If yes, which chunk ID?")
        ])
        
        runnable = PROMPT | self.llm
        
        try:
            response = runnable.invoke({
                "chunk_text": chunk_text[:500],  # Limit length
                "chunk_outline": chunk_outline[:2000]  # Limit length
            }).content.strip()
            
            # Extract chunk ID using Pydantic
            if create_extraction_chain_pydantic is None:
                # Fallback: try to extract manually from response
                import re
                chunk_id_match = re.search(r'[a-f0-9]{5}', response)
                if chunk_id_match:
                    extracted_id = chunk_id_match.group()
                    # Validate the extracted ID exists and is not current chunk
                    if extracted_id != current_chunk_id:
                        if valid_chunk_ids is None or extracted_id in valid_chunk_ids:
                            return extracted_id
                return None
            
            extraction_chain = create_extraction_chain_pydantic(
                pydantic_schema=ChunkID,
                llm=self.llm
            )
            extracted = extraction_chain.invoke(response)
            
            if extracted.get("text"):
                chunk_id = extracted["text"][0].chunk_id
                if chunk_id and len(chunk_id) == self.id_truncate_limit and chunk_id != current_chunk_id:
                    # Validate the extracted ID exists in valid chunk IDs
                    if valid_chunk_ids is None or chunk_id in valid_chunk_ids:
                        return chunk_id
            
            return None
        except Exception as e:
            if self.print_logging:
                logger.warning(f"Error finding merge target: {e}")
            return None
    
    def _get_chunk_outline(self, chunks: Dict[str, Dict]) -> str:
        """Generate outline of chunks for LLM context."""
        outline = ""
        for chunk_id, chunk in chunks.items():
            text_preview = ' '.join(chunk['propositions'])[:200]
            outline += f"Chunk ID: {chunk_id}\nText: {text_preview}...\n\n"
        return outline
    
    def _add_metadata(self, chunks: Dict[str, Dict]) -> Dict[str, Dict]:
        """Add titles and summaries to chunks."""
        logger.info("Generating metadata for chunks...")
        
        for chunk_id, chunk in chunks.items():
            chunk_text = ' '.join(chunk['propositions'])
            
            # Check if this is a person chunk
            entity_type = chunk.get('entity_type', '')
            entity_name = chunk.get('entity_name_normalized', '')
            is_person_chunk = entity_type == 'személy'
            
            # Generate summary (with special handling for person chunks)
            if not chunk.get('summary'):
                if is_person_chunk:
                    chunk['summary'] = self._generate_person_summary(chunk_text, entity_name)
                else:
                    chunk['summary'] = self._generate_summary(chunk_text)
            
            # Generate title
            if not chunk.get('title'):
                chunk['title'] = self._generate_title(chunk['summary'])
        
        return chunks
    
    def _generate_summary(self, text: str) -> str:
        """Generate a brief summary for a chunk."""
        PROMPT = ChatPromptTemplate.from_messages([
            (
                "system",
                """
                Generate a very brief 1-sentence summary of what this chunk is about.
                The summary should help identify the topic and theme.
                Only respond with the summary, nothing else.
                """
            ),
            ("user", "Chunk text:\n{text}")
        ])
        
        runnable = PROMPT | self.llm
        
        try:
            summary = runnable.invoke({
                "text": text[:1000]  # Limit length
            }).content.strip()
            return summary
        except Exception as e:
            logger.warning(f"Error generating summary: {e}")
            return ""
    
    def _generate_person_summary(self, text: str, person_name: str = "") -> str:
        """
        Generate a summary for a person chunk.
        
        The summary should indicate that this chunk is a person description.
        
        Args:
            text: Chunk text content
            person_name: Normalized person name (if available)
        
        Returns:
            Summary indicating this is a person description
        """
        PROMPT = ChatPromptTemplate.from_messages([
            (
                "system",
                """
                This chunk contains information about a Hungarian person.
                Generate a very brief 1-sentence summary that clearly indicates this is a person description.
                
                The summary should:
                - State that this is a person description
                - Include the person's name if mentioned
                - Briefly mention key information about the person (role, alias, etc.)
                
                Format: "Ez egy személyleírás: [person name] - [brief description]"
                
                Only respond with the summary, nothing else.
                """
            ),
            ("user", "Person chunk text:\n{text}\n\nPerson name (if known): {person_name}")
        ])
        
        runnable = PROMPT | self.llm
        
        try:
            summary = runnable.invoke({
                "text": text[:1000],  # Limit length
                "person_name": person_name or "ismeretlen"
            }).content.strip()
            
            # Ensure summary indicates it's a person description
            if "személy" not in summary.lower() and "person" not in summary.lower():
                # Fallback: prepend person description indicator
                if person_name:
                    summary = f"Ez egy személyleírás: {person_name} - {summary}"
                else:
                    summary = f"Ez egy személyleírás: {summary}"
            
            return summary
        except Exception as e:
            logger.warning(f"Error generating person summary: {e}")
            # Fallback summary
            if person_name:
                return f"Ez egy személyleírás: {person_name}"
            return "Ez egy személyleírás"
    
    def _generate_title(self, summary: str) -> str:
        """Generate a brief title from a summary."""
        PROMPT = ChatPromptTemplate.from_messages([
            (
                "system",
                """
                Generate a very brief title (2-5 words) based on this summary.
                Only respond with the title, nothing else.
                """
            ),
            ("user", "Summary:\n{summary}")
        ])
        
        runnable = PROMPT | self.llm
        
        try:
            title = runnable.invoke({"summary": summary}).content.strip()
            return title
        except Exception as e:
            logger.warning(f"Error generating title: {e}")
            return ""


def refine_chunks_with_agentic(
    chunks: List[Dict],
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    enable_merging: Optional[bool] = None,
    enable_metadata: Optional[bool] = None,
    max_chunk_tokens: Optional[int] = None,
    max_merges_per_iteration: Optional[int] = None,
    config: Optional['Config'] = None,
    **kwargs
) -> List[Dict]:
    """
    Convenience function to refine chunks.
    
    Args:
        chunks: List of chunk dictionaries
        provider: LLM provider - 'openai', 'anthropic', 'ollama', or 'custom'
        api_key: API key for the provider
        model: Model name (defaults based on provider)
        base_url: Custom base URL for API
        enable_merging: Whether to merge similar chunks
        enable_metadata: Whether to generate metadata
        max_chunk_tokens: Maximum tokens per merged chunk (default: 800)
        max_merges_per_iteration: Maximum merges per iteration (default: 5)
        **kwargs: Additional arguments for AgenticChunkRefiner
    
    Returns:
        Refined chunks
    """
    refiner = AgenticChunkRefiner(
        provider=provider,
        api_key=api_key,
        model=model,
        base_url=base_url,
        enable_merging=enable_merging,
        enable_metadata=enable_metadata,
        max_chunk_tokens=max_chunk_tokens,
        max_merges_per_iteration=max_merges_per_iteration,
        config=config,
        **kwargs
    )
    return refiner.refine_chunks(chunks)
