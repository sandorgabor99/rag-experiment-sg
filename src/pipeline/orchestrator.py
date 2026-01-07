"""
Unified application for the Knowledge Layer pipeline.

This app orchestrates the complete pipeline:
1. Chunk text -> chunks.jsonl
2. Refine chunks (optional) -> chunks_refined.jsonl
3. Extract entities -> chunks_with_entities.jsonl
4. Embed chunks -> embedded.jsonl
5. Search/Query
6. Generate Answer (optional) -> Uses LLaMA to answer questions

Usage:
    # Run full pipeline
    python app.py --run-all

    # Run specific steps
    python app.py --chunk --refine --extract-entities --embed

    # Search
    python app.py --search --query "your question"

    # Search + Answer
    python app.py --search --query "your question" --build-context --answer
"""

import argparse
import logging
import pickle
import shutil
import sys
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Import pipeline modules
from src.processing.chunking import (
    chunk_text_simple,
    chunk_text_sentence_aware,
    chunk_text_hungarian_aware
)
from src.knowledge_layer import (
    read_knowledge_entries,
    write_knowledge_entries,
    create_knowledge_entry
)
from src.processing.cleaning import clean_text 

import tiktoken

# Optional imports
try:
    from src.llm.refiner import refine_chunks_with_agentic, AgenticChunkRefiner
    AGENTIC_AVAILABLE = True
except ImportError:
    AGENTIC_AVAILABLE = False
    logger.warning("Agentic refiner not available. Install langchain packages for refinement.")

try:
    from src.entities.extractor import EntityExtractor
    ENTITY_EXTRACTION_AVAILABLE = True
except ImportError:
    ENTITY_EXTRACTION_AVAILABLE = False
    logger.warning("Entity extraction not available. Install spacy for entity extraction.")

try:
    from src.embeddings.generator import EmbeddingGenerator, write_embeddings, read_embeddings, HUNGARIAN_MODELS
    from src.search.searcher import EmbeddingSearcher
    from src.search.vector_store import VectorStore, load_vector_store_from_embeddings, create_metadata_filter
    from sentence_transformers import SentenceTransformer
    EMBEDDING_AVAILABLE = True
except ImportError:
    EMBEDDING_AVAILABLE = False
    logger.warning("Embedding/search not available. Install sentence-transformers for embeddings.")

try:
    from src.search.context_builder import ContextBuilder, build_context_from_search
    CONTEXT_BUILDER_AVAILABLE = True
except ImportError:
    CONTEXT_BUILDER_AVAILABLE = False
    logger.warning("Context builder not available. Install tiktoken for context building.")

try:
    from src.config import Config, load_config
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    Config = None
    load_config = None
    logger.warning("Config module not available. Configuration features disabled.")

# Metrics imports
try:
    from src.metrics import MetricsLogger, PipelineMetrics, RetrievalMetrics, ContextMetrics, LatencyMetrics
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False
    MetricsLogger = None
    PipelineMetrics = None
    RetrievalMetrics = None
    ContextMetrics = None
    LatencyMetrics = None
    logger.warning("Metrics module not available. Metrics logging disabled.")


class KnowledgeLayerPipeline:
    """Orchestrates the complete Knowledge Layer pipeline."""
    
    def __init__(
        self,
        input_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        chunk_size: Optional[int] = None,
        overlap: Optional[int] = None,
        strategy: Optional[str] = None,
        embedding_model: Optional[str] = None,
        vector_store_path: Optional[Path] = None,
        config: Optional[Config] = None
    ):
        """
        Initialize the pipeline.
        
        Args:
            input_dir: Directory containing cleaned text files (overrides config)
            output_dir: Directory for output files (overrides config)
            chunk_size: Maximum tokens per chunk (overrides config)
            overlap: Token overlap between chunks (overrides config)
            strategy: Chunking strategy (overrides config)
            embedding_model: Model name for embeddings (overrides config)
            vector_store_path: Path to vector store (overrides config)
            config: Optional Config instance (if None, loads from config.yaml)
        """
        # Load config if not provided
        if config is None and CONFIG_AVAILABLE:
            try:
                config = load_config()
            except Exception as e:
                logger.warning(f"Failed to load config: {e}. Using defaults.")
                config = None
        
        self.config = config
        
        # Get values with priority: function arg > config > default
        chunking_config = config.chunking if config else {}
        paths_config = config.paths if config else {}
        embedding_config = config.embedding if config else {}
        
        self.input_dir = Path(input_dir) if input_dir else Path(
            chunking_config.get('input_dir', paths_config.get('input_dir', 'data/raw'))
        )
        self.output_dir = Path(output_dir) if output_dir else Path(
            chunking_config.get('output_dir', paths_config.get('output_dir', 'data/processed'))
        )
        self.chunk_size = chunk_size if chunk_size is not None else chunking_config.get('chunk_size', 400)
        self.overlap = overlap if overlap is not None else chunking_config.get('overlap', 60)
        self.strategy = strategy if strategy else chunking_config.get('strategy', 'hungarian-aware')
        self.embedding_model = embedding_model if embedding_model else embedding_config.get('model', 'baseline')
        
        # File paths (use config paths if available, otherwise construct from output_dir)
        paths_config = config.paths if config else {}
        if paths_config.get('chunks_file'):
            self.chunks_file = Path(paths_config['chunks_file'])
        else:
            self.chunks_file = self.output_dir / "chunks.jsonl"
        
        if paths_config.get('refined_file'):
            self.refined_file = Path(paths_config['refined_file'])
        else:
            self.refined_file = self.output_dir / "chunks_refined.jsonl"
        
        if paths_config.get('entities_file'):
            self.entities_file = Path(paths_config['entities_file'])
        else:
            self.entities_file = self.output_dir / "chunks_with_entities.jsonl"
        
        if paths_config.get('embedded_file'):
            self.embedded_file = Path(paths_config['embedded_file'])
        else:
            self.embedded_file = self.output_dir / "embedded.jsonl"
        
        if vector_store_path:
            self.vector_store_path = Path(vector_store_path)
        elif paths_config.get('vector_store'):
            self.vector_store_path = Path(paths_config['vector_store'])
        else:
            self.vector_store_path = self.output_dir / "vector_store.pkl"
        
        # Initialize encoder
        self.encoder = tiktoken.get_encoding("cl100k_base")
        
        # Initialize metrics logger
        if METRICS_AVAILABLE:
            metrics_dir = self.output_dir.parent / "metrics"  # data/metrics
            self.metrics_logger = MetricsLogger(log_dir=metrics_dir, enabled=True)
            self._last_metrics = None  # Store last metrics for API access
        else:
            self.metrics_logger = None
            self._last_metrics = None
        
        logger.info(f"Initialized pipeline:")
        logger.info(f"  Input: {self.input_dir}")
        logger.info(f"  Output: {self.output_dir}")
        logger.info(f"  Strategy: {self.strategy}, Chunk size: {self.chunk_size}, Overlap: {self.overlap}")
        logger.info(f"  Embedding model: {self.embedding_model}")
        if self.config:
            logger.info(f"  Config loaded from: {self.config._config_path}")
        if self.metrics_logger:
            logger.info(f"  Metrics logging enabled: {self.metrics_logger.log_dir}")


    def step0_clean(self) -> Path:
        """
        Step 0: Clean text files.
        
        First copies files from data/init/ to data/raw/ if needed,
        then cleans them in place.
        
        Returns:
            Path to cleaned text files
        """
        start_time = time.time()
        logger.info("=" * 60)
        logger.info("STEP 0: Cleaning text files")
        logger.info("=" * 60)

        # Check if we need to copy files from data/init/ to data/raw/
        init_dir = self.input_dir.parent / "init"
        if init_dir.exists() and init_dir.is_dir():
            init_txt_files = list(init_dir.glob("*.txt"))
            init_json_files = list(init_dir.glob("*.json"))
            init_files = init_txt_files + init_json_files
            
            # Copy missing files from init to raw
            if init_files:
                self.input_dir.mkdir(parents=True, exist_ok=True)
                copied = 0
                for init_file in init_files:
                    raw_file = self.input_dir / init_file.name
                    if not raw_file.exists():
                        logger.info(f"Copying {init_file.name} from {init_dir} to {self.input_dir}")
                        shutil.copy2(init_file, raw_file)
                        copied += 1
                if copied > 0:
                    logger.info(f"Copied {copied} file(s) from {init_dir} to {self.input_dir}")

        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {self.input_dir}")
        
        # Find text and JSON files in raw directory
        txt_files = list(self.input_dir.glob("*.txt"))
        json_files = list(self.input_dir.glob("*.json"))
        all_files = txt_files + json_files
        
        if not all_files:
            raise FileNotFoundError(f"No .txt or .json files found in {self.input_dir}")

        # Clean text files in raw directory using clean_text() from cleaning.py
        # JSON files are not cleaned (they are structured data)
        # This applies: line ending normalization, page number removal, formatting fixes
        for file in txt_files:
            logger.info(f"Cleaning: {file.name}")
            text = file.read_text(encoding="utf-8")
            cleaned_text = clean_text(text)  # Uses clean_text() from src.processing.cleaning
            file.write_text(cleaned_text, encoding="utf-8")
            logger.info(f"✓ Cleaned {file.name}")
        
        # Log JSON files (no cleaning needed)
        for file in json_files:
            logger.info(f"Found JSON file: {file.name} (no cleaning needed)")
        
        elapsed_time = time.time() - start_time
        logger.info(f"⏱️  STEP 0 completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
        return self.input_dir
    
    def step1_chunk(self) -> Path:
        """
        Step 1: Chunk text and JSON files.
        
        Returns:
            Path to chunks file
        """
        start_time = time.time()
        logger.info("=" * 60)
        logger.info("STEP 1: Chunking text and JSON files")
        logger.info("=" * 60)
        
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {self.input_dir}")
        
        # Find text and JSON files
        txt_files = list(self.input_dir.glob("*.txt"))
        json_files = list(self.input_dir.glob("*.json"))
        all_files = txt_files + json_files
        
        if not all_files:
            raise FileNotFoundError(f"No .txt or .json files found in {self.input_dir}")
        
        logger.info(f"Found {len(txt_files)} text file(s) and {len(json_files)} JSON file(s)")
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        all_chunks = []
        
        # Process text files
        for file in txt_files:
            logger.info(f"Processing text file: {file.name}")
            text = file.read_text(encoding="utf-8")
            
            # Check if file fits in a single chunk
            text_tokens = len(self.encoder.encode(text))
            if text_tokens <= self.chunk_size:
                # File is small enough - keep as single chunk
                logger.info(f"  File fits in single chunk ({text_tokens} tokens <= {self.chunk_size}), keeping intact")
                # For small files, use Hungarian-aware to get proper metadata
                if self.strategy == 'hungarian-aware':
                    chunks_with_metadata = chunk_text_hungarian_aware(
                        text, self.chunk_size, self.overlap, self.encoder, source_file=file.name
                    )
                else:
                    chunks_with_metadata = [(text, {'language': 'hu', 'entity_type': 'szabályzat'})]
            else:
                # File needs chunking
                logger.info(f"  File needs chunking ({text_tokens} tokens > {self.chunk_size})")
                # Chunk based on strategy
                if self.strategy == 'simple':
                    chunks = chunk_text_simple(text, self.chunk_size, self.overlap, self.encoder)
                    # Convert to (text, metadata) format for consistency
                    chunks_with_metadata = [(chunk, {}) for chunk in chunks]
                elif self.strategy == 'sentence-aware':
                    chunks = chunk_text_sentence_aware(text, self.chunk_size, self.overlap, self.encoder)
                    # Convert to (text, metadata) format for consistency
                    chunks_with_metadata = [(chunk, {}) for chunk in chunks]
                elif self.strategy == 'hungarian-aware':
                    # Hungarian-aware chunking returns (text, metadata) tuples
                    chunks_with_metadata = chunk_text_hungarian_aware(
                        text, self.chunk_size, self.overlap, self.encoder, source_file=file.name
                    )
                else:
                    raise ValueError(f"Unknown strategy: {self.strategy}")
            
            # Create Knowledge Layer entries
            for i, (chunk_text, chunk_metadata) in enumerate(chunks_with_metadata):
                entry = create_knowledge_entry(
                    source=file.name,
                    chunk_index=i,
                    text=chunk_text
                )
                # Add Hungarian metadata to entry
                entry.update(chunk_metadata)
                all_chunks.append(entry)
            
            logger.info(f"✓ Created {len(chunks_with_metadata)} chunk(s) from {file.name}")
        
        # Process JSON files
        import json
        for file in json_files:
            logger.info(f"Processing JSON file: {file.name}")
            try:
                json_text = file.read_text(encoding="utf-8")
                json_data = json.loads(json_text)
                
                # JSON files are always processed with Hungarian-aware strategy
                # (they contain structured person data)
                if self.strategy == 'hungarian-aware':
                    chunks_with_metadata = chunk_text_hungarian_aware(
                        text="",  # Empty text for JSON
                        chunk_size=self.chunk_size,
                        overlap=self.overlap,
                        encoder=self.encoder,
                        source_file=file.name,
                        is_json=True,
                        json_data=json_data
                    )
                else:
                    logger.warning(f"JSON file {file.name} requires 'hungarian-aware' strategy. Using it anyway.")
                    chunks_with_metadata = chunk_text_hungarian_aware(
                        text="",
                        chunk_size=self.chunk_size,
                        overlap=self.overlap,
                        encoder=self.encoder,
                        source_file=file.name,
                        is_json=True,
                        json_data=json_data
                    )
                
                # Create Knowledge Layer entries
                for i, (chunk_text, chunk_metadata) in enumerate(chunks_with_metadata):
                    entry = create_knowledge_entry(
                        source=file.name,
                        chunk_index=i,
                        text=chunk_text
                    )
                    # Add Hungarian metadata to entry
                    entry.update(chunk_metadata)
                    all_chunks.append(entry)
                
                logger.info(f"✓ Created {len(chunks_with_metadata)} chunk(s) from {file.name}")
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON file {file.name}: {e}")
                raise
            except Exception as e:
                logger.error(f"Failed to process JSON file {file.name}: {e}", exc_info=True)
                raise
        
        # Write chunks
        write_knowledge_entries(all_chunks, self.chunks_file)
        elapsed_time = time.time() - start_time
        logger.info(f"✓ Step 1 complete: {len(all_chunks)} chunks written to {self.chunks_file}")
        logger.info(f"⏱️  STEP 1 completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
        
        return self.chunks_file
    
    def step2_refine(
        self,
        input_file: Path,
        provider: str = 'ollama',
        model: str = 'llama3',
        enable_merging: bool = True,
        enable_metadata: bool = True,
        max_chunk_tokens: int = 800,
        max_merges_per_iteration: int = 5
    ) -> Path:
        """
        Step 2: Refine chunks using agentic refiner (optional).
        
        Args:
            input_file: Input chunks file
            provider: LLM provider
            model: Model name
            enable_merging: Whether to merge similar chunks
            enable_metadata: Whether to generate metadata
            max_chunk_tokens: Max tokens per merged chunk
            max_merges_per_iteration: Max merges per iteration
        
        Returns:
            Path to refined chunks file
        """
        start_time = time.time()
        logger.info("=" * 60)
        logger.info("STEP 2: Refining chunks (optional)")
        logger.info("=" * 60)
        
        if not AGENTIC_AVAILABLE:
            logger.warning("Agentic refiner not available. Skipping refinement.")
            logger.warning("Install langchain packages to enable refinement.")
            return input_file
        
        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        # Get refiner config with priority: function arg > config > default
        refiner_config = {}
        if self.config:
            refiner_config = self.config.llm.get('refiner', {})
        
        # Check if refinement is enabled in config
        if not refiner_config.get('enabled', False):
            logger.info("Refinement disabled in config. Skipping.")
            return input_file
        
        # Use config values if function args not provided
        provider = provider if provider else refiner_config.get('provider', 'ollama')
        model = model if model else refiner_config.get('model', 'llama3')
        enable_merging = enable_merging if enable_merging is not None else refiner_config.get('enable_merging', True)
        enable_metadata = enable_metadata if enable_metadata is not None else refiner_config.get('enable_metadata', True)
        max_chunk_tokens = max_chunk_tokens if max_chunk_tokens is not None else refiner_config.get('max_chunk_tokens', 600)
        max_merges_per_iteration = max_merges_per_iteration if max_merges_per_iteration is not None else refiner_config.get('max_merges_per_iteration', 5)
        
        # Load chunks
        chunks = read_knowledge_entries(input_file)
        logger.info(f"Loaded {len(chunks)} chunks for refinement")
        
        try:
            # Refine
            refined_chunks = refine_chunks_with_agentic(
                chunks=chunks,
                provider=provider,
                model=model,
                enable_merging=enable_merging,
                enable_metadata=enable_metadata,
                max_chunk_tokens=max_chunk_tokens,
                max_merges_per_iteration=max_merges_per_iteration,
                config=self.config
            )
            
            # Write refined chunks
            write_knowledge_entries(refined_chunks, self.refined_file)
            elapsed_time = time.time() - start_time
            logger.info(f"✓ Step 2 complete: {len(refined_chunks)} refined chunks written to {self.refined_file}")
            logger.info(f"⏱️  STEP 2 completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
            
            return self.refined_file
        except Exception as e:
            logger.error(f"Refinement failed: {e}", exc_info=True)
            logger.warning("Continuing with original chunks...")
            return input_file
    
    def step3_extract_entities(
        self,
        input_file: Path,
        method: Optional[str] = None,
        model: Optional[str] = None
    ) -> Path:
        """
        Step 3: Extract entities from chunks.
        
        Args:
            input_file: Input chunks file
            method: Extraction method ('spacy', 'llm', 'hybrid')
            model: spaCy model name
        
        Returns:
            Path to chunks with entities file
        """
        start_time = time.time()
        logger.info("=" * 60)
        logger.info("STEP 3: Extracting entities")
        logger.info("=" * 60)
        
        if not ENTITY_EXTRACTION_AVAILABLE:
            logger.warning("Entity extraction not available. Skipping entity extraction.")
            logger.warning("Install spacy and huspacy to enable entity extraction: pip install spacy huspacy && python -m huspacy download")
            return input_file
        
        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        # Get entity extraction config with priority: function arg > config > default
        entity_config = {}
        if self.config:
            entity_config = self.config.llm.get('entity_extraction', {})
        
        method = method if method else entity_config.get('method', 'spacy')
        model = model if model else entity_config.get('model', 'hu_core_news_lg')
        
        # Load chunks
        chunks = read_knowledge_entries(input_file)
        logger.info(f"Loaded {len(chunks)} chunks for entity extraction")
        
        try:
            # Extract entities
            extractor = EntityExtractor(method=method, model=model)
            enriched_chunks = extractor.extract_from_chunks(chunks)
            
            # Write enriched chunks
            write_knowledge_entries(enriched_chunks, self.entities_file)
            elapsed_time = time.time() - start_time
            logger.info(f"✓ Step 3 complete: Entities extracted and written to {self.entities_file}")
            logger.info(f"⏱️  STEP 3 completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
            
            return self.entities_file
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}", exc_info=True)
            logger.warning("Continuing without entities...")
            return input_file
    
    def step4_embed(
        self,
        input_file: Path,
        batch_size: Optional[int] = None
    ) -> Path:
        """
        Step 4: Generate embeddings for chunks.
        
        Args:
            input_file: Input chunks file
            batch_size: Batch size for embedding generation
        
        Returns:
            Path to embedded chunks file
        """
        start_time = time.time()
        logger.info("=" * 60)
        logger.info("STEP 4: Generating embeddings")
        logger.info("=" * 60)
        
        if not EMBEDDING_AVAILABLE:
            raise ImportError("Embedding generation requires sentence-transformers")
        
        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        # Load chunks
        chunks = read_knowledge_entries(input_file)
        logger.info(f"Loaded {len(chunks)} chunks for embedding")
        
        # Generate embeddings with metadata
        generator = EmbeddingGenerator(model=self.embedding_model, config=self.config)
        
        # Prepare chunking config for metadata hash
        chunking_config = None
        try:
            from src.embeddings.metadata import METADATA_AVAILABLE
            if METADATA_AVAILABLE:
                chunking_config = {
                    'chunking_strategy': self.strategy,
                    'chunk_size': self.chunk_size,
                    'overlap': self.overlap,
                    'embedding_model': self.embedding_model
                }
        except ImportError:
            pass
        
        result = generator.embed_knowledge_entries(
            chunks,
            batch_size=batch_size,
            generate_metadata=True,
            chunking_config=chunking_config
        )
        
        # Handle tuple return (embedded_chunks, metadata) or single return for backward compatibility
        if isinstance(result, tuple):
            embedded_chunks, embedding_metadata = result
        else:
            embedded_chunks = result
            embedding_metadata = None
        
        # Write embedded chunks
        write_embeddings(embedded_chunks, self.embedded_file)
        
        # Store metadata for later use in index building
        if embedding_metadata:
            logger.info(f"Generated embedding metadata: model={embedding_metadata.model_name}, "
                       f"dim={embedding_metadata.embedding_dim}, "
                       f"config_hash={embedding_metadata.config_hash[:8] if embedding_metadata.config_hash else 'N/A'}...")
        
        logger.info(f"✓ Step 4 complete: Embeddings written to {self.embedded_file}")
        
        return self.embedded_file
    
    def step5_search(
        self,
        query: str,
        top_k: Optional[int] = None,
        embeddings_file: Optional[Path] = None,
        similarity_metric: Optional[str] = None,
        source_filter: Optional[str] = None,
        build_context: bool = False,
        max_context_tokens: Optional[int] = None,
        context_order: Optional[str] = None,
        reduce_redundancy: Optional[bool] = None,
        diverse_search: Optional[bool] = None
    ) -> Tuple[List[tuple], Optional[str]]:
        """
        Step 5: Search/Query the knowledge base using VectorStore.
        
        Args:
            query: Search query text
            top_k: Number of results to return
            embeddings_file: Path to embeddings file (defaults to self.embedded_file)
            similarity_metric: 'cosine' or 'dot_product'
            source_filter: Optional source file name to filter results
            build_context: Whether to build a context block from results
            max_context_tokens: Maximum tokens in context block
            context_order: Order chunks in context - 'relevance' or 'chunk_index'
            reduce_redundancy: Whether to reduce redundant chunks
        
        Returns:
            Tuple of (search_results, context_block)
            - search_results: List of (entry, similarity_score) tuples
            - context_block: Optional context block string (if build_context=True)
        """
        start_time = time.time()
        pipeline_start_time = time.perf_counter()  # High precision for latency metrics
        logger.info("=" * 60)
        logger.info("STEP 5: Searching knowledge base")
        logger.info("=" * 60)
        
        if not EMBEDDING_AVAILABLE:
            raise ImportError("Search requires sentence-transformers")
        
        # Get search and context config with priority: function arg > config > default
        search_config = {}
        context_config = {}
        if self.config:
            search_config = self.config.search
            context_config = self.config.context
        
        top_k = top_k if top_k is not None else search_config.get('top_k_default', 20)
        similarity_metric = similarity_metric if similarity_metric else search_config.get('similarity_metric', 'cosine')
        diverse_search = diverse_search if diverse_search is not None else search_config.get('diverse_search', True)
        max_context_tokens = max_context_tokens if max_context_tokens is not None else context_config.get('max_tokens_default', 8000)
        context_order = context_order if context_order else context_config.get('order_by', 'relevance')
        reduce_redundancy = reduce_redundancy if reduce_redundancy is not None else context_config.get('reduce_redundancy', True)
        
        embeddings_path = embeddings_file or self.embedded_file
        
        # Load model early to check embedding dimension
        # Resolve model name from HUNGARIAN_MODELS if it's a key
        actual_model_name = HUNGARIAN_MODELS.get(self.embedding_model, self.embedding_model)
        if actual_model_name != self.embedding_model:
            logger.info(f"Resolved model alias '{self.embedding_model}' to '{actual_model_name}'")
        model = SentenceTransformer(actual_model_name)
        model_dim = model.get_sentence_embedding_dimension()
        logger.info(f"Current embedding model dimension: {model_dim}")
        
        # Try loading from VectorStore first (faster)
        # But rebuild if embeddings file is newer OR if dimension mismatch
        should_rebuild = False
        if self.vector_store_path.exists() and embeddings_path and embeddings_path.exists():
            store_mtime = self.vector_store_path.stat().st_mtime
            embeddings_mtime = embeddings_path.stat().st_mtime
            if embeddings_mtime > store_mtime:
                logger.info(f"Embeddings file is newer than VectorStore, rebuilding...")
                should_rebuild = True
            
            # Check for dimension mismatch
            if not should_rebuild:
                try:
                    with open(self.vector_store_path, 'rb') as f:
                        store_data = pickle.load(f)
                        store_dim = store_data.get('embedding_dim')
                        if store_dim != model_dim:
                            logger.warning(
                                f"Dimension mismatch detected: VectorStore has {store_dim} dimensions, "
                                f"but current model has {model_dim} dimensions. Rebuilding VectorStore..."
                            )
                            should_rebuild = True
                except Exception as e:
                    logger.warning(f"Could not check VectorStore dimension: {e}. Rebuilding...")
                    should_rebuild = True
        
        if self.vector_store_path.exists() and not should_rebuild:
            logger.info(f"Loading VectorStore from {self.vector_store_path}")
            searcher = EmbeddingSearcher(store_path=self.vector_store_path)
        elif embeddings_path and embeddings_path.exists():
            logger.info(f"Creating VectorStore from {embeddings_path}")
            # Read embeddings and verify dimension
            entries = read_embeddings(embeddings_path)
            if entries:
                first_embedding = entries[0].get('embedding')
                if first_embedding and len(first_embedding) != model_dim:
                    raise ValueError(
                        f"Embedding dimension mismatch: Embeddings file has {len(first_embedding)} dimensions, "
                        f"but current model has {model_dim} dimensions.\n"
                        f"Please re-embed your chunks with the current model:\n"
                        f"  python main.py --embed"
                    )
            # Create VectorStore from embeddings file
            searcher = EmbeddingSearcher(
                entries=entries,
                store_path=self.vector_store_path
            )
            # Save for future use
            searcher.save_store(self.vector_store_path)
            logger.info(f"Saved VectorStore to {self.vector_store_path}")
        else:
            raise FileNotFoundError(
                f"Neither embeddings file ({embeddings_path}) nor VectorStore "
                f"({self.vector_store_path}) found. Run embedding step first."
            )
        
        # Create metadata filter if needed
        metadata_filter = None
        if source_filter:
            metadata_filter = create_metadata_filter(source=source_filter)
            logger.info(f"Applying source filter: '{source_filter}'")
            
            # Debug: Check available sources in store
            if hasattr(searcher.store, 'metadata') and searcher.store.metadata:
                available_sources = set(m.get('source') for m in searcher.store.metadata if m.get('source'))
                logger.info(f"Available sources in store: {sorted(available_sources)}")
                if source_filter not in available_sources:
                    logger.warning(
                        f"Source filter '{source_filter}' not found in available sources. "
                        f"Available: {sorted(available_sources)}"
                    )
        
        # Search (model already loaded above)
        # Use diverse search to ensure results from multiple sources
        retrieval_metrics = None
        if diverse_search and not source_filter:
            search_result = searcher.search_by_text_diverse(
                query,
                model,
                top_k=top_k,
                similarity_metric=similarity_metric,
                metadata_filter=metadata_filter,
                boost_entity_names=search_config.get('boost_entity_names', True) if self.config else True,
                return_metrics=True
            )
            # Handle tuple return (results, metrics) or single return for backward compatibility
            if isinstance(search_result, tuple):
                results, retrieval_metrics = search_result
            else:
                results = search_result
        else:
            # Use regular search if source filter is specified or diverse_search is disabled
            search_result = searcher.search_by_text(
                query,
                model,
                top_k=top_k,
                similarity_metric=similarity_metric,
                metadata_filter=metadata_filter,
                boost_entity_names=search_config.get('boost_entity_names', True) if self.config else True,
                return_metrics=True
            )
            # Handle tuple return (results, metrics) or single return for backward compatibility
            if isinstance(search_result, tuple):
                results, retrieval_metrics = search_result
            else:
                results = search_result
        
        search_elapsed = time.time() - start_time
        search_elapsed_ms = search_elapsed * 1000
        logger.info(f"✓ Found {len(results)} results for query: '{query}'")
        logger.info(f"⏱️  Search completed in {search_elapsed:.2f} seconds")
        
        # Log retrieval metrics if available
        if retrieval_metrics:
            logger.info(f"Retrieval metrics: avg_similarity={retrieval_metrics.top_k_similarity_avg:.3f}, "
                       f"spread={retrieval_metrics.top_1_vs_top_k_spread:.3f}, "
                       f"hit_density={retrieval_metrics.hit_density:.3f}, "
                       f"entity_boost={retrieval_metrics.entity_boost_applied}")
        
        # Build context block if requested
        context_block = None
        context_metrics = None
        context_build_time_ms = 0.0
        if build_context:
            context_start = time.perf_counter()
            if not CONTEXT_BUILDER_AVAILABLE:
                logger.warning("Context builder not available. Install tiktoken.")
            else:
                try:
                    # Use config for ContextBuilder if available
                    similarity_threshold = context_config.get('similarity_threshold', 0.7) if self.config else 0.7
                    clean_text = context_config.get('clean_text', True) if self.config else True
                    
                    builder = ContextBuilder(
                        max_tokens=max_context_tokens,
                        order_by=context_order,
                        reduce_redundancy=reduce_redundancy,
                        similarity_threshold=similarity_threshold,
                        clean_text=clean_text,
                        config=self.config
                    )
                    
                    # Track chunks before for metrics
                    chunks_before = len(results)
                    context_block = builder.build_context(results)
                    context_tokens = len(builder.encoder.encode(context_block)) if context_block else 0
                    context_build_time_ms = (time.perf_counter() - context_start) * 1000
                    
                    # Calculate context metrics (simplified)
                    if METRICS_AVAILABLE and ContextMetrics:
                        # Get unique sources from results that contributed to context
                        unique_sources = set()
                        for entry, _ in results:
                            unique_sources.add(entry.get('source', 'unknown'))
                        unique_source_count = len(unique_sources)
                        
                        # Estimate chunks included (simplified - assume all chunks before dedup contribute)
                        chunks_before_dedup = chunks_before
                        chunks_after_dedup = chunks_before  # Simplified - ContextBuilder doesn't expose this
                        chunks_included = chunks_before  # Simplified
                        
                        redundancy_ratio = 0.0  # Simplified - would need ContextBuilder to expose this
                        
                        context_metrics = ContextMetrics(
                            context_tokens_used=context_tokens,
                            unique_source_count=unique_source_count,
                            redundancy_ratio=redundancy_ratio,
                            truncation_events=context_tokens >= max_context_tokens if context_block else False,
                            chunks_included=chunks_included,
                            chunks_excluded=0,  # Simplified
                            chunks_before_dedup=chunks_before_dedup,
                            chunks_after_dedup=chunks_after_dedup,
                            context_build_time_ms=context_build_time_ms
                        )
                    
                    logger.info(f"✓ Built context block: {context_tokens} tokens")
                except Exception as e:
                    logger.error(f"Failed to build context: {e}", exc_info=True)
        
        # Log metrics if metrics logger is available
        if METRICS_AVAILABLE and self.metrics_logger and PipelineMetrics:
            import uuid
            from datetime import datetime
            
            # Create latency metrics
            latency_metrics = LatencyMetrics(
                search_time_ms=search_elapsed_ms,
                context_build_time_ms=context_build_time_ms,
                total_time_ms=search_elapsed_ms + context_build_time_ms
            )
            
            # Create pipeline metrics
            pipeline_metrics = PipelineMetrics(
                request_id=str(uuid.uuid4()),
                timestamp=datetime.utcnow().isoformat(),
                query=query,
                retrieval_metrics=retrieval_metrics if retrieval_metrics else None,
                context_metrics=context_metrics,
                latency_metrics=latency_metrics,
                success=True
            )
            
            # Log metrics
            try:
                self.metrics_logger.log(pipeline_metrics)
                self._last_metrics = pipeline_metrics  # Store for API access
                logger.info(f"✓ Metrics logged to {self.metrics_logger.log_dir}")
            except Exception as e:
                logger.warning(f"Failed to log metrics: {e}", exc_info=True)
        
        return results, context_block
    
    def step6_answer(
        self,
        question: str,
        context: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Step 6: Generate answer using LLaMA.
        
        Args:
            question: User's question
            context: Context text from knowledge base
            provider: LLM provider - 'ollama', 'openai', 'anthropic', or 'custom'
            model: Model name (default: llama3)
            temperature: Temperature for LLM (default: 0.7)
            max_tokens: Maximum tokens for answer (default: 500)
        
        Returns:
            Generated answer string
        """
        start_time = time.time()
        logger.info("=" * 60)
        logger.info("STEP 6: Generating answer with LLaMA")
        logger.info("=" * 60)
        
        try:
            from src.llm.qa import QuestionAnswerer, QA_AVAILABLE
            
            if not QA_AVAILABLE:
                raise ImportError(
                    "QA module requires langchain packages. "
                    "Install with: pip install langchain langchain-openai langchain-community langchain-anthropic"
                )
            
            qa = QuestionAnswerer(
                provider=provider,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                config=self.config
            )
            
            answer = qa.answer(question, context)
            elapsed_time = time.time() - start_time
            llm_time_ms = elapsed_time * 1000
            logger.info(f"✓ Generated answer: {len(answer)} characters")
            logger.info(f"⏱️  STEP 6 completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
            
            # Update last metrics with LLM timing if available
            if METRICS_AVAILABLE and self.metrics_logger and self._last_metrics:
                if self._last_metrics.latency_metrics:
                    self._last_metrics.latency_metrics.llm_time_ms = llm_time_ms
                    self._last_metrics.latency_metrics.total_time_ms += llm_time_ms
                self._last_metrics.answer_length = len(answer)
                # Log updated metrics
                try:
                    self.metrics_logger.log(self._last_metrics)
                except Exception as e:
                    logger.warning(f"Failed to log updated metrics: {e}", exc_info=True)
            
            return answer
            
        except ImportError as e:
            logger.error(f"Failed to import QA module: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to generate answer: {e}", exc_info=True)
            raise
    
    def run_pipeline(
        self,
        refine: bool = False,
        extract_entities: bool = True,
        embed: bool = True,
        refine_provider: Optional[str] = None,
        refine_model: Optional[str] = None,
        entity_method: Optional[str] = None
    ) -> Dict[str, Path]:
        """
        Run the complete pipeline.
        
        Args:
            refine: Whether to refine chunks
            extract_entities: Whether to extract entities
            embed: Whether to generate embeddings
            refine_provider: LLM provider for refinement
            refine_model: Model for refinement
            entity_method: Entity extraction method
        
        Returns:
            Dictionary mapping step names to output file paths
        """
        pipeline_start_time = time.time()
        logger.info("=" * 60)
        logger.info("STARTING KNOWLEDGE LAYER PIPELINE")
        logger.info("=" * 60)
        
        results = {}
        
        # Step 1: Chunk
        chunks_file = self.step1_chunk()
        results['chunks'] = chunks_file
        current_file = chunks_file
        
        # Step 2: Refine (optional)
        if refine:
            refined_file = self.step2_refine(
                current_file,
                provider=refine_provider,
                model=refine_model
            )
            results['refined'] = refined_file
            current_file = refined_file
        
        # Step 3: Extract entities
        if extract_entities:
            entities_file = self.step3_extract_entities(
                current_file,
                method=entity_method
            )
            results['entities'] = entities_file
            current_file = entities_file
        
        # Step 4: Embed
        if embed:
            embedded_file = self.step4_embed(current_file)
            results['embedded'] = embedded_file
        
        pipeline_elapsed = time.time() - pipeline_start_time
        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 60)
        logger.info(f"⏱️  TOTAL PIPELINE TIME: {pipeline_elapsed:.2f} seconds ({pipeline_elapsed/60:.2f} minutes)")
        
        return results


def main():
    parser = argparse.ArgumentParser(
        description='Knowledge Layer Pipeline - Complete text processing pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline
  python main.py --run-all

  # Run with custom config
  python main.py --config custom_config.yaml --run-all

  # Show effective configuration
  python main.py --show-config

  # Run specific steps
  python main.py --chunk --refine --extract-entities --embed

  # Search
  python main.py --search --query "What is Frankenstein about?"

  # Custom configuration
  python main.py --run-all --chunk-size 500 --embedding-model all-mpnet-base-v2
        """
    )
    
    # Config arguments (must be first)
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to config.yaml file (default: config.yaml in project root)'
    )
    parser.add_argument(
        '--show-config',
        action='store_true',
        help='Display effective configuration and exit'
    )
    
    # Input/Output
    parser.add_argument(
        '--input-dir',
        type=str,
        default='data/raw',
        help='Input directory with cleaned text files (default: data/raw)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/processed',
        help='Output directory for all files (default: data/processed)'
    )
    
    # Chunking
    parser.add_argument(
        '--chunk',
        action='store_true',
        help='Run chunking step'
    )
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=400,
        help='Maximum tokens per chunk (default: 400)'
    )
    parser.add_argument(
        '--overlap',
        type=int,
        default=60,
        help='Token overlap between chunks (default: 60)'
    )
    parser.add_argument(
        '--strategy',
        type=str,
        default='hungarian-aware',
        choices=['simple', 'sentence-aware', 'hungarian-aware'],
        help='Chunking strategy: simple (token-based), sentence-aware (sentence boundaries), hungarian-aware (Hungarian-optimized with entity preservation) (default: hungarian-aware)'
    )
    
    # Refinement
    parser.add_argument(
        '--refine',
        action='store_true',
        help='Run refinement step (optional)'
    )
    parser.add_argument(
        '--refine-provider',
        type=str,
        default='ollama',
        choices=['ollama', 'openai', 'anthropic', 'custom'],
        help='LLM provider for refinement (default: ollama)'
    )
    parser.add_argument(
        '--refine-model',
        type=str,
        default='llama3',
        help='Model name for refinement (default: llama3)'
    )
    parser.add_argument(
        '--no-merging',
        action='store_true',
        help='Disable chunk merging during refinement'
    )
    parser.add_argument(
        '--no-metadata',
        action='store_true',
        help='Disable metadata generation during refinement'
    )
    
    # Entity extraction
    parser.add_argument(
        '--extract-entities',
        action='store_true',
        help='Run entity extraction step'
    )
    parser.add_argument(
        '--entity-method',
        type=str,
        default='spacy',
        choices=['spacy', 'llm', 'hybrid'],
        help='Entity extraction method (default: spacy)'
    )
    
    # Embedding
    parser.add_argument(
        '--embed',
        action='store_true',
        help='Run embedding generation step'
    )
    parser.add_argument(
        '--embedding-model',
        type=str,
        default='baseline',
        help='Embedding model name (default: baseline, maps to all-MiniLM-L6-v2)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for embedding generation (default: 32)'
    )
    
    # Search
    parser.add_argument(
        '--search',
        action='store_true',
        help='Run search/query step'
    )
    parser.add_argument(
        '--query',
        type=str,
        help='Search query text'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=5,
        help='Number of search results (default: 5)'
    )
    parser.add_argument(
        '--embeddings-file',
        type=str,
        help='Path to embeddings file (default: output_dir/embedded.jsonl)'
    )
    parser.add_argument(
        '--similarity-metric',
        type=str,
        choices=['cosine', 'dot_product'],
        default='cosine',
        help='Similarity metric for search: cosine or dot_product (default: cosine)'
    )
    parser.add_argument(
        '--source-filter',
        type=str,
        help='Filter search results by source file name'
    )
    parser.add_argument(
        '--build-context',
        action='store_true',
        help='Build a single context block from search results (for LLM input)'
    )
    parser.add_argument(
        '--max-context-tokens',
        type=int,
        default=8000,
        help='Maximum tokens in context block (default: 8000)'
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
    
    # Question-Answering
    parser.add_argument(
        '--answer',
        action='store_true',
        help='Generate answer using LLM (requires --search and --build-context)'
    )
    parser.add_argument(
        '--llm-provider',
        type=str,
        default='ollama',
        choices=['ollama', 'openai', 'anthropic', 'custom'],
        help='LLM provider for QA (default: ollama)'
    )
    parser.add_argument(
        '--llm-model',
        type=str,
        default='llama3',
        help='LLM model name for QA (default: llama3)'
    )
    parser.add_argument(
        '--llm-temperature',
        type=float,
        default=0.7,
        help='Temperature for LLM (default: 0.7)'
    )
    parser.add_argument(
        '--max-answer-tokens',
        type=int,
        default=500,
        help='Maximum tokens for answer (default: 500)'
    )
    
    # Text cleaning
    parser.add_argument(
        '--clean',
        action='store_true',
        help='Clean text files (copy from data/init/ to data/raw/ and apply cleaning)'
    )
    
    # Convenience flags
    parser.add_argument(
        '--run-all',
        action='store_true',
        help='Run complete pipeline (clean -> chunk -> refine -> extract-entities -> embed)'
    )
    
    args = parser.parse_args()
    
    # Load config (with custom path if provided)
    config = None
    if CONFIG_AVAILABLE:
        try:
            config_path = Path(args.config) if args.config else None
            config = load_config(config_path)
        except Exception as e:
            logger.warning(f"Failed to load config: {e}. Continuing with defaults.")
    
    # Show config and exit if requested
    if args.show_config:
        if config:
            config.show()
        else:
            logger.warning("Config not available. Install PyYAML to use configuration.")
        sys.exit(0)
    
    # Initialize pipeline with config (CLI args will override config values)
    # Priority: CLI arg > config > default
    chunking_config = config.chunking if config else {}
    paths_config = config.paths if config else {}
    embedding_config = config.embedding if config else {}
    
    pipeline = KnowledgeLayerPipeline(
        input_dir=Path(args.input_dir) if args.input_dir != 'data/raw' else None,
        output_dir=Path(args.output_dir) if args.output_dir != 'data/processed' else None,
        chunk_size=args.chunk_size if args.chunk_size != 400 else None,
        overlap=args.overlap if args.overlap != 60 else None,
        strategy=args.strategy if args.strategy != 'hungarian-aware' else None,
        embedding_model=args.embedding_model if args.embedding_model != 'baseline' else None,
        config=config
    )
    
    # Step 0: Clean (if requested or part of run-all)
    if args.clean or args.run_all:
        pipeline.step0_clean()
    
    # Determine which steps to run
    if args.run_all:
        # Run complete pipeline
        pipeline_start_time = time.time()
        logger.info("=" * 60)
        logger.info("RUNNING COMPLETE PIPELINE (--run-all)")
        logger.info("=" * 60)
        
        # Use CLI args if different from defaults, otherwise config will be used
        refine_provider = args.refine_provider if args.refine_provider != 'ollama' else None
        refine_model = args.refine_model if args.refine_model != 'llama3' else None
        entity_method = args.entity_method if args.entity_method != 'spacy' else None
        
        results = pipeline.run_pipeline(
            refine=True,
            extract_entities=True,
            embed=True,
            refine_provider=refine_provider,
            refine_model=refine_model,
            entity_method=entity_method
        )
        
        pipeline_elapsed = time.time() - pipeline_start_time
        logger.info("\nPipeline outputs:")
        for step, path in results.items():
            logger.info(f"  {step}: {path}")
        logger.info(f"\n⏱️  COMPLETE PIPELINE completed in {pipeline_elapsed:.2f} seconds ({pipeline_elapsed/60:.2f} minutes)")
    
    elif args.search:
        # Search mode
        if not args.query:
            logger.error("--query is required for search")
            sys.exit(1)
        
        embeddings_path = Path(args.embeddings_file) if args.embeddings_file else None
        
        # If --answer is used, we need context, so force build_context
        build_context = args.build_context or args.answer
        
        # Get search config with CLI override
        search_config = config.search if config else {}
        context_config = config.context if config else {}
        
        top_k = args.top_k if args.top_k != 5 else None
        similarity_metric = args.similarity_metric if args.similarity_metric != 'cosine' else None
        max_context_tokens = args.max_context_tokens if args.max_context_tokens != 8000 else None
        context_order = args.context_order if args.context_order != 'relevance' else None
        reduce_redundancy = None if args.no_reduce_redundancy else None  # Will use config default
        
        results, context_block = pipeline.step5_search(
            query=args.query,
            top_k=top_k,
            embeddings_file=embeddings_path,
            similarity_metric=similarity_metric,
            source_filter=args.source_filter,
            build_context=build_context,
            max_context_tokens=max_context_tokens,
            context_order=context_order,
            reduce_redundancy=reduce_redundancy
        )
        
        # Verify we have search results
        if not results:
            logger.error("No search results found. Check your query and embeddings file.")
            sys.exit(1)
        
        logger.info(f"Found {len(results)} search results")
        
        # Print results
        print(f"\n{'='*60}")
        print(f"Search Results for: '{args.query}'")
        print(f"{'='*60}\n")
        
        for i, (entry, score) in enumerate(results, 1):
            print(f"Result {i} (similarity: {score:.4f}):")
            print(f"  ID: {entry.get('id')}")
            print(f"  Source: {entry.get('source')}")
            if 'title' in entry:
                print(f"  Title: {entry.get('title')}")
            if 'summary' in entry:
                print(f"  Summary: {entry.get('summary')}")
            text_preview = entry.get('text', '')[:200]
            print(f"  Text: {text_preview}...")
            if 'entities' in entry:
                entities = entry['entities']
                if entities.get('people'):
                    print(f"  People: {', '.join(entities['people'][:3])}")
                if entities.get('locations'):
                    print(f"  Locations: {', '.join(entities['locations'][:3])}")
            print()
        
        # Print context block if built
        if context_block:
            print(f"\n{'='*60}")
            print(f"CONTEXT BLOCK (for LLM input)")
            print(f"{'='*60}\n")
            try:
                print(context_block)
            except UnicodeEncodeError:
                # Handle Unicode encoding issues on Windows
                print(context_block.encode('utf-8', errors='replace').decode('utf-8', errors='replace'))
            print(f"\n{'='*60}\n")
        
        # Generate answer if requested
        if args.answer:
            if not context_block:
                logger.warning("--answer requires --build-context. Building context now...")
                # Build context if not already built
                if not CONTEXT_BUILDER_AVAILABLE:
                    logger.error("Context builder not available. Install tiktoken.")
                    sys.exit(1)
                try:
                    # Use config for ContextBuilder if available
                    context_config = config.context if config else {}
                    similarity_threshold = context_config.get('similarity_threshold', 0.7)
                    clean_text = context_config.get('clean_text', True)
                    
                    builder = ContextBuilder(
                        max_tokens=args.max_context_tokens,
                        order_by=args.context_order,
                        reduce_redundancy=not args.no_reduce_redundancy,
                        similarity_threshold=similarity_threshold,
                        clean_text=clean_text,
                        config=config
                    )
                    context_block = builder.build_context(results)
                    if not context_block or not context_block.strip():
                        logger.error("Context block is empty. Check if search returned results.")
                        sys.exit(1)
                    logger.info(f"Built context block: {len(builder.encoder.encode(context_block))} tokens, {len(context_block)} characters")
                except Exception as e:
                    logger.error(f"Failed to build context: {e}", exc_info=True)
                    sys.exit(1)
            
            # Verify context is not empty before generating answer
            if not context_block or not context_block.strip():
                logger.error("Cannot generate answer: context block is empty")
                logger.error("Make sure --build-context is used or search returned results")
                sys.exit(1)
            
            logger.info(f"Generating answer with context ({len(context_block)} characters)")
            try:
                # Get QA config with CLI override
                qa_config = config.llm.get('qa', {}) if config else {}
                provider = args.llm_provider if args.llm_provider != 'ollama' else None
                model = args.llm_model if args.llm_model != 'llama3' else None
                temperature = args.llm_temperature if args.llm_temperature != 0.7 else None
                max_tokens = args.max_answer_tokens if args.max_answer_tokens != 500 else None
                
                answer = pipeline.step6_answer(
                    question=args.query,
                    context=context_block,
                    provider=provider,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                print(f"\n{'='*60}")
                print(f"ANSWER")
                print(f"{'='*60}\n")
                try:
                    print(answer)
                except UnicodeEncodeError:
                    # Handle Unicode encoding issues on Windows
                    print(answer.encode('utf-8', errors='replace').decode('utf-8', errors='replace'))
                print(f"\n{'='*60}\n")
                
            except Exception as e:
                logger.error(f"Failed to generate answer: {e}", exc_info=True)
                sys.exit(1)
    
    else:
        # Run individual steps
        current_file = None
        
        if args.chunk:
            current_file = pipeline.step1_chunk()
        
        if args.refine and current_file:
            # Use CLI args if different from defaults, otherwise use config
            provider = args.refine_provider if args.refine_provider != 'ollama' else None
            model = args.refine_model if args.refine_model != 'llama3' else None
            enable_merging = None if args.no_merging else None  # Will use config default
            enable_metadata = None if args.no_metadata else None  # Will use config default
            
            current_file = pipeline.step2_refine(
                current_file,
                provider=provider,
                model=model,
                enable_merging=enable_merging,
                enable_metadata=enable_metadata
            )
        
        if args.extract_entities and current_file:
            method = args.entity_method if args.entity_method != 'spacy' else None
            current_file = pipeline.step3_extract_entities(
                current_file,
                method=method
            )
        
        if args.embed and current_file:
            batch_size = args.batch_size if args.batch_size != 32 else None
            pipeline.step4_embed(current_file, batch_size=batch_size)
        
        if not any([args.clean, args.chunk, args.refine, args.extract_entities, args.embed, args.search]):
            logger.warning("No steps specified. Use --run-all or specify individual steps.")
            parser.print_help()


if __name__ == "__main__":
    main()
