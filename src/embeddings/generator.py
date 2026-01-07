"""
Embedding module for generating vector representations of Knowledge Layer chunks.

This module provides functionality to:
- Generate embeddings using static (pre-trained) models
- Support sentence/paragraph level embeddings
- Produce deterministic, fast embeddings
- Support dimensions from 384-1536
- Output one vector per chunk for search purposes
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from src.config import Config

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
except ImportError:
    print("Error: sentence-transformers and numpy are required.")
    print("Install with: pip install sentence-transformers numpy")
    sys.exit(1)

from src.knowledge_layer import read_knowledge_entries, validate_knowledge_entry

try:
    from src.embeddings.metadata import (
        EmbeddingMetadata,
        IndexHeader,
        generate_config_hash,
        generate_embedding_fingerprint,
        get_model_version
    )
    METADATA_AVAILABLE = True
except ImportError:
    METADATA_AVAILABLE = False
    EmbeddingMetadata = None
    IndexHeader = None
    generate_config_hash = None
    generate_embedding_fingerprint = None
    get_model_version = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Model recommendations by dimension
MODEL_RECOMMENDATIONS = {
    384: "all-MiniLM-L6-v2",  # Fast, 384 dimensions
    768: "all-mpnet-base-v2",  # High quality, 768 dimensions
    # Note: 1536 dimensions not available in standard sentence-transformers models
    # For 1536d, consider: OpenAI text-embedding-ada-002 API or Vec2Vec transformation
}
HUNGARIAN_MODELS = {
    "best": "intfloat/e5-mistral-7b-instruct",  # Best overall performance (7B, 4096d - requires high memory)
    "fast": "intfloat/multilingual-e5-large-instruct",  # Good balance (large model)
    "gritlm": "GritLM/GritLM-7B",  # Best for classification (7B - requires high memory)
    "multilingual-384": "paraphrase-multilingual-MiniLM-L12-v2",  # 384d, multilingual including Hungarian (recommended for low memory)
    "multilingual-768": "paraphrase-multilingual-mpnet-base-v2",  # 768d, multilingual including Hungarian
    "baseline": "all-MiniLM-L6-v2"  # 384d, English-focused
}


# Default model (384 dimensions - fast and efficient, supports Hungarian)
# Changed from "best" (7B model) to "multilingual-384" for lower memory usage
DEFAULT_MODEL = "baseline"


class EmbeddingGenerator:
    """Generator for creating embeddings from Knowledge Layer entries."""
    
    def __init__(
        self,
        model: Optional[str] = None,
        model_name: Optional[str] = None,  # Deprecated, use model
        device: Optional[str] = None,
        batch_size: Optional[int] = None,
        config: Optional['Config'] = None
    ):
        """
        Initialize the embedding generator.
        
        Args:
            model: Name of the sentence-transformers model to use (overrides config)
            model_name: Deprecated, use model instead
            device: Device to run on ('cpu', 'cuda', or None for auto-detection) (overrides config)
            batch_size: Batch size for embedding generation (overrides config)
            config: Optional Config instance
        """
        # Support deprecated model_name parameter
        if model_name and not model:
            model = model_name
        
        # Get config values with priority: function arg > config > default
        embedding_config = {}
        if config:
            embedding_config = config.embedding
        
        model = model if model else embedding_config.get('model', DEFAULT_MODEL)
        device = device if device else embedding_config.get('device', None)
        self.batch_size = batch_size if batch_size is not None else embedding_config.get('batch_size', 32)
        
        # Resolve model name from HUNGARIAN_MODELS if it's a key
        actual_model_name = HUNGARIAN_MODELS.get(model, model)
        if actual_model_name != model:
            logger.info(f"Resolved model alias '{model}' to '{actual_model_name}'")
        
        logger.info(f"Loading embedding model: {actual_model_name}")
        try:
            self.model = SentenceTransformer(actual_model_name, device=device)
            self.model.eval()  # Set to evaluation mode for deterministic outputs
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            self.model_name = actual_model_name  # Store for metadata generation
            logger.info(f"Model loaded successfully. Embedding dimension: {self.embedding_dim}")
            
            # Validate dimension is in acceptable range
            if self.embedding_dim < 384 or self.embedding_dim > 1536:
                logger.warning(f"Embedding dimension {self.embedding_dim} is outside recommended range (384-1536)")
        except Exception as e:
            logger.error(f"Failed to load model {model}: {e}")
            raise
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text chunk.
        
        Args:
            text: Text to embed
        
        Returns:
            List of floats representing the embedding vector
        """
        embedding = self.model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return embedding.tolist()
    
    def embed_batch(self, texts: List[str], batch_size: Optional[int] = None) -> List[List[float]]:
        """
        Generate embeddings for multiple texts efficiently.
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process at once (defaults to self.batch_size)
        
        Returns:
            List of embedding vectors
        """
        # Use provided batch_size or instance default
        effective_batch_size = batch_size if batch_size is not None else self.batch_size
        
        embeddings = self.model.encode(
            texts,
            batch_size=effective_batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 100
        )
        return embeddings.tolist()
    
    def embed_knowledge_entries(
        self,
        entries: List[Dict[str, Any]],
        batch_size: Optional[int] = None,
        generate_metadata: bool = True,
        chunking_config: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[Dict[str, Any]], Optional[Any]]:
        """
        Generate embeddings for Knowledge Layer entries.
        
        Args:
            entries: List of Knowledge Layer entry dictionaries
            batch_size: Number of entries to process at once
            generate_metadata: Whether to generate embedding metadata
            chunking_config: Optional chunking configuration dict for config hash
        
        Returns:
            Tuple of (entries with embeddings, EmbeddingMetadata if generated)
        """
        # Validate entries
        for entry in entries:
            if not validate_knowledge_entry(entry):
                raise ValueError(f"Invalid Knowledge Layer entry: {entry}")
        
        # Extract texts
        texts = [entry["text"] for entry in entries]
        
        # Use provided batch_size or instance default
        effective_batch_size = batch_size if batch_size is not None else self.batch_size
        
        # Generate embeddings in batches
        logger.info(f"Generating embeddings for {len(texts)} chunks...")
        embeddings = self.embed_batch(texts, batch_size=effective_batch_size)
        
        # Generate metadata if requested
        metadata = None
        if generate_metadata and METADATA_AVAILABLE:
            try:
                model_version = get_model_version(self.model_name)
                metadata = EmbeddingMetadata(
                    model_name=self.model_name,
                    model_version=model_version,
                    embedding_dim=self.embedding_dim,
                    build_timestamp=None  # Will be set in __post_init__
                )
                
                # Generate config hash if chunking config provided
                if chunking_config:
                    config_hash = generate_config_hash(**chunking_config)
                    metadata.config_hash = config_hash
                    logger.info(f"Generated config hash: {config_hash[:8]}...")
            except Exception as e:
                logger.warning(f"Failed to generate embedding metadata: {e}")
        
        # Add embeddings to entries
        result = []
        for entry, embedding in zip(entries, embeddings):
            entry_with_embedding = entry.copy()
            entry_with_embedding["embedding"] = embedding
            result.append(entry_with_embedding)
        
        # Return tuple if metadata was generated, otherwise just the list (backward compatibility)
        if generate_metadata and metadata is not None:
            return result, metadata
        return result


def write_embeddings(
    entries_with_embeddings: List[Dict[str, Any]],
    output_path: Path
) -> None:
    """
    Write Knowledge Layer entries with embeddings to JSONL file.
    
    Args:
        entries_with_embeddings: List of entries with embedding vectors
        output_path: Path to output JSONL file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in entries_with_embeddings:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
    logger.info(f"Wrote {len(entries_with_embeddings)} entries with embeddings to {output_path}")


def read_embeddings(input_path: Path) -> List[Dict[str, Any]]:
    """
    Read Knowledge Layer entries with embeddings from JSONL file.
    
    Args:
        input_path: Path to input JSONL file
    
    Returns:
        List of entries with embeddings
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    entries = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                entries.append(entry)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at line {line_num}: {e}")
    
    return entries


def main():
    parser = argparse.ArgumentParser(
        description='Generate embeddings for Knowledge Layer chunks using static models.'
    )
    parser.add_argument(
        '--input-file',
        type=str,
        required=True,
        help='Input JSONL file with Knowledge Layer entries'
    )
    parser.add_argument(
        '--output-file',
        type=str,
        required=True,
        help='Output JSONL file with embeddings'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=DEFAULT_MODEL,
        help=f'Model name (default: {DEFAULT_MODEL}). '
             f'Recommended: all-MiniLM-L6-v2 (384d), all-mpnet-base-v2 (768d)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for embedding generation (default: 32)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use: cpu, cuda, or None for auto-detection (default: None)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.batch_size <= 0:
        logger.error("Batch size must be positive")
        sys.exit(1)
    
    input_path = Path(args.input_file)
    output_path = Path(args.output_file)
    
    # Validate input file exists
    if not input_path.exists():
        logger.error(f"Input file does not exist: {input_path}")
        sys.exit(1)
    
    # Read Knowledge Layer entries
    try:
        logger.info(f"Reading Knowledge Layer entries from {input_path}")
        entries = read_knowledge_entries(input_path, validate=True)
        logger.info(f"Loaded {len(entries)} entries")
    except Exception as e:
        logger.error(f"Failed to read input file: {e}", exc_info=True)
        sys.exit(1)
    
    if not entries:
        logger.warning("No entries to process")
        return
    
    # Initialize embedding generator
    try:
        generator = EmbeddingGenerator(model_name=args.model, device=args.device)
    except Exception as e:
        logger.error(f"Failed to initialize embedding generator: {e}", exc_info=True)
        sys.exit(1)
    
    # Generate embeddings
    try:
        entries_with_embeddings = generator.embed_knowledge_entries(
            entries,
            batch_size=args.batch_size
        )
        logger.info(f"Successfully generated embeddings for {len(entries_with_embeddings)} entries")
    except Exception as e:
        logger.error(f"Failed to generate embeddings: {e}", exc_info=True)
        sys.exit(1)
    
    # Write output
    try:
        write_embeddings(entries_with_embeddings, output_path)
        logger.info(f"Successfully wrote embeddings to {output_path}")
        logger.info(f"Embedding dimension: {generator.embedding_dim}")
    except Exception as e:
        logger.error(f"✗ Failed to write output file: {e}", exc_info=True)
        sys.exit(1)
    
    logger.info("Processing complete")


if __name__ == "__main__":
    main()
