"""
Embedding metadata models and utilities for versioning and tracking.

This module provides data structures and utilities for tracking embedding
metadata including model information, build timestamps, and configuration hashes.
"""

import hashlib
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingMetadata:
    """
    Metadata for embedding generation.
    
    Tracks model information, dimensions, build time, and configuration
    to enable reproducibility and versioning.
    """
    model_name: str
    model_version: Optional[str] = None
    embedding_dim: int = 0
    build_timestamp: Optional[str] = None
    config_hash: Optional[str] = None
    
    def __post_init__(self):
        """Set default timestamp if not provided."""
        if self.build_timestamp is None:
            self.build_timestamp = datetime.utcnow().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EmbeddingMetadata':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class IndexHeader:
    """
    Header information for vector store index.
    
    Contains versioning, fingerprinting, and metadata to enable
    index validation and lifecycle management.
    """
    version: str = "1.0"
    embedding_fingerprint: Optional[str] = None
    chunker_config_hash: Optional[str] = None
    metadata: Optional[EmbeddingMetadata] = None
    created_at: Optional[str] = None
    chunk_count: int = 0
    
    def __post_init__(self):
        """Set default timestamp if not provided."""
        if self.created_at is None:
            self.created_at = datetime.utcnow().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        if self.metadata:
            result['metadata'] = self.metadata.to_dict()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'IndexHeader':
        """Create from dictionary."""
        if 'metadata' in data and isinstance(data['metadata'], dict):
            data['metadata'] = EmbeddingMetadata.from_dict(data['metadata'])
        return cls(**data)


def generate_config_hash(
    chunking_strategy: str,
    chunk_size: int,
    overlap: int,
    embedding_model: str,
    **kwargs
) -> str:
    """
    Generate a hash of chunking configuration.
    
    Args:
        chunking_strategy: Chunking strategy name
        chunk_size: Chunk size in tokens
        overlap: Overlap in tokens
        embedding_model: Embedding model name
        **kwargs: Additional config parameters
    
    Returns:
        MD5 hash of configuration
    """
    config_dict = {
        'chunking_strategy': chunking_strategy,
        'chunk_size': chunk_size,
        'overlap': overlap,
        'embedding_model': embedding_model,
        **kwargs
    }
    
    # Sort keys for consistent hashing
    config_str = json.dumps(config_dict, sort_keys=True)
    return hashlib.md5(config_str.encode('utf-8')).hexdigest()


def generate_embedding_fingerprint(embeddings: list) -> str:
    """
    Generate a fingerprint (hash) of embeddings for integrity checking.
    
    Args:
        embeddings: List of embedding vectors
    
    Returns:
        SHA256 hash of embeddings
    """
    # Convert to JSON string for hashing
    # Use first few embeddings and total count for fingerprint
    if not embeddings:
        return hashlib.sha256(b'').hexdigest()
    
    # Sample strategy: hash first embedding, last embedding, and count
    sample = {
        'first': embeddings[0][:10] if len(embeddings[0]) > 10 else embeddings[0],
        'last': embeddings[-1][:10] if len(embeddings[-1]) > 10 else embeddings[-1],
        'count': len(embeddings),
        'dim': len(embeddings[0]) if embeddings else 0
    }
    
    sample_str = json.dumps(sample, sort_keys=True)
    return hashlib.sha256(sample_str.encode('utf-8')).hexdigest()


def get_model_version(model_name: str) -> Optional[str]:
    """
    Get version information for a sentence-transformers model.
    
    Args:
        model_name: Model name or alias
    
    Returns:
        Version string if available, None otherwise
    """
    try:
        from sentence_transformers import __version__ as st_version
        # For now, return sentence-transformers version
        # In future, could query model metadata
        return f"sentence-transformers-{st_version}"
    except ImportError:
        return None
