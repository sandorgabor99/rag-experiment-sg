"""
Embeddings Module

Provides embedding generation for knowledge chunks.
"""

from .generator import EmbeddingGenerator, write_embeddings, read_embeddings

__all__ = [
    'EmbeddingGenerator',
    'write_embeddings',
    'read_embeddings'
]
