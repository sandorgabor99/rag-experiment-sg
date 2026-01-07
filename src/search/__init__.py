"""
Search Module

Provides vector search, context building, and retrieval functionality.
"""

from .searcher import EmbeddingSearcher
from .vector_store import VectorStore, load_vector_store_from_embeddings, create_metadata_filter
from .context_builder import ContextBuilder, build_context_from_search

__all__ = [
    'EmbeddingSearcher',
    'VectorStore',
    'load_vector_store_from_embeddings',
    'create_metadata_filter',
    'ContextBuilder',
    'build_context_from_search'
]
