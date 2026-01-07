"""
Dependency injection for FastAPI endpoints.
"""

import logging
from pathlib import Path
from typing import Optional

from src.pipeline.orchestrator import KnowledgeLayerPipeline
from src.config import load_config

logger = logging.getLogger(__name__)

# Global pipeline instance (loaded on startup)
_pipeline: Optional[KnowledgeLayerPipeline] = None


def get_pipeline() -> KnowledgeLayerPipeline:
    """
    Get or create pipeline instance (singleton).
    
    Returns:
        KnowledgeLayerPipeline instance
    """
    global _pipeline
    
    if _pipeline is None:
        # Load config
        config = None
        try:
            config = load_config()
        except Exception as e:
            logger.warning(f"Failed to load config: {e}")
        
        # Initialize pipeline
        _pipeline = KnowledgeLayerPipeline(config=config)
        
        # Load vector store if it exists
        if _pipeline.vector_store_path.exists():
            try:
                from src.search.searcher import EmbeddingSearcher
                from sentence_transformers import SentenceTransformer
                
                # Load embedding model
                embedding_config = config.embedding if config else {}
                model_name = embedding_config.get('model', 'baseline')
                from src.embeddings.generator import HUNGARIAN_MODELS
                actual_model_name = HUNGARIAN_MODELS.get(model_name, model_name)
                model = SentenceTransformer(actual_model_name)
                
                # Create searcher (loads vector store)
                searcher = EmbeddingSearcher(store_path=_pipeline.vector_store_path)
                _pipeline._searcher = searcher  # Store for reuse
                _pipeline._embedding_model = model  # Store for reuse
                
                logger.info("Pipeline initialized and vector store loaded")
            except Exception as e:
                logger.error(f"Failed to load vector store: {e}", exc_info=True)
                raise
    
    return _pipeline


def get_searcher():
    """Get EmbeddingSearcher instance from pipeline."""
    pipeline = get_pipeline()
    if not hasattr(pipeline, '_searcher') or pipeline._searcher is None:
        raise RuntimeError("Searcher not initialized. Vector store may not be loaded.")
    return pipeline._searcher


def get_embedding_model():
    """Get SentenceTransformer model from pipeline."""
    pipeline = get_pipeline()
    if not hasattr(pipeline, '_embedding_model') or pipeline._embedding_model is None:
        raise RuntimeError("Embedding model not initialized.")
    return pipeline._embedding_model
