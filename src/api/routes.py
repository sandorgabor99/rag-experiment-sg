"""
FastAPI route handlers.
"""

import logging
import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware

from .models import (
    QueryRequest,
    QueryResponse,
    SourceInfo,
    MetricsSummary,
    HealthResponse,
    MetadataResponse
)
from .dependencies import get_pipeline, get_searcher, get_embedding_model

try:
    from src.security.sanitizer import sanitize_query, sanitize_context
    SECURITY_AVAILABLE = True
except ImportError:
    SECURITY_AVAILABLE = False
    sanitize_query = lambda x, **kwargs: x
    sanitize_context = lambda x: x

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    pipeline = Depends(get_pipeline),
    searcher = Depends(get_searcher),
    model = Depends(get_embedding_model)
):
    """
    Main query endpoint for RAG system.
    
    Processes query, searches knowledge base, builds context, and generates answer.
    """
    request_id = str(uuid.uuid4())
    
    try:
        # Sanitize query
        sanitized_query = sanitize_query(request.query, max_length=1000)
        if not sanitized_query:
            raise HTTPException(status_code=400, detail="Query is empty after sanitization")
        
        # Determine top_k based on mode
        if request.mode == "all":
            top_k = 50  # Higher for "all" queries
        else:
            top_k = request.top_k or 20  # Use request value or default
        
        # Search
        results, context_block = pipeline.step5_search(
            query=sanitized_query,
            top_k=top_k,
            build_context=True,
            source_filter=request.source_filter
        )
        
        if not results:
            raise HTTPException(status_code=404, detail="No results found for query")
        
        if not context_block:
            raise HTTPException(status_code=500, detail="Failed to build context")
        
        # Sanitize context before sending to LLM
        sanitized_context = sanitize_context(context_block)
        
        # Generate answer
        answer = pipeline.step6_answer(
            question=sanitized_query,
            context=sanitized_context
        )
        
        # Build sources list
        sources = []
        for entry, score in results[:10]:  # Limit to top 10 for response
            sources.append(SourceInfo(
                source=entry.get('source', 'unknown'),
                chunk_id=entry.get('id', 'unknown'),
                similarity=float(score),
                text_preview=entry.get('text', '')[:200]
            ))
        
        # Calculate confidence (heuristic: average similarity)
        if results:
            avg_similarity = sum(score for _, score in results[:top_k]) / min(len(results), top_k)
        else:
            avg_similarity = 0.0
        
        # Build metrics summary (from pipeline's last metrics if available)
        metrics_summary = MetricsSummary()
        if hasattr(pipeline, '_last_metrics') and pipeline._last_metrics:
            last_metrics = pipeline._last_metrics
            if last_metrics.retrieval_metrics:
                metrics_summary.retrieval = last_metrics.retrieval_metrics.to_dict()
            if last_metrics.context_metrics:
                metrics_summary.context = last_metrics.context_metrics.to_dict()
            if last_metrics.latency_metrics:
                metrics_summary.latency = last_metrics.latency_metrics.to_dict()
        
        # Build debug info if requested
        debug_info = None
        if request.debug:
            debug_info = {
                'context_block': context_block[:500],  # First 500 chars
                'results_count': len(results),
                'top_k_used': top_k
            }
        
        return QueryResponse(
            answer=answer,
            sources=sources,
            confidence=float(avg_similarity),
            metrics=metrics_summary,
            request_id=request_id,
            debug_info=debug_info
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")


@router.get("/health", response_model=HealthResponse)
async def health(pipeline = Depends(get_pipeline)):
    """Health check endpoint."""
    index_loaded = hasattr(pipeline, '_searcher') and pipeline._searcher is not None
    
    index_version = None
    index_chunk_count = None
    
    if index_loaded:
        try:
            searcher = pipeline._searcher
            if hasattr(searcher, 'store') and hasattr(searcher.store, 'index_metadata'):
                if searcher.store.index_metadata:
                    index_version = searcher.store.index_metadata.version
                    index_chunk_count = searcher.store.index_metadata.chunk_count
        except Exception as e:
            logger.warning(f"Failed to get index metadata: {e}")
    
    return HealthResponse(
        status="healthy" if index_loaded else "unhealthy",
        index_loaded=index_loaded,
        index_version=index_version,
        index_chunk_count=index_chunk_count
    )


@router.get("/metadata", response_model=MetadataResponse)
async def metadata(pipeline = Depends(get_pipeline)):
    """Get index metadata."""
    try:
        searcher = get_searcher()
        if hasattr(searcher, 'store') and hasattr(searcher.store, 'index_metadata'):
            index_metadata = searcher.store.index_metadata
            if index_metadata:
                return MetadataResponse(
                    index_metadata=index_metadata.to_dict(),
                    embedding_model=index_metadata.metadata.model_name if index_metadata.metadata else None,
                    embedding_dim=index_metadata.metadata.embedding_dim if index_metadata.metadata else None,
                    config_hash=index_metadata.chunker_config_hash,
                    created_at=index_metadata.created_at
                )
        
        return MetadataResponse()
    except Exception as e:
        logger.error(f"Failed to get metadata: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get metadata: {str(e)}")


@router.get("/metrics")
async def metrics(pipeline = Depends(get_pipeline)):
    """Get aggregated metrics (last N requests)."""
    # For now, return empty dict
    # In future, could aggregate from metrics logs
    return {
        "message": "Metrics aggregation not yet implemented",
        "metrics_log_dir": str(pipeline.metrics_logger.log_dir) if pipeline.metrics_logger else None
    }
