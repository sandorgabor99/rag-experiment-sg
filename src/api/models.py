"""
Request and response models for FastAPI endpoints.
"""

from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field, validator


class SourceInfo(BaseModel):
    """Information about a source chunk."""
    source: str
    chunk_id: str
    similarity: float = Field(..., ge=0.0, le=1.0)
    text_preview: str = Field(..., max_length=200)


class MetricsSummary(BaseModel):
    """Summary of pipeline metrics."""
    retrieval: Optional[Dict[str, Any]] = None
    context: Optional[Dict[str, Any]] = None
    latency: Optional[Dict[str, Any]] = None


class QueryRequest(BaseModel):
    """Request model for /query endpoint."""
    query: str = Field(..., min_length=1, max_length=1000)
    language: Optional[str] = Field(default="hu", pattern="^(hu|en)$")
    top_k: Optional[int] = Field(default=None, ge=1, le=100)
    mode: Literal["entity", "all", "default"] = "default"
    debug: bool = False
    source_filter: Optional[str] = None
    
    @validator('query')
    def validate_query(cls, v):
        """Validate query is not empty after stripping."""
        if not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()


class QueryResponse(BaseModel):
    """Response model for /query endpoint."""
    answer: str
    sources: List[SourceInfo]
    confidence: float = Field(..., ge=0.0, le=1.0)
    metrics: MetricsSummary
    request_id: str
    debug_info: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    """Response model for /health endpoint."""
    status: str
    index_loaded: bool
    index_version: Optional[str] = None
    index_chunk_count: Optional[int] = None


class MetadataResponse(BaseModel):
    """Response model for /metadata endpoint."""
    index_metadata: Optional[Dict[str, Any]] = None
    embedding_model: Optional[str] = None
    embedding_dim: Optional[int] = None
    config_hash: Optional[str] = None
    created_at: Optional[str] = None
