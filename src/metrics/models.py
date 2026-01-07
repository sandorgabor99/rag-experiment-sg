"""
Metric data models for observability.

Defines dataclasses for tracking retrieval, context, and latency metrics.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
from datetime import datetime


@dataclass
class RetrievalMetrics:
    """Metrics for retrieval/search phase."""
    top_k_similarity_avg: float = 0.0
    top_1_vs_top_k_spread: float = 0.0
    entity_boost_applied: bool = False
    hit_density: float = 0.0  # Ratio of same source vs diverse sources
    query_length: int = 0  # Character count
    query_tokens: int = 0  # Token count (if available)
    retrieval_time_ms: float = 0.0
    results_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class ContextMetrics:
    """Metrics for context building phase."""
    context_tokens_used: int = 0
    unique_source_count: int = 0
    redundancy_ratio: float = 0.0  # (chunks_before - chunks_after) / chunks_before
    truncation_events: bool = False
    chunks_included: int = 0
    chunks_excluded: int = 0
    chunks_before_dedup: int = 0
    chunks_after_dedup: int = 0
    context_build_time_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class LatencyMetrics:
    """Latency breakdown metrics."""
    embed_time_ms: float = 0.0
    search_time_ms: float = 0.0
    context_build_time_ms: float = 0.0
    llm_time_ms: float = 0.0
    total_time_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class PipelineMetrics:
    """Complete pipeline metrics for a single request."""
    request_id: str = ""
    timestamp: str = ""
    pipeline_version: str = "1.0.0"
    query: str = ""
    retrieval_metrics: Optional[RetrievalMetrics] = None
    context_metrics: Optional[ContextMetrics] = None
    latency_metrics: Optional[LatencyMetrics] = None
    answer_length: int = 0
    success: bool = True
    error: Optional[str] = None
    
    def __post_init__(self):
        """Set default timestamp if not provided."""
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        if self.retrieval_metrics:
            result['retrieval_metrics'] = self.retrieval_metrics.to_dict()
        if self.context_metrics:
            result['context_metrics'] = self.context_metrics.to_dict()
        if self.latency_metrics:
            result['latency_metrics'] = self.latency_metrics.to_dict()
        return result
