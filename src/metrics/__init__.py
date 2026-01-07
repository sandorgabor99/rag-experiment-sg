"""
Metrics and observability module.

Provides structured logging, metric collection, and analysis tools
for monitoring RAG pipeline performance.
"""

from .models import (
    RetrievalMetrics,
    ContextMetrics,
    LatencyMetrics,
    PipelineMetrics
)
from .logger import MetricsLogger

__all__ = [
    'RetrievalMetrics',
    'ContextMetrics',
    'LatencyMetrics',
    'PipelineMetrics',
    'MetricsLogger'
]
