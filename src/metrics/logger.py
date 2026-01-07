"""
Structured metrics logger for RAG pipeline.

Logs metrics in JSONL format for easy analysis and aggregation.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional
from .models import PipelineMetrics

logger = logging.getLogger(__name__)


class MetricsLogger:
    """
    Structured logger for pipeline metrics.
    
    Writes metrics to JSONL files with daily rotation.
    """
    
    def __init__(
        self,
        log_dir: Path = Path("data/metrics"),
        pipeline_version: str = "1.0.0",
        enabled: bool = True
    ):
        """
        Initialize metrics logger.
        
        Args:
            log_dir: Directory for metric log files
            pipeline_version: Version string for pipeline
            enabled: Whether logging is enabled
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.pipeline_version = pipeline_version
        self.enabled = enabled
        self._current_date = None
        self._log_file = None
    
    def _get_log_file(self) -> Path:
        """Get log file path for current date."""
        today = datetime.utcnow().date().isoformat()
        if today != self._current_date:
            self._current_date = today
            self._log_file = self.log_dir / f"rag_metrics_{today}.jsonl"
        return self._log_file
    
    def log(self, metrics: PipelineMetrics) -> None:
        """
        Log pipeline metrics.
        
        Args:
            metrics: PipelineMetrics instance to log
        """
        if not self.enabled:
            return
        
        # Ensure pipeline version is set
        if not metrics.pipeline_version:
            metrics.pipeline_version = self.pipeline_version
        
        try:
            log_file = self._get_log_file()
            with open(log_file, 'a', encoding='utf-8') as f:
                json_str = json.dumps(metrics.to_dict(), ensure_ascii=False)
                f.write(json_str + '\n')
        except Exception as e:
            logger.error(f"Failed to log metrics: {e}", exc_info=True)
    
    def log_dict(self, metrics_dict: dict) -> None:
        """
        Log metrics from dictionary.
        
        Args:
            metrics_dict: Dictionary with metric data
        """
        if not self.enabled:
            return
        
        try:
            # Create PipelineMetrics from dict
            metrics = PipelineMetrics(**metrics_dict)
            self.log(metrics)
        except Exception as e:
            logger.error(f"Failed to log metrics dict: {e}", exc_info=True)
            # Fallback: log raw dict
            try:
                log_file = self._get_log_file()
                with open(log_file, 'a', encoding='utf-8') as f:
                    json_str = json.dumps(metrics_dict, ensure_ascii=False)
                    f.write(json_str + '\n')
            except Exception as e2:
                logger.error(f"Failed to log raw metrics: {e2}")
