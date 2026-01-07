"""
Configuration module for Knowledge Layer RAG Pipeline.

Provides centralized configuration management with support for:
- YAML configuration files
- Environment variable overrides
- CLI argument overrides (handled in orchestrator)
"""

from .config import Config, load_config

__all__ = ['Config', 'load_config']
