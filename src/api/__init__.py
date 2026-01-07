"""
FastAPI serving layer for RAG pipeline.

Provides REST API endpoints for querying the knowledge base.
"""

from .main import app

__all__ = ['app']
