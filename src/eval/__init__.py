"""
Evaluation module for RAG pipeline.

Provides tools for evaluating retrieval quality and measuring improvements.
"""

from .dataset import load_gold_dataset
from .evaluator import Evaluator
from .runner import run_evaluation

__all__ = ['load_gold_dataset', 'Evaluator', 'run_evaluation']
