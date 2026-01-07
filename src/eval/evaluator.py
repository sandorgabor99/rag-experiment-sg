"""
Evaluation engine for RAG pipeline.

Runs evaluation on gold Q/A dataset and calculates metrics.
"""

import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from .dataset import GoldQAPair, load_gold_dataset
from .metrics import EvaluationMetrics, calculate_hit_at_k, calculate_entity_hit_rate

logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluator for RAG pipeline."""
    
    def __init__(self, pipeline):
        """
        Initialize evaluator.
        
        Args:
            pipeline: KnowledgeLayerPipeline instance
        """
        self.pipeline = pipeline
    
    def evaluate(
        self,
        gold_pairs: List[GoldQAPair],
        top_k: int = 10,
        boost_entity_names: bool = True
    ) -> EvaluationMetrics:
        """
        Evaluate pipeline on gold Q/A pairs.
        
        Args:
            gold_pairs: List of gold Q/A pairs
            top_k: Number of results to retrieve
            boost_entity_names: Whether to use entity boosting
        
        Returns:
            EvaluationMetrics instance
        """
        logger.info(f"Evaluating on {len(gold_pairs)} gold Q/A pairs...")
        
        hits_at_1 = 0
        hits_at_3 = 0
        hits_at_k = 0
        entity_hits = 0.0
        total_similarity = 0.0
        
        for i, pair in enumerate(gold_pairs):
            try:
                # Search
                results, _ = self.pipeline.step5_search(
                    query=pair.question,
                    top_k=top_k,
                    build_context=False,
                    diverse_search=False
                )
                
                if not results:
                    logger.warning(f"No results for query {pair.id}: {pair.question}")
                    continue
                
                # Calculate hits
                if calculate_hit_at_k(results, pair.ground_truth_sources, pair.ground_truth_entities, 1):
                    hits_at_1 += 1
                if calculate_hit_at_k(results, pair.ground_truth_sources, pair.ground_truth_entities, 3):
                    hits_at_3 += 1
                if calculate_hit_at_k(results, pair.ground_truth_sources, pair.ground_truth_entities, top_k):
                    hits_at_k += 1
                
                # Calculate entity hit rate
                entity_hit_rate = calculate_entity_hit_rate(
                    results,
                    pair.ground_truth_entities,
                    top_k
                )
                entity_hits += entity_hit_rate
                
                # Average similarity
                if results:
                    avg_sim = sum(score for _, score in results[:top_k]) / min(len(results), top_k)
                    total_similarity += avg_sim
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{len(gold_pairs)} queries")
            
            except Exception as e:
                logger.error(f"Error evaluating query {pair.id}: {e}", exc_info=True)
                continue
        
        # Calculate final metrics
        total = len(gold_pairs)
        metrics = EvaluationMetrics(
            hit_at_1=hits_at_1 / total if total > 0 else 0.0,
            hit_at_3=hits_at_3 / total if total > 0 else 0.0,
            hit_at_k=hits_at_k / total if total > 0 else 0.0,
            entity_hit_rate=entity_hits / total if total > 0 else 0.0,
            avg_similarity=total_similarity / total if total > 0 else 0.0,
            total_queries=total
        )
        
        logger.info(f"Evaluation complete:")
        logger.info(f"  Hit@1: {metrics.hit_at_1:.3f}")
        logger.info(f"  Hit@3: {metrics.hit_at_3:.3f}")
        logger.info(f"  Hit@{top_k}: {metrics.hit_at_k:.3f}")
        logger.info(f"  Entity Hit Rate: {metrics.entity_hit_rate:.3f}")
        logger.info(f"  Avg Similarity: {metrics.avg_similarity:.3f}")
        
        return metrics
