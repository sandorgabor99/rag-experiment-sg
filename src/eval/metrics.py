"""
Evaluation metrics for RAG pipeline.

Calculates hit@k, entity hit rate, and other retrieval quality metrics.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Set


@dataclass
class EvaluationMetrics:
    """Metrics for a single evaluation run."""
    hit_at_1: float = 0.0
    hit_at_3: float = 0.0
    hit_at_k: float = 0.0
    entity_hit_rate: float = 0.0
    avg_similarity: float = 0.0
    total_queries: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'hit_at_1': self.hit_at_1,
            'hit_at_3': self.hit_at_3,
            'hit_at_k': self.hit_at_k,
            'entity_hit_rate': self.entity_hit_rate,
            'avg_similarity': self.avg_similarity,
            'total_queries': self.total_queries
        }


def calculate_hit_at_k(
    results: List[Dict[str, Any]],
    ground_truth_sources: List[str],
    ground_truth_entities: List[str],
    k: int
) -> bool:
    """
    Check if any of top-k results contain ground truth.
    
    Args:
        results: List of (entry, score) tuples from search
        ground_truth_sources: List of expected source files
        ground_truth_entities: List of expected entity names
        k: Number of top results to check
    
    Returns:
        True if ground truth found in top-k
    """
    top_k_results = results[:k]
    
    # Check sources
    result_sources = {r[0].get('source', '') for r in top_k_results}
    if any(gt_source in result_sources for gt_source in ground_truth_sources):
        return True
    
    # Check entities
    if ground_truth_entities:
        for entry, _ in top_k_results:
            # Check entity_name field
            entity_name = entry.get('entity_name', '').lower()
            name = entry.get('name', '').lower()
            text = entry.get('text', '').lower()
            
            for gt_entity in ground_truth_entities:
                gt_lower = gt_entity.lower()
                if gt_lower in entity_name or gt_lower in name or gt_lower in text:
                    return True
    
    return False


def calculate_entity_hit_rate(
    results: List[Dict[str, Any]],
    ground_truth_entities: List[str],
    k: int
) -> float:
    """
    Calculate percentage of ground truth entities found in top-k.
    
    Args:
        results: List of (entry, score) tuples
        ground_truth_entities: List of expected entity names
        k: Number of top results to check
    
    Returns:
        Ratio of entities found (0.0 to 1.0)
    """
    if not ground_truth_entities:
        return 0.0
    
    top_k_results = results[:k]
    found_entities = set()
    
    for entry, _ in top_k_results:
        entity_name = entry.get('entity_name', '').lower()
        name = entry.get('name', '').lower()
        text = entry.get('text', '').lower()
        
        for gt_entity in ground_truth_entities:
            gt_lower = gt_entity.lower()
            if gt_lower in entity_name or gt_lower in name or gt_lower in text:
                found_entities.add(gt_entity)
    
    return len(found_entities) / len(ground_truth_entities) if ground_truth_entities else 0.0
