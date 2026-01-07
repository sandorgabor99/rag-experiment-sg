"""
Gold Q/A dataset loader.

Loads and validates gold standard question-answer pairs for evaluation.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class GoldQAPair:
    """Single gold Q/A pair."""
    
    def __init__(
        self,
        id: str,
        question: str,
        ground_truth_sources: List[str],
        ground_truth_entities: List[str],
        expected_answer_contains: List[str],
        category: str = "general",
        language: str = "hu"
    ):
        self.id = id
        self.question = question
        self.ground_truth_sources = ground_truth_sources
        self.ground_truth_entities = ground_truth_entities
        self.expected_answer_contains = expected_answer_contains
        self.category = category
        self.language = language
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'question': self.question,
            'ground_truth_sources': self.ground_truth_sources,
            'ground_truth_entities': self.ground_truth_entities,
            'expected_answer_contains': self.expected_answer_contains,
            'category': self.category,
            'language': self.language
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GoldQAPair':
        """Create from dictionary."""
        return cls(
            id=data['id'],
            question=data['question'],
            ground_truth_sources=data.get('ground_truth_sources', []),
            ground_truth_entities=data.get('ground_truth_entities', []),
            expected_answer_contains=data.get('expected_answer_contains', []),
            category=data.get('category', 'general'),
            language=data.get('language', 'hu')
        )


def load_gold_dataset(dataset_path: Path) -> List[GoldQAPair]:
    """
    Load gold Q/A dataset from JSONL file.
    
    Args:
        dataset_path: Path to gold_qa.jsonl file
    
    Returns:
        List of GoldQAPair instances
    """
    if not dataset_path.exists():
        raise FileNotFoundError(f"Gold dataset not found: {dataset_path}")
    
    pairs = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
                pair = GoldQAPair.from_dict(data)
                pairs.append(pair)
            except Exception as e:
                logger.warning(f"Skipping invalid entry at line {line_num}: {e}")
    
    logger.info(f"Loaded {len(pairs)} gold Q/A pairs from {dataset_path}")
    return pairs


def save_gold_dataset(pairs: List[GoldQAPair], output_path: Path) -> None:
    """
    Save gold Q/A dataset to JSONL file.
    
    Args:
        pairs: List of GoldQAPair instances
        output_path: Path to save dataset
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for pair in pairs:
            json_str = json.dumps(pair.to_dict(), ensure_ascii=False)
            f.write(json_str + '\n')
    
    logger.info(f"Saved {len(pairs)} gold Q/A pairs to {output_path}")
