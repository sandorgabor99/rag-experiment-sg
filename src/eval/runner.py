"""
CLI runner for evaluation.
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.pipeline.orchestrator import KnowledgeLayerPipeline
from src.config import load_config
from .dataset import load_gold_dataset
from .evaluator import Evaluator

logger = logging.getLogger(__name__)


def run_evaluation(
    dataset_path: Path,
    output_dir: Path = Path("data/eval/reports"),
    top_k: int = 10,
    boost_entity_names: bool = True,
    config_path: Optional[Path] = None
) -> Path:
    """
    Run evaluation on gold dataset.
    
    Args:
        dataset_path: Path to gold_qa.jsonl
        output_dir: Output directory for reports
        top_k: Number of results to retrieve
        boost_entity_names: Whether to use entity boosting
        config_path: Path to config.yaml
    
    Returns:
        Path to evaluation report
    """
    # Load config
    config = None
    if config_path or True:
        try:
            config = load_config(config_path)
        except Exception as e:
            logger.warning(f"Failed to load config: {e}")
    
    # Initialize pipeline
    pipeline = KnowledgeLayerPipeline(config=config)
    
    # Load gold dataset
    gold_pairs = load_gold_dataset(dataset_path)
    
    # Run evaluation
    evaluator = Evaluator(pipeline)
    metrics = evaluator.evaluate(
        gold_pairs,
        top_k=top_k,
        boost_entity_names=boost_entity_names
    )
    
    # Save report
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"eval_report_{timestamp}.json"
    
    report = {
        'timestamp': timestamp,
        'dataset_path': str(dataset_path),
        'top_k': top_k,
        'boost_entity_names': boost_entity_names,
        'metrics': metrics.to_dict()
    }
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Evaluation report saved to {report_path}")
    return report_path


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Run RAG pipeline evaluation")
    parser.add_argument(
        '--dataset',
        type=Path,
        required=True,
        help='Path to gold_qa.jsonl file'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path("data/eval/reports"),
        help='Output directory for reports'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=10,
        help='Number of results to retrieve'
    )
    parser.add_argument(
        '--no-entity-boost',
        action='store_true',
        help='Disable entity name boosting'
    )
    parser.add_argument(
        '--config',
        type=Path,
        help='Path to config.yaml'
    )
    
    args = parser.parse_args()
    
    try:
        report_path = run_evaluation(
            dataset_path=args.dataset,
            output_dir=args.output_dir,
            top_k=args.top_k,
            boost_entity_names=not args.no_entity_boost,
            config_path=args.config
        )
        print(f"✓ Evaluation complete: {report_path}")
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
