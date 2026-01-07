"""
Index validation script.

Validates index integrity, metadata consistency, and configuration matching.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.search.vector_store import VectorStore
from src.config import load_config

logger = logging.getLogger(__name__)


def validate_index(
    index_path: Path,
    config_path: Optional[Path] = None,
    strict: bool = False
) -> Dict[str, Any]:
    """
    Validate an index file.
    
    Args:
        index_path: Path to index file (.pkl)
        config_path: Path to config.yaml for comparison
        strict: If True, fail on warnings
    
    Returns:
        Validation report dictionary
    """
    report = {
        'index_path': str(index_path),
        'valid': True,
        'errors': [],
        'warnings': [],
        'metadata': None,
        'integrity_checks': {}
    }
    
    # Check file exists
    if not index_path.exists():
        report['valid'] = False
        report['errors'].append(f"Index file not found: {index_path}")
        return report
    
    # Load index
    try:
        store = VectorStore()
        store.load(index_path)
        report['integrity_checks']['load_success'] = True
        report['integrity_checks']['vector_count'] = len(store.metadata)
        report['integrity_checks']['embedding_dim'] = store.embedding_dim
    except Exception as e:
        report['valid'] = False
        report['errors'].append(f"Failed to load index: {e}")
        return report
    
    # Validate metadata if available
    if store.index_metadata:
        report['metadata'] = store.index_metadata.to_dict()
        
        # Check dimension consistency
        if store.index_metadata.metadata:
            expected_dim = store.index_metadata.metadata.embedding_dim
            actual_dim = store.embedding_dim
            if expected_dim != actual_dim:
                report['warnings'].append(
                    f"Dimension mismatch: metadata says {expected_dim}, actual is {actual_dim}"
                )
                if strict:
                    report['valid'] = False
        
        # Check config hash if config provided
        if config_path and config_path.exists():
            try:
                config = load_config(config_path)
                from src.embeddings.metadata import generate_config_hash
                
                chunking_config = config.chunking if config else {}
                embedding_config = config.embedding if config else {}
                
                current_hash = generate_config_hash(
                    chunking_strategy=chunking_config.get('strategy', 'hungarian-aware'),
                    chunk_size=chunking_config.get('chunk_size', 400),
                    overlap=chunking_config.get('overlap', 60),
                    embedding_model=embedding_config.get('model', 'baseline')
                )
                
                stored_hash = store.index_metadata.chunker_config_hash
                if stored_hash and stored_hash != current_hash:
                    report['warnings'].append(
                        f"Config hash mismatch: index was built with different config. "
                        f"Stored: {stored_hash[:8]}..., Current: {current_hash[:8]}..."
                    )
                    if strict:
                        report['valid'] = False
            except Exception as e:
                report['warnings'].append(f"Could not validate config hash: {e}")
        
        # Check embedding fingerprint
        if store.index_metadata.embedding_fingerprint:
            report['integrity_checks']['has_fingerprint'] = True
        else:
            report['warnings'].append("No embedding fingerprint found")
    else:
        report['warnings'].append("No index metadata found (index may be from older version)")
    
    # Validate vector consistency
    if store.vectors is not None:
        if len(store.vectors) != len(store.metadata):
            report['errors'].append(
                f"Vector count mismatch: {len(store.vectors)} vectors vs {len(store.metadata)} metadata entries"
            )
            report['valid'] = False
        
        # Check dimension consistency
        if len(store.vectors) > 0:
            actual_dim = store.vectors.shape[1] if len(store.vectors.shape) > 1 else store.vectors.shape[0]
            if actual_dim != store.embedding_dim:
                report['errors'].append(
                    f"Dimension mismatch: vectors have {actual_dim} dims, store says {store.embedding_dim}"
                )
                report['valid'] = False
    
    return report


def main():
    """CLI entry point for validate_index."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate vector store index")
    parser.add_argument(
        'index_path',
        type=Path,
        help='Path to index file (.pkl)'
    )
    parser.add_argument(
        '--config',
        type=Path,
        help='Path to config.yaml for comparison'
    )
    parser.add_argument(
        '--strict',
        action='store_true',
        help='Fail on warnings'
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output as JSON'
    )
    
    args = parser.parse_args()
    
    report = validate_index(
        args.index_path,
        config_path=args.config,
        strict=args.strict
    )
    
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"\n{'='*60}")
        print(f"Index Validation Report: {args.index_path}")
        print(f"{'='*60}\n")
        
        if report['valid']:
            print("✓ Index is VALID")
        else:
            print("✗ Index is INVALID")
        
        if report['errors']:
            print("\nErrors:")
            for error in report['errors']:
                print(f"  ✗ {error}")
        
        if report['warnings']:
            print("\nWarnings:")
            for warning in report['warnings']:
                print(f"  ⚠ {warning}")
        
        if report['integrity_checks']:
            print("\nIntegrity Checks:")
            for check, value in report['integrity_checks'].items():
                print(f"  • {check}: {value}")
        
        if report['metadata']:
            print("\nMetadata:")
            metadata = report['metadata']
            if 'metadata' in metadata and metadata['metadata']:
                meta = metadata['metadata']
                print(f"  • Model: {meta.get('model_name', 'unknown')}")
                print(f"  • Version: {meta.get('model_version', 'unknown')}")
                print(f"  • Dimension: {meta.get('embedding_dim', 'unknown')}")
                print(f"  • Config Hash: {meta.get('config_hash', 'N/A')[:16] if meta.get('config_hash') else 'N/A'}...")
            print(f"  • Created: {metadata.get('created_at', 'unknown')}")
            print(f"  • Chunk Count: {metadata.get('chunk_count', 'unknown')}")
    
    sys.exit(0 if report['valid'] else 1)


if __name__ == "__main__":
    main()
