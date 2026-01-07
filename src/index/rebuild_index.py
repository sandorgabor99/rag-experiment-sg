"""
Index rebuild script with soft-delete archive strategy.

Rebuilds index with current configuration and archives old versions.
"""

import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.index.build_index import build_index
from src.config import load_config

logger = logging.getLogger(__name__)


def archive_index(
    index_path: Path,
    archive_dir: Path,
    keep_last_n: int = 5
) -> None:
    """
    Archive an index file (soft-delete).
    
    Args:
        index_path: Path to index file to archive
        archive_dir: Archive directory
        keep_last_n: Keep only last N archived versions
    """
    archive_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate archive name with timestamp
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    archive_name = f"{index_path.stem}_archived_{timestamp}{index_path.suffix}"
    archive_path = archive_dir / archive_name
    
    # Move index file
    if index_path.exists():
        shutil.move(str(index_path), str(archive_path))
        logger.info(f"Archived index: {archive_path}")
    
    # Move metadata file if exists
    metadata_path = index_path.with_suffix('.metadata.json')
    if metadata_path.exists():
        archive_metadata = archive_path.with_suffix('.metadata.json')
        shutil.move(str(metadata_path), str(archive_metadata))
        logger.info(f"Archived metadata: {archive_metadata}")
    
    # Clean up old archives (keep only last N)
    archived_files = sorted(archive_dir.glob(f"{index_path.stem}_archived_*.pkl"), reverse=True)
    if len(archived_files) > keep_last_n:
        for old_file in archived_files[keep_last_n:]:
            old_file.unlink()
            # Also remove metadata if exists
            old_metadata = old_file.with_suffix('.metadata.json')
            if old_metadata.exists():
                old_metadata.unlink()
            logger.info(f"Removed old archive: {old_file}")


def rebuild_index(
    input_file: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    archive_old: bool = True,
    keep_last_n: int = 5,
    config_path: Optional[Path] = None
) -> Path:
    """
    Rebuild index with current configuration.
    
    Args:
        input_file: Path to embedded chunks file
        output_dir: Output directory for indexes
        archive_old: Whether to archive old indexes
        keep_last_n: Number of archived versions to keep
        config_path: Path to config.yaml
    
    Returns:
        Path to new index file
    """
    # Load config
    config = None
    if config_path or True:
        try:
            config = load_config(config_path)
        except Exception as e:
            logger.warning(f"Failed to load config: {e}")
    
    # Determine output directory
    if output_dir is None:
        output_dir = Path("data/processed/indexes")
    output_dir = Path(output_dir)
    archive_dir = output_dir / "archive"
    
    # Archive existing latest index if it exists
    if archive_old:
        latest_path = output_dir / "vector_store_latest.pkl"
        if not latest_path.exists():
            # Try to find latest from pointer file
            latest_txt = output_dir / "vector_store_latest.txt"
            if latest_txt.exists():
                with open(latest_txt, 'r') as f:
                    latest_name = f.read().strip()
                latest_path = output_dir / latest_name
        
        if latest_path.exists() and latest_path.is_file():
            logger.info(f"Archiving existing index: {latest_path}")
            archive_index(latest_path, archive_dir, keep_last_n=keep_last_n)
    
    # Build new index
    logger.info("Building new index with current configuration...")
    new_index_path = build_index(
        input_file=input_file,
        output_dir=output_dir,
        create_latest_symlink=True,
        config_path=config_path
    )
    
    logger.info(f"✓ Index rebuilt: {new_index_path}")
    return new_index_path


def main():
    """CLI entry point for rebuild_index."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Rebuild vector store index with archiving")
    parser.add_argument(
        '--input-file',
        type=Path,
        help='Path to embedded chunks file (defaults to config)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        help='Output directory for indexes (default: data/processed/indexes)'
    )
    parser.add_argument(
        '--no-archive',
        action='store_true',
        help='Do not archive old indexes'
    )
    parser.add_argument(
        '--keep-last-n',
        type=int,
        default=5,
        help='Number of archived versions to keep (default: 5)'
    )
    parser.add_argument(
        '--config',
        type=Path,
        help='Path to config.yaml'
    )
    
    args = parser.parse_args()
    
    try:
        index_path = rebuild_index(
            input_file=args.input_file,
            output_dir=args.output_dir,
            archive_old=not args.no_archive,
            keep_last_n=args.keep_last_n,
            config_path=args.config
        )
        print(f"✓ Index rebuilt successfully: {index_path}")
    except Exception as e:
        logger.error(f"Failed to rebuild index: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
