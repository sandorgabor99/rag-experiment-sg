"""
Index building script with versioning support.

Builds vector store indexes with timestamped names and metadata tracking.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.pipeline.orchestrator import KnowledgeLayerPipeline
from src.config import load_config

logger = logging.getLogger(__name__)


def build_index(
    input_file: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    create_latest_symlink: bool = True,
    config_path: Optional[Path] = None
) -> Path:
    """
    Build a versioned index from embedded chunks.
    
    Args:
        input_file: Path to embedded chunks file (defaults to config paths)
        output_dir: Output directory for indexes (defaults to data/processed/indexes)
        create_latest_symlink: Whether to create 'latest' symlink
        config_path: Path to config.yaml
    
    Returns:
        Path to created index file
    """
    # Load config
    config = None
    if config_path or True:  # Always try to load config
        try:
            config = load_config(config_path)
        except Exception as e:
            logger.warning(f"Failed to load config: {e}")
    
    # Initialize pipeline
    pipeline = KnowledgeLayerPipeline(config=config)
    
    # Determine input file
    if input_file is None:
        input_file = pipeline.embedded_file
        if not input_file.exists():
            raise FileNotFoundError(
                f"Embedded file not found: {input_file}. Run embedding step first."
            )
    
    # Determine output directory
    if output_dir is None:
        output_dir = Path("data/processed/indexes")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamped index name
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    index_name = f"vector_store_v{timestamp}.pkl"
    index_path = output_dir / index_name
    
    logger.info(f"Building index: {index_path}")
    
    # Build index using VectorStore creation
    from src.search.vector_store import load_vector_store_from_embeddings
    from src.knowledge_layer import read_knowledge_entries
    
    # Load embedded entries
    entries = read_knowledge_entries(input_file)
    logger.info(f"Loaded {len(entries)} embedded entries")
    
    # Create VectorStore with metadata
    try:
        from src.embeddings.metadata import IndexHeader, EmbeddingMetadata, generate_embedding_fingerprint
        
        # Extract metadata from first entry if available
        embedding_metadata = None
        if entries and 'embedding' in entries[0]:
            # Try to get metadata from pipeline or generate it
            embedding_config = config.embedding if config else {}
            chunking_config = config.chunking if config else {}
            
            from src.embeddings.metadata import get_model_version, generate_config_hash
            
            model_name = embedding_config.get('model', 'baseline')
            embedding_dim = len(entries[0]['embedding']) if entries else 0
            
            embedding_metadata = EmbeddingMetadata(
                model_name=model_name,
                model_version=get_model_version(model_name),
                embedding_dim=embedding_dim
            )
            
            # Generate config hash
            if chunking_config:
                config_hash = generate_config_hash(
                    chunking_strategy=chunking_config.get('strategy', 'hungarian-aware'),
                    chunk_size=chunking_config.get('chunk_size', 400),
                    overlap=chunking_config.get('overlap', 60),
                    embedding_model=model_name
                )
                embedding_metadata.config_hash = config_hash
        
        # Create index header
        index_header = IndexHeader(
            metadata=embedding_metadata,
            chunk_count=len(entries)
        )
        
        # Generate embedding fingerprint
        if entries:
            embeddings_list = [e.get('embedding', []) for e in entries if 'embedding' in e]
            if embeddings_list:
                index_header.embedding_fingerprint = generate_embedding_fingerprint(embeddings_list)
        
    except ImportError:
        logger.warning("Metadata module not available, building index without metadata")
        index_header = None
    
    # Create VectorStore directly (load_vector_store_from_embeddings doesn't support index_metadata parameter)
    from src.search.vector_store import VectorStore
    import numpy as np
    
    # Extract vectors and metadata
    vectors = np.array([entry['embedding'] for entry in entries], dtype=np.float32)
    metadata_list = [
        {k: v for k, v in entry.items() if k != 'embedding'}
        for entry in entries
    ]
    ids = [entry['id'] for entry in entries]
    
    # Create VectorStore with metadata
    store = VectorStore(
        embedding_dim=vectors.shape[1],
        normalize_embeddings=True,
        persist_path=index_path,
        index_metadata=index_header
    )
    
    # Add all vectors
    store.add_batch(vectors, metadata_list, ids)
    
    # Save with metadata
    store.save(index_path, save_metadata_json=True)
    
    logger.info(f"✓ Index built: {index_path}")
    
    # Create latest symlink (if supported on platform)
    if create_latest_symlink:
        latest_path = output_dir / "vector_store_latest.pkl"
        try:
            if latest_path.exists() or latest_path.is_symlink():
                latest_path.unlink()
            latest_path.symlink_to(index_path.name)
            logger.info(f"✓ Created symlink: {latest_path} -> {index_path.name}")
        except (OSError, NotImplementedError) as e:
            # Symlinks not supported on Windows without admin or in some environments
            logger.warning(f"Could not create symlink (not supported on this platform): {e}")
            # Create a text file with the latest index name instead
            latest_txt = output_dir / "vector_store_latest.txt"
            with open(latest_txt, 'w') as f:
                f.write(index_path.name)
            logger.info(f"✓ Created latest pointer file: {latest_txt}")
    
    return index_path


def main():
    """CLI entry point for build_index."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Build versioned vector store index")
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
        '--no-symlink',
        action='store_true',
        help='Do not create latest symlink'
    )
    parser.add_argument(
        '--config',
        type=Path,
        help='Path to config.yaml'
    )
    
    args = parser.parse_args()
    
    try:
        index_path = build_index(
            input_file=args.input_file,
            output_dir=args.output_dir,
            create_latest_symlink=not args.no_symlink,
            config_path=args.config
        )
        print(f"✓ Index built successfully: {index_path}")
    except Exception as e:
        logger.error(f"Failed to build index: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
