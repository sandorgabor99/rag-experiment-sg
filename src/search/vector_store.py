"""
Vector Store module for persistent, searchable vector storage.

This module implements a vector database with:
- Cosine and dot-product similarity search
- Top-k retrieval
- Metadata filtering
- Batch insert
- Persistence (save/load from disk)
- Reindexable
- Low latency (<50ms) optimized

This is the "memory" component of the Knowledge Layer system.
"""

import json
import logging
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable
import numpy as np

try:
    from src.embeddings.metadata import IndexHeader, EmbeddingMetadata
    METADATA_AVAILABLE = True
except ImportError:
    METADATA_AVAILABLE = False
    IndexHeader = None
    EmbeddingMetadata = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class VectorStore:
    """
    Persistent vector store with metadata filtering and fast similarity search.
    
    Features:
    - Cosine and dot-product similarity search
    - Top-k retrieval
    - Metadata filtering
    - Batch insert
    - Persistence (save/load)
    - Reindexable
    - Optimized for low latency
    """
    
    def __init__(
        self,
        embedding_dim: Optional[int] = None,
        normalize_embeddings: bool = True,
        index_metadata: Optional[Any] = None,
        persist_path: Optional[Path] = None
    ):
        """
        Initialize vector store.
        
        Args:
            embedding_dim: Expected embedding dimension (auto-detected if None)
            normalize_embeddings: Whether to normalize embeddings for cosine similarity
            persist_path: Path to save/load the store (optional)
        """
        self.embedding_dim = embedding_dim
        self.normalize_embeddings = normalize_embeddings
        self.persist_path = persist_path
        self.index_metadata = index_metadata  # IndexHeader if available
        
        # Storage: vectors and metadata
        self.vectors: np.ndarray = None  # Shape: (n_vectors, embedding_dim)
        self.metadata: List[Dict[str, Any]] = []  # List of metadata dicts
        self.ids: List[str] = []  # List of unique IDs
        
        # Index for fast ID lookup
        self._id_to_index: Dict[str, int] = {}
        
        # Cache for normalized vectors (if normalization enabled)
        self._normalized_vectors: Optional[np.ndarray] = None
        self._normalized_cache_valid = False
        
        logger.info(f"Initialized VectorStore (normalize={normalize_embeddings})")
    
    def add(
        self,
        vector: np.ndarray,
        metadata: Dict[str, Any],
        id: Optional[str] = None
    ) -> str:
        """
        Add a single vector with metadata.
        
        Args:
            vector: Embedding vector (1D array)
            metadata: Metadata dictionary (must include 'id' or id must be provided)
            id: Optional unique ID (uses metadata['id'] if not provided)
        
        Returns:
            The ID of the added vector
        """
        # Validate vector
        vector = np.asarray(vector, dtype=np.float32)
        if vector.ndim != 1:
            raise ValueError(f"Vector must be 1D, got shape {vector.shape}")
        
        # Set embedding dimension if first vector
        if self.embedding_dim is None:
            self.embedding_dim = vector.shape[0]
            logger.info(f"Set embedding dimension to {self.embedding_dim}")
        elif vector.shape[0] != self.embedding_dim:
            raise ValueError(
                f"Vector dimension {vector.shape[0]} doesn't match "
                f"expected {self.embedding_dim}"
            )
        
        # Get or generate ID
        if id is None:
            id = metadata.get('id')
            if id is None:
                raise ValueError("Either 'id' parameter or metadata['id'] must be provided")
        
        # Check for duplicate ID
        if id in self._id_to_index:
            raise ValueError(f"ID '{id}' already exists in store")
        
        # Add to storage
        if self.vectors is None:
            self.vectors = vector.reshape(1, -1)
        else:
            self.vectors = np.vstack([self.vectors, vector.reshape(1, -1)])
        
        self.metadata.append(metadata.copy())
        self.ids.append(id)
        self._id_to_index[id] = len(self.ids) - 1
        
        # Invalidate normalized cache
        self._normalized_cache_valid = False
        
        return id
    
    def add_batch(
        self,
        vectors: np.ndarray,
        metadata_list: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """
        Add multiple vectors in batch (more efficient than individual adds).
        
        Args:
            vectors: Array of vectors (shape: n_vectors, embedding_dim)
            metadata_list: List of metadata dictionaries
            ids: Optional list of IDs (uses metadata['id'] if not provided)
        
        Returns:
            List of IDs for added vectors
        """
        vectors = np.asarray(vectors, dtype=np.float32)
        
        if vectors.ndim != 2:
            raise ValueError(f"Vectors must be 2D array, got shape {vectors.shape}")
        
        n_vectors = vectors.shape[0]
        if len(metadata_list) != n_vectors:
            raise ValueError(
                f"Number of vectors ({n_vectors}) doesn't match "
                f"number of metadata entries ({len(metadata_list)})"
            )
        
        # Set embedding dimension if first batch
        if self.embedding_dim is None:
            self.embedding_dim = vectors.shape[1]
            logger.info(f"Set embedding dimension to {self.embedding_dim}")
        elif vectors.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Vector dimension {vectors.shape[1]} doesn't match "
                f"expected {self.embedding_dim}"
            )
        
        # Generate or validate IDs
        if ids is None:
            ids = [m.get('id') for m in metadata_list]
            if any(id is None for id in ids):
                raise ValueError("All metadata entries must have 'id' field")
        
        # Check for duplicate IDs
        duplicate_ids = set(ids) & set(self.ids)
        if duplicate_ids:
            raise ValueError(f"Duplicate IDs found: {duplicate_ids}")
        
        # Add to storage
        if self.vectors is None:
            self.vectors = vectors
        else:
            self.vectors = np.vstack([self.vectors, vectors])
        
        self.metadata.extend([m.copy() for m in metadata_list])
        self.ids.extend(ids)
        
        # Update ID index
        start_idx = len(self.ids) - n_vectors
        for i, id in enumerate(ids):
            self._id_to_index[id] = start_idx + i
        
        # Invalidate normalized cache
        self._normalized_cache_valid = False
        
        logger.info(f"Added batch of {n_vectors} vectors")
        return ids
    
    def _get_normalized_vectors(self) -> np.ndarray:
        """Get normalized vectors (cached for performance)."""
        if not self.normalize_embeddings:
            return self.vectors
        
        if not self._normalized_cache_valid or self._normalized_vectors is None:
            # Normalize vectors: divide by L2 norm
            norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
            self._normalized_vectors = self.vectors / norms
            self._normalized_cache_valid = True
        
        return self._normalized_vectors
    
    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        similarity_metric: str = 'cosine',
        metadata_filter: Optional[Callable[[Dict[str, Any]], bool]] = None
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Search for similar vectors.
        
        Args:
            query_vector: Query embedding vector (1D array)
            top_k: Number of results to return
            similarity_metric: 'cosine' or 'dot_product'
            metadata_filter: Optional function to filter metadata (returns True to include)
        
        Returns:
            List of (metadata, similarity_score) tuples, sorted by similarity (descending)
        """
        if self.vectors is None or len(self.vectors) == 0:
            return []
        
        start_time = time.time()
        
        query_vector = np.asarray(query_vector, dtype=np.float32)
        if query_vector.ndim != 1:
            raise ValueError(f"Query vector must be 1D, got shape {query_vector.shape}")
        if query_vector.shape[0] != self.embedding_dim:
            raise ValueError(
                f"Query dimension {query_vector.shape[0]} doesn't match "
                f"store dimension {self.embedding_dim}"
            )
        
        # Normalize query if using cosine similarity
        if similarity_metric == 'cosine':
            query_norm = np.linalg.norm(query_vector)
            if query_norm == 0:
                query_vector = query_vector
            else:
                query_vector = query_vector / query_norm
            vectors = self._get_normalized_vectors()
        elif similarity_metric == 'dot_product':
            vectors = self.vectors
        else:
            raise ValueError(f"Unknown similarity metric: {similarity_metric}")
        
        # Compute similarities (vectorized for speed)
        similarities = np.dot(vectors, query_vector)
        
        # Apply metadata filter if provided
        if metadata_filter is not None:
            # Get indices that pass filter
            valid_indices = [
                i for i in range(len(self.metadata))
                if metadata_filter(self.metadata[i])
            ]
            if not valid_indices:
                return []
            
            # Filter similarities and metadata
            filtered_similarities = similarities[valid_indices]
            filtered_metadata = [self.metadata[i] for i in valid_indices]
            
            # Get top-k from filtered results
            top_indices = np.argsort(filtered_similarities)[::-1][:top_k]
            results = [
                (filtered_metadata[i], float(filtered_similarities[i]))
                for i in top_indices
            ]
        else:
            # Get top-k indices
            top_indices = np.argsort(similarities)[::-1][:top_k]
            results = [
                (self.metadata[i], float(similarities[i]))
                for i in top_indices
            ]
        
        elapsed_ms = (time.time() - start_time) * 1000
        if elapsed_ms > 50:
            logger.warning(f"Search took {elapsed_ms:.2f}ms (target: <50ms)")
        
        return results
    
    def get_by_id(self, id: str) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        """
        Get vector and metadata by ID.
        
        Args:
            id: Vector ID
        
        Returns:
            (vector, metadata) tuple or None if not found
        """
        if id not in self._id_to_index:
            return None
        
        idx = self._id_to_index[id]
        return (self.vectors[idx], self.metadata[idx])
    
    def delete(self, id: str) -> bool:
        """
        Delete a vector by ID.
        
        Args:
            id: Vector ID
        
        Returns:
            True if deleted, False if not found
        """
        if id not in self._id_to_index:
            return False
        
        idx = self._id_to_index[id]
        
        # Remove from arrays
        self.vectors = np.delete(self.vectors, idx, axis=0)
        del self.metadata[idx]
        del self.ids[idx]
        
        # Rebuild ID index
        self._id_to_index = {id: i for i, id in enumerate(self.ids)}
        
        # Invalidate normalized cache
        self._normalized_cache_valid = False
        
        return True
    
    def reindex(self) -> None:
        """
        Rebuild internal indices (useful after manual modifications).
        """
        self._id_to_index = {id: i for i, id in enumerate(self.ids)}
        self._normalized_cache_valid = False
        logger.info("Reindexed vector store")
    
    def save(self, path: Optional[Path] = None, save_metadata_json: bool = True) -> None:
        """
        Save vector store to disk.
        
        Args:
            path: Save path (uses self.persist_path if None)
            save_metadata_json: Whether to save metadata as separate JSON file
        """
        save_path = path or self.persist_path
        if save_path is None:
            raise ValueError("No save path provided")
        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Update index metadata if available
        if METADATA_AVAILABLE and self.index_metadata:
            self.index_metadata.chunk_count = len(self.metadata)
            if self.vectors is not None and len(self.vectors) > 0:
                # Generate embedding fingerprint
                from src.embeddings.metadata import generate_embedding_fingerprint
                embeddings_list = self.vectors.tolist()
                self.index_metadata.embedding_fingerprint = generate_embedding_fingerprint(embeddings_list)
        
        # Save as pickle for efficiency (vectors + metadata)
        data = {
            'vectors': self.vectors,
            'metadata': self.metadata,
            'ids': self.ids,
            'embedding_dim': self.embedding_dim,
            'normalize_embeddings': self.normalize_embeddings,
            'id_to_index': self._id_to_index,
            'index_metadata': self.index_metadata.to_dict() if (METADATA_AVAILABLE and self.index_metadata) else None
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)
        
        # Save metadata JSON separately if requested
        if save_metadata_json and METADATA_AVAILABLE and self.index_metadata:
            metadata_path = save_path.with_suffix('.metadata.json')
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(self.index_metadata.to_dict(), f, indent=2)
            logger.info(f"Saved index metadata to {metadata_path}")
        
        logger.info(f"Saved vector store to {save_path} ({len(self.ids)} vectors)")
    
    def load(self, path: Optional[Path] = None) -> None:
        """
        Load vector store from disk.
        
        Args:
            path: Load path (uses self.persist_path if None)
        """
        load_path = path or self.persist_path
        if load_path is None:
            raise ValueError("No load path provided")
        
        load_path = Path(load_path)
        if not load_path.exists():
            raise FileNotFoundError(f"Vector store file not found: {load_path}")
        
        with open(load_path, 'rb') as f:
            data = pickle.load(f)
        
        self.vectors = data['vectors']
        self.metadata = data['metadata']
        self.ids = data['ids']
        self.embedding_dim = data['embedding_dim']
        self.normalize_embeddings = data.get('normalize_embeddings', True)
        self._id_to_index = data.get('id_to_index', {})
        
        # Load index metadata if available
        if METADATA_AVAILABLE and 'index_metadata' in data and data['index_metadata']:
            try:
                self.index_metadata = IndexHeader.from_dict(data['index_metadata'])
                logger.info(f"Loaded index metadata: version={self.index_metadata.version}, "
                          f"model={self.index_metadata.metadata.model_name if self.index_metadata.metadata else 'unknown'}")
            except Exception as e:
                logger.warning(f"Failed to load index metadata: {e}")
                self.index_metadata = None
        else:
            # Try to load from separate JSON file
            metadata_path = load_path.with_suffix('.metadata.json')
            if metadata_path.exists() and METADATA_AVAILABLE:
                try:
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        metadata_dict = json.load(f)
                    self.index_metadata = IndexHeader.from_dict(metadata_dict)
                    logger.info(f"Loaded index metadata from {metadata_path}")
                except Exception as e:
                    logger.warning(f"Failed to load metadata JSON: {e}")
                    self.index_metadata = None
        
        # Rebuild index if missing
        if not self._id_to_index:
            self.reindex()
        
        # Invalidate normalized cache
        self._normalized_cache_valid = False
        
        logger.info(f"Loaded vector store from {load_path} ({len(self.ids)} vectors)")
    
    def size(self) -> int:
        """Get number of vectors in store."""
        return len(self.ids) if self.ids else 0
    
    def clear(self) -> None:
        """Clear all vectors and metadata."""
        self.vectors = None
        self.metadata = []
        self.ids = []
        self._id_to_index = {}
        self._normalized_cache_valid = False
        logger.info("Cleared vector store")


def create_metadata_filter(
    source: Optional[str] = None,
    chunk_index: Optional[int] = None,
    min_chunk_index: Optional[int] = None,
    max_chunk_index: Optional[int] = None,
    custom_filter: Optional[Callable[[Dict[str, Any]], bool]] = None
) -> Callable[[Dict[str, Any]], bool]:
    """
    Create a metadata filter function.
    
    Args:
        source: Filter by exact source match
        chunk_index: Filter by exact chunk_index
        min_chunk_index: Filter by minimum chunk_index (inclusive)
        max_chunk_index: Filter by maximum chunk_index (inclusive)
        custom_filter: Custom filter function
    
    Returns:
        Filter function that returns True if metadata matches criteria
    """
    def filter_func(metadata: Dict[str, Any]) -> bool:
        # Source filter
        if source is not None:
            metadata_source = metadata.get('source')
            if metadata_source != source:
                logger.debug(f"Filtered out: source '{metadata_source}' != '{source}'")
                return False
            logger.debug(f"Matched source: '{metadata_source}' == '{source}'")
        
        # Chunk index filters
        chunk_idx = metadata.get('chunk_index')
        if chunk_index is not None:
            if chunk_idx != chunk_index:
                return False
        if min_chunk_index is not None:
            if chunk_idx is None or chunk_idx < min_chunk_index:
                return False
        if max_chunk_index is not None:
            if chunk_idx is None or chunk_idx > max_chunk_index:
                return False
        
        # Custom filter
        if custom_filter is not None:
            if not custom_filter(metadata):
                return False
        
        return True
    
    return filter_func


def load_vector_store_from_embeddings(
    embeddings_file: Path,
    persist_path: Optional[Path] = None,
    normalize_embeddings: bool = True
) -> VectorStore:
    """
    Load a VectorStore from a JSONL file with embeddings.
    
    Args:
        embeddings_file: Path to JSONL file with Knowledge Layer entries and embeddings
        persist_path: Optional path to save the vector store
        normalize_embeddings: Whether to normalize embeddings
    
    Returns:
        VectorStore instance
    """
    from knowledge_layer import read_knowledge_entries
    
    logger.info(f"Loading embeddings from {embeddings_file}")
    entries = read_knowledge_entries(embeddings_file, validate=False)
    
    if not entries:
        raise ValueError("No entries found in embeddings file")
    
    if 'embedding' not in entries[0]:
        raise ValueError("Entries do not contain embeddings")
    
    # Extract vectors and metadata
    vectors = np.array([entry['embedding'] for entry in entries], dtype=np.float32)
    metadata_list = [
        {k: v for k, v in entry.items() if k != 'embedding'}
        for entry in entries
    ]
    ids = [entry['id'] for entry in entries]
    
    # Create and populate store
    store = VectorStore(
        embedding_dim=vectors.shape[1],
        normalize_embeddings=normalize_embeddings,
        persist_path=persist_path,
        index_metadata=index_metadata
    )
    store.add_batch(vectors, metadata_list, ids)
    
    logger.info(f"Created VectorStore with {store.size()} vectors")
    return store
