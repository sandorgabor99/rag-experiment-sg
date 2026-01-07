"""
Knowledge Layer module for managing structured knowledge chunks.

This module provides functionality to create, validate, and manage Knowledge Layer
entries with the following attributes:
- id: Unique identifier for the chunk
- source: Original file name
- chunk_index: Index of the chunk within the source file
- text: The chunk text content
"""

import json
from pathlib import Path
from typing import Dict, List, Optional


def create_knowledge_entry(
    source: str,
    chunk_index: int,
    text: str,
    id_prefix: Optional[str] = None
) -> Dict[str, any]:
    """
    Create a Knowledge Layer entry with required attributes.
    
    Args:
        source: Original file name
        chunk_index: Index of the chunk within the source file (0-based)
        text: The chunk text content
        id_prefix: Optional prefix for the ID. If None, uses source filename without extension
    
    Returns:
        Dictionary containing Knowledge Layer entry with id, source, chunk_index, and text
    """
    # Generate ID: use id_prefix if provided, otherwise use source filename stem
    if id_prefix is None:
        # Extract filename without extension from source
        source_stem = Path(source).stem
        chunk_id = f"{source_stem}_{chunk_index}"
    else:
        chunk_id = f"{id_prefix}_{chunk_index}"
    
    return {
        "id": chunk_id,
        "source": source,
        "chunk_index": chunk_index,
        "text": text
        # Optional: entities can be added later via entity_extractor.py
        # "entities": {
        #     "people": [],
        #     "locations": [],
        #     "organizations": [],
        #     "dates": [],
        #     "events": []
        # }
    }


def validate_knowledge_entry(entry: Dict[str, any]) -> bool:
    """
    Validate that a Knowledge Layer entry has all required attributes.
    
    Args:
        entry: Dictionary to validate
    
    Returns:
        True if valid, False otherwise
    """
    required_attributes = {"id", "source", "chunk_index", "text"}
    
    if not isinstance(entry, dict):
        return False
    
    if not required_attributes.issubset(entry.keys()):
        return False
    
    # Validate types
    if not isinstance(entry["id"], str):
        return False
    if not isinstance(entry["source"], str):
        return False
    if not isinstance(entry["chunk_index"], int):
        return False
    if not isinstance(entry["text"], str):
        return False
    
    # Validate optional entities if present
    if "entities" in entry:
        if not isinstance(entry["entities"], dict):
            return False
        expected_entity_types = {"people", "locations", "organizations", "dates", "events"}
        for entity_type in expected_entity_types:
            if entity_type in entry["entities"]:
                if not isinstance(entry["entities"][entity_type], list):
                    return False
    
    return True


def write_knowledge_entries(
    entries: List[Dict[str, any]],
    output_path: Path,
    validate: bool = True
) -> None:
    """
    Write Knowledge Layer entries to a JSONL file.
    
    Args:
        entries: List of Knowledge Layer entry dictionaries
        output_path: Path to output JSONL file
        validate: Whether to validate entries before writing
    
    Raises:
        ValueError: If validation fails for any entry
        IOError: If file writing fails
    """
    if validate:
        for i, entry in enumerate(entries):
            if not validate_knowledge_entry(entry):
                raise ValueError(f"Invalid Knowledge Layer entry at index {i}: {entry}")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def read_knowledge_entries(
    input_path: Path,
    validate: bool = True
) -> List[Dict[str, any]]:
    """
    Read Knowledge Layer entries from a JSONL file.
    
    Args:
        input_path: Path to input JSONL file
        validate: Whether to validate entries after reading
    
    Returns:
        List of Knowledge Layer entry dictionaries
    
    Raises:
        FileNotFoundError: If input file doesn't exist
        ValueError: If validation fails for any entry
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    entries = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                if validate and not validate_knowledge_entry(entry):
                    raise ValueError(
                        f"Invalid Knowledge Layer entry at line {line_num}: {entry}"
                    )
                entries.append(entry)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at line {line_num}: {e}")
    
    return entries
