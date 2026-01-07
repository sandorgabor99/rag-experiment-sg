"""
Script to update imports in reorganized files.

This script updates all relative imports to use the new package structure.
"""

import re
from pathlib import Path

# Mapping of old imports to new imports
IMPORT_MAPPINGS = {
    r'^from knowledge_layer import': 'from src.knowledge_layer import',
    r'^import knowledge_layer': 'import src.knowledge_layer',
    r'^from chunk import': 'from src.processing.chunking import',
    r'^import chunk': 'import src.processing.chunking',
    r'^from embed import': 'from src.embeddings.generator import',
    r'^import embed': 'import src.embeddings.generator',
    r'^from search import': 'from src.search.searcher import',
    r'^import search': 'import src.search.searcher',
    r'^from vector_store import': 'from src.search.vector_store import',
    r'^import vector_store': 'import src.search.vector_store',
    r'^from context_builder import': 'from src.search.context_builder import',
    r'^import context_builder': 'import src.search.context_builder',
    r'^from agentic_refiner import': 'from src.llm.refiner import',
    r'^import agentic_refiner': 'import src.llm.refiner',
    r'^from qa import': 'from src.llm.qa import',
    r'^import qa': 'import src.llm.qa',
    r'^from entity_extractor import': 'from src.entities.extractor import',
    r'^import entity_extractor': 'import src.entities.extractor',
}

def update_file_imports(file_path: Path):
    """Update imports in a single file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Apply all import mappings
        for old_pattern, new_import in IMPORT_MAPPINGS.items():
            content = re.sub(old_pattern, new_import, content, flags=re.MULTILINE)
        
        # Only write if changes were made
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Updated: {file_path}")
            return True
        return False
    except Exception as e:
        print(f"Error updating {file_path}: {e}")
        return False

def main():
    """Update imports in all Python files in src/."""
    src_dir = Path(__file__).parent / "src"
    
    if not src_dir.exists():
        print(f"Source directory not found: {src_dir}")
        return
    
    updated_count = 0
    for py_file in src_dir.rglob("*.py"):
        if update_file_imports(py_file):
            updated_count += 1
    
    print(f"\nUpdated {updated_count} files.")

if __name__ == "__main__":
    main()
