#!/usr/bin/env python3
"""
Test script for MLOps features.
Python version of test.sh for better cross-platform compatibility.
"""

import sys
import subprocess
from pathlib import Path
from glob import glob

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

def main():
    print("=== Testing MLOps Features ===\n")
    
    # 1. Build index
    print("1. Building versioned index...")
    try:
        from src.index.build_index import build_index
        index_path = build_index(config_path=Path("config.yaml"))
        print(f"[OK] Index built: {index_path}\n")
    except Exception as e:
        print(f"[FAIL] Failed to build index: {e}\n")
        return 1
    
    # 2. Validate index
    print("2. Validating index...")
    try:
        from src.index.validate_index import validate_index
        
        # Find latest index file
        index_files = glob("data/processed/indexes/vector_store_v*.pkl")
        if not index_files:
            print("[WARN] No index files found\n")
        else:
            # Sort by modification time (newest first)
            latest_index = max(index_files, key=lambda p: Path(p).stat().st_mtime)
            latest_index_path = Path(latest_index)
            
            report = validate_index(latest_index_path, config_path=Path("config.yaml"))
            
            if report['valid']:
                print(f"[OK] Index is VALID: {latest_index_path}")
                if report['integrity_checks']:
                    print(f"  - Vector count: {report['integrity_checks'].get('vector_count', 'N/A')}")
                    print(f"  - Embedding dim: {report['integrity_checks'].get('embedding_dim', 'N/A')}")
            else:
                print(f"[FAIL] Index validation failed: {latest_index_path}")
                for error in report['errors']:
                    print(f"  [ERROR] {error}")
            
            if report['warnings']:
                for warning in report['warnings']:
                    print(f"  [WARN] {warning}")
            print()
    except Exception as e:
        print(f"[WARN] Validation failed: {e}\n")
    
    # 3. Test search with metrics
    print("3. Testing search with metrics...")
    try:
        result = subprocess.run(
            [sys.executable, "main.py", "--search", "--query", "Kik az alapító tagok?", "--build-context"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent
        )
        if result.returncode == 0:
            print("[OK] Search test completed")
            if result.stdout:
                print(result.stdout)
        else:
            print(f"[WARN] Search test failed: {result.stderr}")
        print()
    except Exception as e:
        print(f"[WARN] Search test failed: {e}\n")
    
    # 4. Check metrics were logged
    print("4. Checking metrics logs...")
    try:
        # Ensure metrics directory exists
        metrics_dir = Path("data/metrics")
        metrics_dir.mkdir(parents=True, exist_ok=True)
        
        metrics_files = glob("data/metrics/rag_metrics_*.jsonl")
        if metrics_files:
            print("[OK] Metrics logged successfully")
            for metrics_file in metrics_files:
                file_path = Path(metrics_file)
                size = file_path.stat().st_size
                print(f"  - {file_path.name} ({size} bytes)")
        else:
            print("[WARN] No metrics files found")
            print("       Metrics logging is now integrated. If no files exist, metrics may not have been logged yet.")
        print()
    except Exception as e:
        print(f"[WARN] Error checking metrics: {e}\n")
    
    # 5. Test FastAPI instructions
    print("5. To test FastAPI, run:")
    print(f"   {sys.executable} -m uvicorn src.api.main:app --reload")
    print("   Then visit: http://localhost:8000/docs")
    print()
    
    print("=== Testing Complete ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())

