#!/usr/bin/env python3
"""
Main entry point for the Knowledge Layer Pipeline.

This is the primary CLI interface for the pipeline.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.pipeline.orchestrator import main

if __name__ == "__main__":
    main()
