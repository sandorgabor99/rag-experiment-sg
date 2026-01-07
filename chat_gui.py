"""
Launch script for the Knowledge Layer Chat GUI.

Usage:
    python chat_gui.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.ui.chat_app import main

if __name__ == "__main__":
    main()
