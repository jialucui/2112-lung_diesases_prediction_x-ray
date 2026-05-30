#!/usr/bin/env python3
"""
Root-level shortcut for CLI inference:
  python detect.py image.jpg
  python detect.py a.jpg b.jpg --json
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.cli import main

if __name__ == "__main__":
    main()
