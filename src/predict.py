"""Compatibility alias: prefer `python -m src.prediction` or `python src/prediction.py`."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.prediction import main

if __name__ == "__main__":
    main()
