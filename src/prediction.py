"""兼容旧命令 `python -m src.prediction`；请优先使用 `python detect.py` 或 `python -m src.inference`."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.cli import main

if __name__ == "__main__":
    main()
