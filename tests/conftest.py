"""pytest configuration – add the project root to sys.path so that
``import config`` and ``from src.xxx import ...`` work from the tests/ dir.
"""

import sys
from pathlib import Path

# Project root is one level up from tests/
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
