"""Shared test-path setup for the repository's script-style modules."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

for path in (ROOT, ROOT / "minigrid_part", ROOT / "car_racing_part"):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)
