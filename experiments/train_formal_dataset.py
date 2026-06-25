"""Compatibility wrapper for the earlier formal-Dataset baseline command.

Use ``experiments/train.py`` and ``experiments/test.py`` for the standard
handoff workflow. This wrapper is kept so older notes and screenshots that
mention ``train_formal_dataset.py`` still have a working command.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.train import main


if __name__ == "__main__":
    main()

