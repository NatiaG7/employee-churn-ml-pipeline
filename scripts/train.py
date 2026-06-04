#!/usr/bin/env python3
"""Train churn models. Requires data/raw/HR_comma_sep.csv."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.train import train  # noqa: E402

if __name__ == "__main__":
    train()
