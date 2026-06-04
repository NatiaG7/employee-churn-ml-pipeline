"""Paths and training defaults."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw" / "HR_comma_sep.csv"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"

RANDOM_STATE = 123
TEST_SIZE = 0.2
SMOTE_RANDOM_STATE = 123
