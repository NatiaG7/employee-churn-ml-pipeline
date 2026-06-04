"""Score employee churn risk from trained model."""

import json
from pathlib import Path

import joblib
import pandas as pd

from src.features import prepare_features, load_and_clean

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"


def risk_zone(probability: float) -> str:
    """Map churn probability to HR action zone."""
    if probability < 0.2:
        return "Safe (Green)"
    if probability < 0.6:
        return "Low Risk (Yellow)"
    if probability < 0.9:
        return "Medium Risk (Orange)"
    return "High Risk (Red)"


def load_feature_columns(model_dir: Path) -> list[str]:
    """Load feature column names saved during training."""
    path = model_dir / "feature_columns.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run: python scripts/train.py"
        )

    try:
        with open(path, encoding="utf-8") as f:
            columns = json.load(f)
    except (UnicodeDecodeError, json.JSONDecodeError):
        # Backward compat: older runs used joblib.dump(..., "feature_columns.json")
        columns = joblib.load(path)

    if not isinstance(columns, list):
        raise ValueError(f"Expected list of column names in {path}, got {type(columns)}")
    return columns


def load_model(model_dir: Path | None = None):
    model_dir = model_dir or MODEL_DIR
    model_path = model_dir / "churn_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing {model_path}. Run: python scripts/train.py"
        )
    model = joblib.load(model_path)
    columns = load_feature_columns(model_dir)
    return model, columns


def score_dataframe(df: pd.DataFrame, model_dir: Path | None = None) -> pd.DataFrame:
    """Add turnover_probability and risk_zone columns."""
    model, columns = load_model(model_dir)
    X, _ = prepare_features(df)
    X = X.reindex(columns=columns, fill_value=0)

    proba = model.predict_proba(X)[:, 1]
    out = df.copy()
    out["turnover_probability"] = proba
    out["risk_zone"] = [risk_zone(p) for p in proba]
    return out


def score_csv(csv_path: str | Path, model_dir: Path | None = None) -> pd.DataFrame:
    df = load_and_clean(str(csv_path))
    return score_dataframe(df, model_dir)
