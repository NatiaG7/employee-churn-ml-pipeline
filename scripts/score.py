#!/usr/bin/env python3
"""Score employees with churn probability and HR risk zones."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.features import load_and_clean  # noqa: E402
from src.predict import score_dataframe  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Score HR churn risk")
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "data" / "raw" / "HR_comma_sep.csv",
        help="CSV with same schema as training data",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "risk_scores.csv",
        help="Where to save scored rows",
    )
    parser.add_argument("--top", type=int, default=10, help="Print top N highest risk")
    args = parser.parse_args()

    if not (ROOT / "models" / "churn_model.pkl").exists():
        print("Model not found. Run: python scripts/train.py", file=sys.stderr)
        sys.exit(1)

    df = load_and_clean(str(args.data))
    scored = score_dataframe(df)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cols = ["satisfaction_level", "last_evaluation", "number_project", "left", "sales", "salary"]
    cols = [c for c in cols if c in scored.columns]
    out_cols = cols + ["turnover_probability", "risk_zone"]
    scored[out_cols].to_csv(args.output, index=False)
    print(f"Saved scores to {args.output}")

    top = scored.nlargest(args.top, "turnover_probability")
    print(f"\nTop {args.top} churn risk:")
    print(top[out_cols].to_string(index=False))


if __name__ == "__main__":
    main()
