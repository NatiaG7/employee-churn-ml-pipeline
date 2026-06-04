"""Train churn models with SMOTE and cross-validation."""

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.metrics import classification_report, roc_auc_score

from src.features import load_and_clean, prepare_features
from src.predict import score_dataframe

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"


def evaluate_cv(model, X, y, n_splits: int = 5) -> dict:
    """Stratified k-fold CV; return classification report dict."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)
    y_pred = cross_val_predict(model, X, y, cv=skf)
    return classification_report(y, y_pred, output_dict=True)


def train(
    data_path: str | Path | None = None,
    model_dir: Path | None = None,
    output_dir: Path | None = None,
) -> dict:
    """Train models, save best (Gradient Boosting), metrics, and figures."""
    data_path = Path(data_path or ROOT / "data" / "raw" / "HR_comma_sep.csv")
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}. See data/README.md for download instructions."
        )

    model_dir = model_dir or MODEL_DIR
    output_dir = output_dir or OUTPUT_DIR
    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = load_and_clean(str(data_path))
    X, y = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=123
    )

    smote = SMOTE(random_state=123)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=123),
        "gradient_boosting": GradientBoostingClassifier(random_state=123),
    }

    metrics = {
        "_meta": {
            "status": "executed",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "dataset": str(data_path.name),
            "rows": len(df),
            "left_rate": float(y.mean()),
        },
        "cv_reports": {},
        "test_auc": {},
        "test_classification_report": {},
    }

    model_probs = {}

    for name, model in models.items():
        metrics["cv_reports"][name] = evaluate_cv(model, X_train_sm, y_train_sm)
        model.fit(X_train_sm, y_train_sm)
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        model_probs[name] = y_prob
        metrics["test_auc"][name] = round(float(roc_auc_score(y_test, y_prob)), 4)
        if name == "gradient_boosting":
            metrics["test_classification_report"] = classification_report(
                y_test, y_pred, output_dict=True
            )

    best_name = "gradient_boosting"
    best_model = models[best_name]
    joblib.dump(best_model, model_dir / "churn_model.pkl")
    with open(model_dir / "feature_columns.json", "w", encoding="utf-8") as f:
        json.dump(list(X.columns), f, indent=2)

    y_prob_best = model_probs[best_name]
    y_pred_best = (y_prob_best >= 0.5).astype(int)

    try:
        from src.evaluate import plot_confusion_matrix, plot_roc_curves

        plot_roc_curves(model_probs, y_test, FIGURES_DIR / "roc_comparison.png")
        plot_confusion_matrix(
            y_test,
            y_pred_best,
            FIGURES_DIR / "confusion_matrix_gradient_boosting.png",
            title="Gradient Boosting — Test Set",
        )
        metrics["figures"] = [
            "outputs/figures/roc_comparison.png",
            "outputs/figures/confusion_matrix_gradient_boosting.png",
        ]
    except Exception as e:
        metrics["figures_error"] = str(e)

    test_df = df.loc[X_test.index].copy()
    scored = score_dataframe(test_df, model_dir=model_dir)
    risk_counts = scored["risk_zone"].value_counts().to_dict()
    metrics["risk_zone_distribution_test"] = risk_counts
    metrics["best_model"] = best_name
    metrics["train_rows"] = int(len(X_train_sm))
    metrics["test_rows"] = int(len(X_test))

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("Test ROC-AUC:")
    print(json.dumps(metrics["test_auc"], indent=2))
    print("\nRisk zones (test set):", risk_counts)
    return metrics
