"""Evaluation plots and reports for churn models."""

from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay, confusion_matrix, ConfusionMatrixDisplay


def plot_roc_curves(model_probs: dict, y_test, output_path: Path) -> None:
    """
    model_probs: dict of model_name -> y_prob array
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))

    for name, y_prob in model_probs.items():
        RocCurveDisplay.from_predictions(y_test, y_prob, name=name)

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    plt.title("ROC Curve Comparison (Test Set)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()


def plot_confusion_matrix(y_test, y_pred, output_path: Path, title: str = "Churn model") -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Stayed (0)", "Left (1)"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()
