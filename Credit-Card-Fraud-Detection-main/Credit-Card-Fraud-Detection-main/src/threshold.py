"""
Threshold optimization module for credit card fraud detection.
Finds the optimal classification threshold that maximizes F1 or MCC scores.
"""

from typing import Tuple, Dict, Optional
import numpy as np
import pandas as pd
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, matthews_corrcoef, precision_score, recall_score

from src.utils import setup_logger

logger = setup_logger()


def find_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    metric: str = "f1",
    thresholds: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Sweep probability thresholds to find the one maximising the chosen metric.

    Args:
        y_true: Ground-truth binary labels (0/1).
        y_proba: Predicted fraud probabilities from predict_proba.
        metric: Optimisation metric – 'f1' or 'mcc'.
        thresholds: Array of thresholds to sweep. Defaults to np.linspace(0.01, 0.99, 200).

    Returns:
        dict with keys:
            - 'optimal_threshold': best threshold value
            - 'best_score': score at optimal threshold
            - 'precision': precision at optimal threshold
            - 'recall': recall at optimal threshold
            - 'f1': f1 at optimal threshold
            - 'mcc': mcc at optimal threshold
    """
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 200)

    metric = metric.lower()
    if metric not in ("f1", "mcc"):
        raise ValueError(f"metric must be 'f1' or 'mcc', got '{metric}'")

    logger.info(f"Sweeping {len(thresholds)} thresholds to optimise {metric.upper()}...")

    best_threshold = 0.5
    best_score = -np.inf
    scores = []

    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)
        if metric == "f1":
            score = f1_score(y_true, y_pred, zero_division=0)
        else:
            score = matthews_corrcoef(y_true, y_pred)
        scores.append(score)
        if score > best_score:
            best_score = score
            best_threshold = thresh

    # Compute all metrics at the best threshold
    y_pred_best = (y_proba >= best_threshold).astype(int)
    result = {
        "optimal_threshold": float(best_threshold),
        "best_score": float(best_score),
        "metric_optimised": metric,
        "precision": float(precision_score(y_true, y_pred_best, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred_best, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred_best, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred_best)),
    }

    logger.info(
        f"Optimal threshold: {best_threshold:.4f} | "
        f"{metric.upper()}: {best_score:.4f} | "
        f"Precision: {result['precision']:.4f} | "
        f"Recall: {result['recall']:.4f}"
    )
    return result


def apply_threshold(y_proba: np.ndarray, threshold: float) -> np.ndarray:
    """
    Apply a custom probability threshold to produce binary predictions.

    Args:
        y_proba: Predicted fraud probabilities.
        threshold: Classification cutoff (0–1).

    Returns:
        np.ndarray: Binary predictions (0 = legitimate, 1 = fraud).
    """
    if not 0.0 < threshold < 1.0:
        raise ValueError(f"threshold must be in (0, 1), got {threshold}")
    return (y_proba >= threshold).astype(int)


def plot_threshold_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    output_dir: str = "visuals",
    model_name: str = "model",
) -> str:
    """
    Plot F1 and MCC scores as a function of classification threshold.

    Args:
        y_true: Ground-truth binary labels.
        y_proba: Predicted fraud probabilities.
        output_dir: Directory to save the plot.
        model_name: Model identifier used in the filename.

    Returns:
        str: Path to the saved plot.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{model_name}_threshold_curve.png")

    thresholds = np.linspace(0.01, 0.99, 200)
    f1_scores, mcc_scores = [], []

    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)
        f1_scores.append(f1_score(y_true, y_pred, zero_division=0))
        mcc_scores.append(matthews_corrcoef(y_true, y_pred))

    best_f1_thresh = thresholds[np.argmax(f1_scores)]
    best_mcc_thresh = thresholds[np.argmax(mcc_scores)]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(thresholds, f1_scores, label="F1 Score", color="#3498db", linewidth=2)
    ax.plot(thresholds, mcc_scores, label="MCC", color="#e74c3c", linewidth=2)
    ax.axvline(best_f1_thresh, color="#3498db", linestyle="--", alpha=0.7,
               label=f"Best F1 threshold = {best_f1_thresh:.3f}")
    ax.axvline(best_mcc_thresh, color="#e74c3c", linestyle="--", alpha=0.7,
               label=f"Best MCC threshold = {best_mcc_thresh:.3f}")
    ax.axvline(0.5, color="gray", linestyle=":", alpha=0.5, label="Default threshold (0.5)")

    ax.set_xlabel("Classification Threshold", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(f"Threshold Optimisation Curve – {model_name}", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.1, 1.05)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close("all")

    logger.info(f"Threshold curve plot saved: {output_path}")
    return output_path
