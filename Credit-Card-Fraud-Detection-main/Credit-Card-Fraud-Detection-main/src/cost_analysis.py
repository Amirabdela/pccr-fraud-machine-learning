"""
Financial cost analysis module for credit card fraud detection.
Computes the monetary impact of False Positives (wasted investigations) and
False Negatives (missed fraud losses) across classification thresholds.
"""

from typing import Dict, Optional, Tuple
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils import setup_logger

logger = setup_logger()


def compute_financial_cost(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
    avg_fraud_amount: float = 122.21,
    investigation_cost: float = 15.0,
    fraud_recovery_rate: float = 0.25,
) -> Dict[str, float]:
    """
    Compute the total financial cost of model decisions at a given threshold.

    Cost model:
    - False Negative (missed fraud): loss = avg_fraud_amount * (1 - fraud_recovery_rate)
    - False Positive (false alarm):  cost = investigation_cost
    - True Positive (caught fraud):  savings = avg_fraud_amount * fraud_recovery_rate

    Args:
        y_true: Ground-truth binary labels.
        y_proba: Predicted fraud probabilities.
        threshold: Classification cutoff.
        avg_fraud_amount: Average transaction amount for fraudulent transactions (USD).
        investigation_cost: Cost per false-alarm investigation (USD).
        fraud_recovery_rate: Fraction of fraud amount recovered when caught (0–1).

    Returns:
        dict with financial cost breakdown.
    """
    y_pred = (y_proba >= threshold).astype(int)

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))

    fn_cost = fn * avg_fraud_amount * (1 - fraud_recovery_rate)
    fp_cost = fp * investigation_cost
    tp_savings = tp * avg_fraud_amount * fraud_recovery_rate
    total_cost = fn_cost + fp_cost
    net_benefit = tp_savings - total_cost

    result = {
        "threshold": float(threshold),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "fn_cost_usd": round(fn_cost, 2),
        "fp_cost_usd": round(fp_cost, 2),
        "tp_savings_usd": round(tp_savings, 2),
        "total_cost_usd": round(total_cost, 2),
        "net_benefit_usd": round(net_benefit, 2),
        "avg_fraud_amount": avg_fraud_amount,
        "investigation_cost": investigation_cost,
        "fraud_recovery_rate": fraud_recovery_rate,
    }

    logger.info(
        f"Cost analysis @ threshold={threshold:.3f}: "
        f"FN cost=${fn_cost:,.2f} | FP cost=${fp_cost:,.2f} | "
        f"Savings=${tp_savings:,.2f} | Net=${net_benefit:,.2f}"
    )
    return result


def plot_cost_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    output_dir: str = "visuals",
    model_name: str = "model",
    avg_fraud_amount: float = 122.21,
    investigation_cost: float = 15.0,
    fraud_recovery_rate: float = 0.25,
) -> str:
    """
    Plot net benefit, FN cost, and FP cost across all classification thresholds.

    Args:
        y_true: Ground-truth labels.
        y_proba: Predicted fraud probabilities.
        output_dir: Directory for the plot.
        model_name: Model identifier for filename.
        avg_fraud_amount: Average fraud transaction amount.
        investigation_cost: Cost per false-alarm.
        fraud_recovery_rate: Fraud recovery fraction.

    Returns:
        str: Path to the saved cost curve plot.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{model_name}_cost_curve.png")

    thresholds = np.linspace(0.01, 0.99, 200)
    fn_costs, fp_costs, net_benefits = [], [], []

    for thresh in thresholds:
        result = compute_financial_cost(
            y_true, y_proba, thresh,
            avg_fraud_amount, investigation_cost, fraud_recovery_rate
        )
        fn_costs.append(result["fn_cost_usd"])
        fp_costs.append(result["fp_cost_usd"])
        net_benefits.append(result["net_benefit_usd"])

    best_idx = np.argmax(net_benefits)
    best_thresh = thresholds[best_idx]
    best_net = net_benefits[best_idx]

    fig, axes = plt.subplots(2, 1, figsize=(11, 9))

    ax1 = axes[0]
    ax1.plot(thresholds, fn_costs, label="FN Cost (missed fraud loss)", color="#e74c3c", lw=2)
    ax1.plot(thresholds, fp_costs, label="FP Cost (investigation overhead)", color="#f39c12", lw=2)
    ax1.axvline(best_thresh, color="#9b59b6", linestyle="--", alpha=0.8,
                label=f"Optimal threshold = {best_thresh:.3f}")
    ax1.set_ylabel("Cost (USD)", fontsize=11)
    ax1.set_title(f"Financial Cost Breakdown – {model_name}", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    ax2 = axes[1]
    ax2.plot(thresholds, net_benefits, label="Net Benefit (USD)", color="#2ecc71", lw=2.5)
    ax2.axhline(0, color="black", linestyle=":", lw=0.8)
    ax2.axvline(best_thresh, color="#9b59b6", linestyle="--", alpha=0.8,
                label=f"Max net benefit = ${best_net:,.2f} @ {best_thresh:.3f}")
    ax2.fill_between(thresholds, net_benefits, 0,
                     where=[v > 0 for v in net_benefits],
                     alpha=0.15, color="#2ecc71", label="Positive benefit region")
    ax2.set_xlabel("Classification Threshold", fontsize=11)
    ax2.set_ylabel("Net Benefit (USD)", fontsize=11)
    ax2.set_title("Net Financial Benefit vs Threshold", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close("all")

    logger.info(f"Cost curve saved: {output_path}")
    return output_path


def save_cost_report(
    cost_result: Dict,
    output_dir: str = "reports",
    model_name: str = "model",
) -> str:
    """Save cost analysis results to a JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{model_name}_cost_report.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cost_result, f, indent=2)
    logger.info(f"Cost report saved: {output_path}")
    return output_path
