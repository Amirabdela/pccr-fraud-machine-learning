"""
Data Drift Detection module for credit card fraud detection.
Monitors feature distributions between a reference dataset and incoming data
using statistical tests (KS test, Population Stability Index).
"""

from typing import Dict, List, Optional, Tuple
import json
import os
from datetime import datetime
import numpy as np
import pandas as pd
from scipy import stats

from src.utils import setup_logger

logger = setup_logger()


def compute_psi(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Compute the Population Stability Index (PSI) between two distributions.

    PSI < 0.1: No significant drift.
    0.1 <= PSI < 0.2: Moderate drift – monitor.
    PSI >= 0.2: Significant drift – investigate.

    Args:
        reference: Reference distribution values (training data).
        current: Current distribution values (incoming data).
        n_bins: Number of histogram bins.

    Returns:
        float: PSI value.
    """
    eps = 1e-8
    bins = np.percentile(reference, np.linspace(0, 100, n_bins + 1))
    bins = np.unique(bins)

    ref_counts, _ = np.histogram(reference, bins=bins)
    cur_counts, _ = np.histogram(current, bins=bins)

    ref_pct = ref_counts / (len(reference) + eps) + eps
    cur_pct = cur_counts / (len(current) + eps) + eps

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


def detect_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    feature_columns: Optional[List[str]] = None,
    ks_alpha: float = 0.05,
    psi_threshold: float = 0.1,
) -> Dict[str, Dict]:
    """
    Detect feature drift between a reference dataset and current incoming data.

    For each feature, runs:
    - Kolmogorov-Smirnov (KS) test for distributional shift.
    - Population Stability Index (PSI) for magnitude of drift.

    Args:
        reference_df: Training/reference feature DataFrame.
        current_df: Incoming/production feature DataFrame.
        feature_columns: Features to monitor. Defaults to all numeric columns.
        ks_alpha: Significance level for KS test.
        psi_threshold: PSI threshold for flagging drift.

    Returns:
        dict: Per-feature drift results with keys:
            - 'ks_statistic', 'ks_pvalue', 'ks_drift_detected'
            - 'psi', 'psi_drift_detected'
            - 'drift_detected' (combined flag)
    """
    if feature_columns is None:
        feature_columns = reference_df.select_dtypes(include=[np.number]).columns.tolist()

    logger.info(f"Running drift detection on {len(feature_columns)} features...")

    results: Dict[str, Dict] = {}
    drifted_features = []

    for col in feature_columns:
        if col not in reference_df.columns or col not in current_df.columns:
            logger.warning(f"Feature '{col}' not found in both datasets. Skipping.")
            continue

        ref_vals = reference_df[col].dropna().values
        cur_vals = current_df[col].dropna().values

        if len(ref_vals) == 0 or len(cur_vals) == 0:
            continue

        # KS test
        ks_stat, ks_pval = stats.ks_2samp(ref_vals, cur_vals)
        ks_drift = bool(ks_pval < ks_alpha)

        # PSI
        psi_val = compute_psi(ref_vals, cur_vals)
        psi_drift = bool(psi_val >= psi_threshold)

        combined_drift = ks_drift or psi_drift

        results[col] = {
            "ks_statistic": round(float(ks_stat), 6),
            "ks_pvalue": round(float(ks_pval), 6),
            "ks_drift_detected": ks_drift,
            "psi": round(psi_val, 6),
            "psi_drift_detected": psi_drift,
            "drift_detected": combined_drift,
        }

        if combined_drift:
            drifted_features.append(col)

    drift_ratio = len(drifted_features) / max(len(feature_columns), 1)
    logger.info(
        f"Drift detection complete. "
        f"{len(drifted_features)}/{len(feature_columns)} features show drift "
        f"(ratio={drift_ratio:.2%})."
    )
    if drifted_features:
        logger.warning(f"Drifted features: {drifted_features}")

    return results


def generate_drift_report(
    drift_results: Dict[str, Dict],
    output_dir: str = "reports",
    reference_date: Optional[str] = None,
    current_date: Optional[str] = None,
) -> str:
    """
    Save drift detection results as a JSON report and a Markdown summary.

    Args:
        drift_results: Output from detect_drift().
        output_dir: Directory to write reports into.
        reference_date: Optional label for the reference period.
        current_date: Optional label for the current period.

    Returns:
        str: Path to the saved Markdown drift report.
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    # JSON report
    json_path = os.path.join(output_dir, f"drift_report_{timestamp}.json")
    json_payload = {
        "generated_at": datetime.utcnow().isoformat(),
        "reference_period": reference_date or "training data",
        "current_period": current_date or "incoming data",
        "feature_drift": drift_results,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_payload, f, indent=2)

    # Markdown report
    md_path = os.path.join(output_dir, f"drift_report_{timestamp}.md")
    drifted = [f for f, v in drift_results.items() if v.get("drift_detected")]
    stable = [f for f, v in drift_results.items() if not v.get("drift_detected")]

    lines = [
        "# Data Drift Detection Report\n\n",
        f"**Generated:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}  \n",
        f"**Reference period:** {reference_date or 'training data'}  \n",
        f"**Current period:** {current_date or 'incoming data'}  \n\n",
        f"**Drifted features:** {len(drifted)} / {len(drift_results)}  \n\n",
        "## Feature Drift Summary\n\n",
        "| Feature | KS Stat | KS p-value | PSI | Drift? |\n",
        "|---------|---------|-----------|-----|-------|\n",
    ]
    for feat, res in drift_results.items():
        flag = "🔴 YES" if res["drift_detected"] else "🟢 No"
        lines.append(
            f"| `{feat}` | {res['ks_statistic']:.4f} | "
            f"{res['ks_pvalue']:.4f} | {res['psi']:.4f} | {flag} |\n"
        )

    if drifted:
        lines += ["\n## ⚠️ Drifted Features\n\n"]
        for f in drifted:
            lines.append(f"- `{f}`\n")

    with open(md_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    logger.info(f"Drift report saved: JSON→{json_path} | MD→{md_path}")
    return md_path
