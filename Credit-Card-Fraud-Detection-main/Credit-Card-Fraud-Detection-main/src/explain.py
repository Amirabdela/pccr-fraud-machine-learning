"""
Model Explainability module using SHAP (SHapley Additive exPlanations).
Provides feature-level explanations for fraud predictions.
"""

from typing import Optional, List
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils import setup_logger
from src.exceptions import FraudPipelineError

logger = setup_logger()


class ExplainabilityError(FraudPipelineError):
    """Raised when SHAP explanation computation fails."""
    pass


def compute_shap_values(model, X: pd.DataFrame, max_samples: int = 500):
    """
    Compute SHAP values for a trained pipeline using TreeExplainer or KernelExplainer.

    Args:
        model: Fitted sklearn/imblearn Pipeline with a tree-based or linear classifier.
        X: Feature DataFrame to explain (raw, pre-transform).
        max_samples: Maximum number of samples to compute SHAP values for.

    Returns:
        tuple: (shap_values, explainer, X_transformed)
    """
    try:
        import shap
    except ImportError:
        raise ExplainabilityError(
            "shap is not installed. Run: pip install shap>=0.41.0"
        )

    logger.info(f"Computing SHAP values for {min(len(X), max_samples)} samples...")

    try:
        # Sample for efficiency
        X_sample = X.sample(n=min(max_samples, len(X)), random_state=42)

        # Extract preprocessor + classifier from pipeline
        steps = list(model.named_steps.items())
        preprocessor = steps[0][1]

        # Find the classifier (last step)
        classifier = steps[-1][1]

        # Transform features through the pipeline (excluding SMOTE & classifier)
        X_transformed = X_sample.copy()
        for name, step in steps[:-1]:
            if hasattr(step, "transform"):
                try:
                    X_transformed = step.transform(X_transformed)
                except Exception:
                    pass

        # Convert to DataFrame if numpy array
        if hasattr(X_transformed, "toarray"):
            X_transformed = X_transformed.toarray()
        if not isinstance(X_transformed, pd.DataFrame):
            X_transformed = pd.DataFrame(X_transformed)

        # Choose appropriate explainer
        clf_name = type(classifier).__name__.lower()
        if any(x in clf_name for x in ["forest", "xgb", "gradient", "tree", "boost"]):
            logger.info("Using TreeExplainer for tree-based model...")
            explainer = shap.TreeExplainer(classifier)
            shap_values = explainer.shap_values(X_transformed)
            # For binary classification, take fraud class (index 1)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
        else:
            logger.info("Using KernelExplainer (sampling background)...")
            background = shap.sample(X_transformed, 100)
            explainer = shap.KernelExplainer(classifier.predict_proba, background)
            shap_values = explainer.shap_values(X_transformed, nsamples=100)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]

        logger.info("SHAP values computed successfully.")
        return shap_values, explainer, X_transformed

    except ExplainabilityError:
        raise
    except Exception as e:
        logger.error(f"SHAP computation failed: {e}")
        raise ExplainabilityError(f"SHAP computation failed: {str(e)}") from e


def plot_shap_summary(
    shap_values,
    X_transformed: pd.DataFrame,
    feature_names: Optional[List[str]] = None,
    output_dir: str = "visuals",
    model_name: str = "model",
    max_features: int = 20,
) -> str:
    """
    Generate and save a SHAP summary (beeswarm) plot.

    Args:
        shap_values: SHAP values array (n_samples, n_features).
        X_transformed: Transformed feature DataFrame.
        feature_names: Optional list of feature names for axis labels.
        output_dir: Directory to save the plot.
        model_name: Model identifier for the filename.
        max_features: Max features to display on the plot.

    Returns:
        str: Absolute path to the saved plot file.
    """
    try:
        import shap
    except ImportError:
        raise ExplainabilityError("shap is not installed. Run: pip install shap>=0.41.0")

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{model_name}_shap_summary.png")

    logger.info(f"Generating SHAP summary plot → {output_path}")

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        X_transformed,
        feature_names=feature_names,
        max_display=max_features,
        show=False,
        plot_type="dot",
    )
    plt.title(f"SHAP Feature Importance – {model_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close("all")

    logger.info(f"SHAP summary plot saved: {output_path}")
    return output_path


def plot_shap_waterfall(
    shap_values,
    X_transformed: pd.DataFrame,
    sample_index: int = 0,
    feature_names: Optional[List[str]] = None,
    output_dir: str = "visuals",
    model_name: str = "model",
) -> str:
    """
    Generate a SHAP waterfall plot for a single prediction explanation.

    Args:
        shap_values: SHAP values array.
        X_transformed: Transformed feature DataFrame.
        sample_index: Index of the sample to explain.
        feature_names: Feature names for labelling.
        output_dir: Directory to save the plot.
        model_name: Model identifier for the filename.

    Returns:
        str: Path to the saved waterfall plot.
    """
    try:
        import shap
    except ImportError:
        raise ExplainabilityError("shap is not installed. Run: pip install shap>=0.41.0")

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{model_name}_shap_waterfall_sample{sample_index}.png")

    logger.info(f"Generating SHAP waterfall plot for sample #{sample_index} → {output_path}")

    fig, ax = plt.subplots(figsize=(10, 6))

    sample_shap = shap_values[sample_index]
    sample_features = X_transformed.iloc[sample_index].values

    # Bar chart waterfall approximation (compatible across shap versions)
    sorted_idx = np.argsort(np.abs(sample_shap))[::-1][:15]
    values = sample_shap[sorted_idx]
    names = (
        [feature_names[i] for i in sorted_idx]
        if feature_names
        else [f"Feature {i}" for i in sorted_idx]
    )

    colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in values]
    ax.barh(range(len(values)), values, color=colors)
    ax.set_yticks(range(len(values)))
    ax.set_yticklabels(names, fontsize=9)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP Value (impact on fraud probability)", fontsize=11)
    ax.set_title(
        f"SHAP Explanation – Sample #{sample_index} | {model_name}",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close("all")

    logger.info(f"SHAP waterfall plot saved: {output_path}")
    return output_path


def generate_shap_report(
    shap_values,
    X_transformed: pd.DataFrame,
    feature_names: Optional[List[str]] = None,
    top_n: int = 10,
    output_dir: str = "reports",
    model_name: str = "model",
) -> str:
    """
    Generate a Markdown report summarizing the top SHAP feature importances.

    Args:
        shap_values: SHAP values array.
        X_transformed: Transformed features.
        feature_names: Feature names.
        top_n: Number of top features to include.
        output_dir: Directory for the report.
        model_name: Model identifier.

    Returns:
        str: Path to the saved Markdown report.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{model_name}_shap_report.md")

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    sorted_idx = np.argsort(mean_abs_shap)[::-1][:top_n]

    names = (
        [feature_names[i] for i in sorted_idx]
        if feature_names
        else [f"Feature {i}" for i in sorted_idx]
    )
    values = mean_abs_shap[sorted_idx]

    lines = [
        f"# SHAP Feature Importance Report – {model_name}\n",
        f"**Top {top_n} features ranked by mean absolute SHAP value.**\n",
        "| Rank | Feature | Mean |SHAP| |\n",
        "|------|---------|-------------|\n",
    ]
    for rank, (name, val) in enumerate(zip(names, values), start=1):
        lines.append(f"| {rank} | `{name}` | {val:.6f} |\n")

    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    logger.info(f"SHAP report saved: {output_path}")
    return output_path
