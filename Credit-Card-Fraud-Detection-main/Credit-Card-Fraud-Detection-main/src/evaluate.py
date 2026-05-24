"""
Model performance evaluation module.
Computes class-imbalance-resilient metrics and exports automated JSON & Markdown performance reports.
"""

import os
import json
from typing import Dict, Any, Union
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    precision_recall_curve,
    auc,
    confusion_matrix,
    classification_report
)
from imblearn.pipeline import Pipeline as ImblearnPipeline
from sklearn.pipeline import Pipeline as SklearnPipeline

from src.utils import setup_logger
from src.exceptions import InferenceError

# Initialize structured logger
logger = setup_logger()

# Type alias for pipelines
AnyPipeline = Union[SklearnPipeline, ImblearnPipeline]


def calculate_metrics(
    y_true: Union[pd.Series, np.ndarray], 
    y_pred: Union[pd.Series, np.ndarray], 
    y_probs: Union[pd.Series, np.ndarray]
) -> Dict[str, float]:
    """
    Computes classification metrics with a high focus on class imbalance metrics (PR-AUC, MCC).
    
    Args:
        y_true: Target ground truth labels.
        y_pred: Predicted class labels.
        y_probs: Probability estimates of the positive class.
        
    Returns:
        Dict[str, float]: Dictionary mapping metric names to computed values.
    """
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    mcc = float(matthews_corrcoef(y_true, y_pred))
    
    # Area under ROC Curve
    roc_auc = float(roc_auc_score(y_true, y_probs))
    
    # Area under Precision-Recall Curve (PR-AUC)
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_probs)
    pr_auc = float(auc(recall_vals, precision_vals))
    
    return {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "MCC": mcc,
        "ROC-AUC": roc_auc,
        "PR-AUC": pr_auc
    }


def generate_markdown_report(
    model_name: str, 
    metrics: Dict[str, float], 
    cm: np.ndarray, 
    report_path: str
) -> None:
    """
    Auto-generates a publication-grade markdown performance report.
    
    Args:
        model_name: Evaluated model label.
        metrics: Computed metrics dictionary.
        cm: Confusion matrix array.
        report_path: Path to write the markdown report.
    """
    try:
        # Compute rates
        tn, fp, fn, tp = map(int, cm.ravel())
        total = tn + fp + fn + tp
        
        md_content = f"""# Performance Evaluation Report: {model_name}

Auto-generated on the testing partition.

## Executive Metrics Summary

| Evaluation Metric | Score | Scientific Relevance for Fraud Detection |
| :--- | :---: | :--- |
| **Accuracy** | {metrics['Accuracy']:.4f} | Percentage of correct classifications. **Highly misleading** due to major class imbalance. |
| **Precision** | {metrics['Precision']:.4f} | Ratio of true frauds to all flagged anomalies. Controls operational audit friction. |
| **Recall (Sensitivity)** | {metrics['Recall']:.4f} | Percentage of actual frauds intercepted. Controls financial loss prevention. |
| **F1-Score** | {metrics['F1-Score']:.4f} | Harmonic mean of Precision and Recall. Balances both dimensions. |
| **Matthews Correlation Coefficient (MCC)** | {metrics['MCC']:.4f} | Balanced correlation measure ranging from -1 to +1. High score indicates a perfect predictor. |
| **ROC-AUC** | {metrics['ROC-AUC']:.4f} | Area Under ROC Curve. Sensitivity vs False Positive Rate trade-offs. |
| **PR-AUC** | {metrics['PR-AUC']:.4f} | **Primary Metric.** Area Under Precision-Recall Curve. Crucial for heavy minority class splits. |

---

## Confusion Matrix Analysis

```
                  Predicted Legitimate    Predicted Fraudulent
Actual Legitimate       {tn:<10d} (TN)           {fp:<10d} (FP)
Actual Fraudulent       {fn:<10d} (FN)           {tp:<10d} (TP)
```

*   **Total Transactions Analyzed:** {total}
*   **Actual Fraud Cases:** {fn + tp}
*   **Successfully Intercepted (True Positives):** {tp} ({tp / (fn + tp) * 100:.2f}%)
*   **Missed Fraud (False Negatives - Financial Loss):** {fn}
*   **False Alarms (False Positives - Audit Friction):** {fp}

---

## Academic Formulas and Reference

1.  **Matthews Correlation Coefficient (MCC):**
    $$\\text{{MCC}} = \\frac{{\\text{{TP}} \\times \\text{{TN}} - \\text{{FP}} \\times \\text{{FN}}}}{{\\sqrt{{(\\text{{TP}} + \\text{{FP}})(\\text{{TP}} + \\text{{FN}})(\\text{{TN}} + \\text{{FP}})(\\text{{TN}} + \\text{{FN}})}}}}$$
    
2.  **Precision-Recall Area Under Curve (PR-AUC):**
    Integrates the Precision ($P$) as a function of Recall ($R$):
    $$\\text{{PR-AUC}} = \\int_0^1 P(R) \\, dR$$
"""
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        logger.info(f"Markdown executive summary report written to: {report_path}")
        
    except Exception as e:
        logger.warning(f"Failed to generate Markdown report: {e}")


def evaluate_pipeline(
    model: AnyPipeline, 
    X_test: pd.DataFrame, 
    y_test: pd.Series, 
    model_name: str,
    reports_dir: str = "reports"
) -> Dict[str, float]:
    """
    Evaluates a model pipeline on test data, prints metrics, exports JSON, and generates markdown reports.
    
    Args:
        model: Trained scikit-learn or imblearn Pipeline.
        X_test: Test features DataFrame.
        y_test: Test target ground truth labels.
        model_name: Custom name identifier for the model.
        reports_dir: Directory where evaluation reports are stored.
        
    Returns:
        Dict[str, float]: Computed metrics dictionary.
        
    Raises:
        InferenceError: If prediction or metrics calculation fails.
    """
    logger.info(f"Evaluating model pipeline '{model_name}' on test partition...")
    
    try:
        y_pred = model.predict(X_test)
        
        # Safe extraction of probabilities
        if hasattr(model, "predict_proba"):
            y_probs = model.predict_proba(X_test)[:, 1]
        else:
            logger.warning("Pipeline does not support predict_proba. Falling back to discrete predictions.")
            y_probs = y_pred.astype(float)
            
        metrics = calculate_metrics(y_test, y_pred, y_probs)
        
        # Log metrics to console
        logger.info(f"Evaluation Metrics for {model_name}:")
        for metric, val in metrics.items():
            logger.info(f"  {metric:10s} : {val:.4f}")
            
        cm = confusion_matrix(y_test, y_pred)
        logger.info(f"Confusion Matrix:\nTN: {cm[0,0]} | FP: {cm[0,1]}\nFN: {cm[1,0]} | TP: {cm[1,1]}")
        
        report_text = classification_report(y_test, y_pred, digits=4)
        logger.info(f"Classification Report:\n{report_text}")
        
        # Setup reports directory
        os.makedirs(reports_dir, exist_ok=True)
        model_key = model_name.lower().replace(" ", "_")
        
        # Save JSON Report
        json_path = os.path.join(reports_dir, f"{model_key}_report.json")
        report_data = {
            "model_name": model_name,
            "metrics": metrics,
            "confusion_matrix": {
                "TN": int(cm[0,0]),
                "FP": int(cm[0,1]),
                "FN": int(cm[1,0]),
                "TP": int(cm[1,1])
            }
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=4)
        logger.info(f"Serialized JSON report: {json_path}")
        
        # Save Markdown Report
        md_path = os.path.join(reports_dir, f"{model_key}_report.md")
        generate_markdown_report(model_name, metrics, cm, md_path)
        
        return metrics
        
    except Exception as e:
        logger.error(f"Inference evaluation failed for {model_name}: {e}")
        raise InferenceError(f"Pipeline evaluation failed: {str(e)}")
