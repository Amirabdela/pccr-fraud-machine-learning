"""
Data and model prediction visualizer.
Generates publication-quality charts (confusion matrix, ROC, Precision-Recall, correlation heatmap, etc.).
"""

import os
from typing import List, Union, Optional
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Headless backend to prevent display locks in server environments
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc
from imblearn.pipeline import Pipeline as ImblearnPipeline
from sklearn.pipeline import Pipeline as SklearnPipeline

from src.utils import setup_logger

# Initialize structured logger
logger = setup_logger()

# Type alias for pipelines
AnyPipeline = Union[SklearnPipeline, ImblearnPipeline]


def plot_confusion_matrix(
    y_true: Union[pd.Series, np.ndarray], 
    y_pred: Union[pd.Series, np.ndarray], 
    model_name: str,
    output_dir: str = "visuals"
) -> None:
    """
    Plots and exports a styled Confusion Matrix heatmap.
    
    Args:
        y_true: True binary target labels.
        y_pred: Predicted class labels.
        model_name: Name of evaluated model.
        output_dir: Directory where the image will be saved.
    """
    logger.info(f"Generating confusion matrix plot for {model_name}...")
    
    try:
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt="d", 
            cmap="Blues", 
            cbar=False,
            xticklabels=["Legit", "Fraud"],
            yticklabels=["Legit", "Fraud"]
        )
        plt.title(f"Confusion Matrix - {model_name}", fontsize=14, pad=10)
        plt.ylabel("Actual Label", fontsize=12)
        plt.xlabel("Predicted Label", fontsize=12)
        
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        logger.info(f"Confusion matrix plot saved to: {out_path}")
    except Exception as e:
        logger.error(f"Failed to generate confusion matrix plot: {e}")


def plot_evaluation_curves(
    y_true: Union[pd.Series, np.ndarray], 
    y_probs: Union[pd.Series, np.ndarray], 
    model_name: str,
    output_dir: str = "visuals"
) -> None:
    """
    Plots ROC and Precision-Recall curves side-by-side.
    
    Args:
        y_true: Target ground truth labels.
        y_probs: Class probability estimates of the positive class.
        model_name: Name of evaluated model.
        output_dir: Directory where the image will be saved.
    """
    logger.info(f"Generating ROC and PR evaluation curves for {model_name}...")
    
    try:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        
        # 1. ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        roc_auc = auc(fpr, tpr)
        axes[0].plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC-AUC = {roc_auc:.4f}")
        axes[0].plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--")
        axes[0].set_xlim([0.0, 1.0])
        axes[0].set_ylim([0.0, 1.05])
        axes[0].set_xlabel("False Positive Rate (FPR)", fontsize=11)
        axes[0].set_ylabel("True Positive Rate (Recall)", fontsize=11)
        axes[0].set_title("Receiver Operating Characteristic (ROC)", fontsize=12)
        axes[0].legend(loc="lower right")
        axes[0].grid(True, alpha=0.3)
        
        # 2. Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_true, y_probs)
        pr_auc = auc(recall, precision)
        axes[1].plot(recall, precision, color="forestgreen", lw=2, label=f"PR-AUC = {pr_auc:.4f}")
        axes[1].set_xlim([0.0, 1.0])
        axes[1].set_ylim([0.0, 1.05])
        axes[1].set_xlabel("Recall (Sensitivity)", fontsize=11)
        axes[1].set_ylabel("Precision (Positive Predictive Value)", fontsize=11)
        axes[1].set_title("Precision-Recall Curve (PRC)", fontsize=12)
        axes[1].legend(loc="lower left")
        axes[1].grid(True, alpha=0.3)
        
        plt.suptitle(f"Model Evaluation Curves - {model_name}", fontsize=15, y=0.98)
        
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"{model_name.lower().replace(' ', '_')}_evaluation_curves.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        logger.info(f"ROC & PR curves saved successfully to: {out_path}")
    except Exception as e:
        logger.error(f"Failed to generate evaluation curves plot: {e}")


def plot_feature_importance(
    model: AnyPipeline, 
    feature_names: List[str], 
    model_name: str,
    output_dir: str = "visuals"
) -> None:
    """
    Extracts and visualizes top feature importances from a Random Forest classifier.
    
    Args:
        model: Fitted model pipeline containing a classifier step.
        feature_names: List of all input feature names.
        model_name: Model identity label.
        output_dir: Directory where the image will be saved.
    """
    logger.info(f"Extracting feature importances for {model_name}...")
    
    try:
        classifier = model.named_steps["classifier"]
        if not hasattr(classifier, "feature_importances_"):
            logger.warning(f"Fitted model '{model_name}' does not expose feature importances.")
            return
            
        importances = classifier.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Take top 15 features
        top_n = min(15, len(feature_names))
        sorted_feats = [feature_names[i] for i in indices[:top_n]]
        sorted_imp = importances[indices[:top_n]]
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x=sorted_imp, y=sorted_feats, palette="viridis")
        plt.title(f"Top {top_n} Feature Importances - {model_name}", fontsize=14)
        plt.xlabel("Relative Importance (MDI)", fontsize=12)
        plt.ylabel("Features", fontsize=12)
        
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"{model_name.lower().replace(' ', '_')}_feature_importance.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        logger.info(f"Feature importance plot exported to: {out_path}")
        
    except Exception as e:
        logger.error(f"Failed to generate feature importance plot: {e}")


def plot_correlation_matrix(df: pd.DataFrame, output_dir: str = "visuals") -> None:
    """
    Generates a full Pearson correlation matrix heatmap for numerical attributes.
    
    Args:
        df: Raw or preprocessed DataFrame.
        output_dir: Directory where the image will be saved.
    """
    logger.info("Generating Pearson correlation matrix heatmap...")
    
    try:
        plt.figure(figsize=(12, 10))
        corrmat = df.corr()
        
        sns.heatmap(
            corrmat, 
            vmax=0.8, 
            square=True, 
            cmap="coolwarm", 
            cbar_kws={"shrink": 0.8},
            linewidths=0.05
        )
        plt.title("Credit Card Transactions Correlation Matrix", fontsize=15, pad=15)
        
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, "correlation_matrix.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        logger.info(f"Correlation matrix plot saved to: {out_path}")
    except Exception as e:
        logger.error(f"Failed to generate correlation matrix: {e}")


def plot_fraud_distribution(df: pd.DataFrame, output_dir: str = "visuals") -> None:
    """
    Plots a class count distribution (log scale) and a comparison density distribution.
    
    Args:
        df: Input transactional DataFrame.
        output_dir: Directory where the image will be saved.
    """
    logger.info("Generating target class & amount distribution charts...")
    
    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Subplot 1: Count of fraudulent vs legitimate transactions
        class_counts = df["Class"].value_counts()
        sns.barplot(x=class_counts.index, y=class_counts.values, ax=axes[0], palette="Set2")
        axes[0].set_yscale("log")
        axes[0].set_title("Transaction Class Distribution (Log Scale)", fontsize=13)
        axes[0].set_xlabel("Class (0: Legit, 1: Fraud)", fontsize=11)
        axes[0].set_ylabel("Count (Log Scale)", fontsize=11)
        
        total_rows = len(df)
        for idx, count in enumerate(class_counts.values):
            axes[0].text(
                idx, 
                count * 1.2, 
                f"{count}\n({count / total_rows * 100:.4f}%)", 
                ha="center", 
                va="bottom", 
                fontsize=10
            )
            
        # Subplot 2: Capped comparison density of transactions
        sns.kdeplot(
            data=df[df["Class"] == 0], 
            x="Amount", 
            ax=axes[1], 
            label="Legitimate", 
            fill=True, 
            color="blue", 
            alpha=0.3
        )
        sns.kdeplot(
            data=df[df["Class"] == 1], 
            x="Amount", 
            ax=axes[1], 
            label="Fraudulent", 
            fill=True, 
            color="red", 
            alpha=0.4
        )
        axes[1].set_xlim([0, 1000])  # Cap at $1000 to focus on standard ranges
        axes[1].set_title("Transaction Amount Distribution Capped at $1000", fontsize=13)
        axes[1].set_xlabel("Transaction Amount (€)", fontsize=11)
        axes[1].set_ylabel("Density", fontsize=11)
        axes[1].legend()
        
        plt.suptitle("Exploratory Data Analysis - Target Imbalance & Amount Spans", fontsize=15, y=0.98)
        
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, "fraud_distribution.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        logger.info(f"Fraud distribution charts saved to: {out_path}")
    except Exception as e:
        logger.error(f"Failed to generate fraud distribution plots: {e}")
