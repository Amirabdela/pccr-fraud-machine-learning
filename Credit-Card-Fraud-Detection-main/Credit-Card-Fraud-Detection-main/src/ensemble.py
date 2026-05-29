"""
Ensemble Voting Classifier module.
Combines Logistic Regression, Random Forest, and XGBoost into a soft-voting ensemble
for improved fraud detection robustness.
"""

from typing import Optional, List, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SklearnPipeline
from imblearn.pipeline import Pipeline

from src.utils import setup_logger
from src.exceptions import ModelTrainingError

logger = setup_logger()


def train_voting_ensemble(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor: ColumnTransformer,
    random_state: int = 42,
    weights: Optional[List[float]] = None,
) -> SklearnPipeline:
    """
    Trains a soft-voting ensemble of Logistic Regression, Random Forest, and XGBoost.

    The preprocessor is applied once outside the VotingClassifier to avoid
    redundant transformations. Each base estimator receives the same scaled features.

    Args:
        X_train: Raw training features DataFrame.
        y_train: Training labels Series.
        preprocessor: Fitted or unfitted ColumnTransformer.
        random_state: Reproducibility seed.
        weights: Optional per-model weights [lr_weight, rf_weight, xgb_weight].
                 Defaults to equal weighting [1, 1, 1].

    Returns:
        SklearnPipeline: Fitted pipeline (preprocessor → VotingClassifier).

    Raises:
        ModelTrainingError: If ensemble training fails.
    """
    logger.info("Initializing soft-voting ensemble (LR + RF + XGBoost)...")

    try:
        try:
            from xgboost import XGBClassifier
            xgb_available = True
        except ImportError:
            logger.warning("XGBoost not available; ensemble will use LR + RF only.")
            xgb_available = False

        if weights is None:
            weights = [1, 2, 2] if xgb_available else [1, 2]

        # Define base estimators
        lr = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=random_state,
            solver="lbfgs",
        )
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            random_state=random_state,
            n_jobs=-1,
        )

        estimators = [("logistic_regression", lr), ("random_forest", rf)]

        if xgb_available:
            xgb = XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=random_state,
                n_jobs=-1,
                verbosity=0,
            )
            estimators.append(("xgboost", xgb))

        voting_clf = VotingClassifier(
            estimators=estimators,
            voting="soft",
            weights=weights,
            n_jobs=-1,
        )

        pipeline = SklearnPipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("ensemble", voting_clf),
            ]
        )

        logger.info(
            f"Training ensemble with estimators: "
            f"{[name for name, _ in estimators]} | weights={weights}"
        )
        pipeline.fit(X_train, y_train)
        logger.info("Soft-voting ensemble training complete.")
        return pipeline

    except ModelTrainingError:
        raise
    except Exception as e:
        logger.error(f"Ensemble training failed: {e}")
        raise ModelTrainingError(f"Ensemble training failed: {str(e)}") from e


def get_ensemble_member_predictions(
    pipeline: SklearnPipeline,
    X: pd.DataFrame,
) -> pd.DataFrame:
    """
    Extract individual member predictions from a fitted VotingClassifier pipeline.

    Args:
        pipeline: Fitted pipeline containing a VotingClassifier step named 'ensemble'.
        X: Feature DataFrame (raw, pre-transform).

    Returns:
        pd.DataFrame: Per-estimator fraud probabilities + final ensemble probability.
    """
    try:
        preprocessor = pipeline.named_steps["preprocessor"]
        ensemble = pipeline.named_steps["ensemble"]

        X_transformed = preprocessor.transform(X)
        if hasattr(X_transformed, "toarray"):
            X_transformed = X_transformed.toarray()

        results = {}
        for name, estimator in ensemble.estimators_:
            proba = estimator.predict_proba(X_transformed)[:, 1]
            results[f"{name}_fraud_proba"] = proba

        results["ensemble_fraud_proba"] = pipeline.predict_proba(X)[:, 1]
        results["ensemble_prediction"] = pipeline.predict(X)

        return pd.DataFrame(results)

    except Exception as e:
        logger.error(f"Failed to extract member predictions: {e}")
        raise ModelTrainingError(f"Member prediction extraction failed: {str(e)}") from e
