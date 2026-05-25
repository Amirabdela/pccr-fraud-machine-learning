"""
XGBoost model training module for the Credit Card Fraud Detection pipeline.
Provides a gradient-boosted tree alternative to the Random Forest champion model.
"""

from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

from src.utils import setup_logger
from src.exceptions import ModelTrainingError
from src.config import PipelineConfig

logger = setup_logger()


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor: ColumnTransformer,
    cv: int = 5,
    quick_mode: bool = True,
    random_state: int = 42,
    config: Optional[PipelineConfig] = None,
) -> Pipeline:
    """
    Trains an XGBoost classifier with SMOTE oversampling inside a leakage-free pipeline.

    Args:
        X_train: Training features DataFrame.
        y_train: Training target Series (binary 0/1).
        preprocessor: Unfitted ColumnTransformer for feature scaling.
        cv: Number of stratified CV folds.
        quick_mode: If True, skips hyperparameter tuning.
        random_state: Reproducibility seed.
        config: Optional PipelineConfig for hyperparameter configuration.

    Returns:
        Pipeline: Fitted XGBoost pipeline (preprocessor → SMOTE → XGBClassifier).

    Raises:
        ModelTrainingError: If training fails for any reason.
    """
    logger.info(f"Initializing XGBoost champion training (quick_mode={quick_mode})...")

    try:
        from xgboost import XGBClassifier
    except ImportError as e:
        raise ModelTrainingError(
            "xgboost is not installed. Run: pip install xgboost>=1.5.0"
        ) from e

    try:
        smote = SMOTE(random_state=random_state)

        # Default XGBoost hyperparameters tuned for imbalanced fraud detection
        default_params: Dict[str, Any] = {
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "use_label_encoder": False,
            "eval_metric": "logloss",
            "random_state": random_state,
            "n_jobs": -1,
            "verbosity": 0,
        }

        xgb_clf = XGBClassifier(**default_params)

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("smote", smote),
                ("classifier", xgb_clf),
            ]
        )

        if quick_mode:
            logger.info("Executing fast XGBoost training with default hyperparameters...")
            pipeline.fit(X_train, y_train)
            logger.info("XGBoost model fast training complete.")
            return pipeline

        # Full hyperparameter tuning path
        logger.info(f"Starting Randomized Hyperparameter Search over {cv}-Fold Stratified CV...")

        param_dist = {
            "classifier__n_estimators": [100, 200, 300, 500],
            "classifier__max_depth": [3, 5, 6, 8, 10],
            "classifier__learning_rate": [0.01, 0.05, 0.1, 0.2],
            "classifier__subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
            "classifier__colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
            "classifier__min_child_weight": [1, 3, 5, 7],
            "classifier__gamma": [0, 0.1, 0.2, 0.5],
        }

        cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_dist,
            n_iter=20,
            scoring="f1",
            cv=cv_strategy,
            random_state=random_state,
            n_jobs=-1,
            verbose=1,
        )

        search.fit(X_train, y_train)
        logger.info(f"XGBoost tuning complete. Best params: {search.best_params_}")
        logger.info(f"Best CV F1-score: {search.best_score_:.4f}")
        return search.best_estimator_

    except ModelTrainingError:
        raise
    except Exception as e:
        logger.error(f"XGBoost training failed: {e}")
        raise ModelTrainingError(f"XGBoost training failed: {str(e)}") from e
