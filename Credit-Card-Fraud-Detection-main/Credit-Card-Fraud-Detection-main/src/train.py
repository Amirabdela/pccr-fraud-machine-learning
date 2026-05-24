"""
Model training and hyperparameter search module.
Constructs, optimized, and cross-validates baseline and champion modeling pipelines.
"""

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

from src.utils import setup_logger
from src.exceptions import ModelTrainingError
from src.config import PipelineConfig

# Initialize structured logger
logger = setup_logger()


def train_baseline(
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    preprocessor: ColumnTransformer,
    config: Optional[PipelineConfig] = None
) -> Pipeline:
    """
    Constructs and fits a scaled Logistic Regression baseline model.
    
    Args:
        X_train: Training features DataFrame.
        y_train: Training target labels Series.
        preprocessor: Unfitted ColumnTransformer for scaling.
        config: Optional PipelineConfig object to read model params.
        
    Returns:
        Pipeline: Fitted baseline pipeline.
        
    Raises:
        ModelTrainingError: If training fails.
    """
    logger.info("Initializing and training Logistic Regression baseline model...")
    
    try:
        # Load hyperparameters from config if available
        if config is not None:
            params = config.models.baseline.params
        else:
            params = {
                "class_weight": "balanced",
                "max_iter": 1000,
                "random_state": 42,
                "solver": "lbfgs"
            }
            
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(**params))
        ])
        
        pipeline.fit(X_train, y_train)
        logger.info("Logistic Regression baseline model training completed successfully.")
        return pipeline
        
    except Exception as e:
        logger.error(f"Error during baseline model training: {e}")
        raise ModelTrainingError(f"Baseline training failed: {str(e)}")


def train_champion(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor: ColumnTransformer,
    cv: int = 5,
    quick_mode: bool = True,
    random_state: int = 42,
    config: Optional[PipelineConfig] = None
) -> Pipeline:
    """
    Constructs and trains a SMOTE-oversampled Random Forest champion model.
    Prevents data leakage by encapsulating oversampling inside CV folds.
    
    Args:
        X_train: Training features DataFrame.
        y_train: Training target labels Series.
        preprocessor: Unfitted ColumnTransformer for scaling.
        cv: Folds to apply in Stratified Cross Validation.
        quick_mode: If True, skips hyperparameter tuning and fits with default config.
        random_state: Reproducibility seed.
        config: Optional PipelineConfig to read params and search spaces.
        
    Returns:
        Pipeline: Fitted champion pipeline.
        
    Raises:
        ModelTrainingError: If training or hyperparameter search fails.
    """
    logger.info(f"Initializing SMOTE & Random Forest champion training (quick_mode={quick_mode})...")
    
    try:
        smote = SMOTE(random_state=random_state)
        
        # Load hyperparameters and search spaces
        if config is not None:
            rf_params = config.models.champion.params
            tuning_config = config.models.champion.tuning
            cv_folds = tuning_config.get("cv_folds", cv) if tuning_config else cv
            n_iter = tuning_config.get("n_iter", 10) if tuning_config else 10
            
            # Map parameters from configuration
            # To ensure standard compatibility, extract only base parameters for RandomForestClassifier
            rf_base_params = {
                "random_state": rf_params.get("random_state", random_state),
                "n_jobs": rf_params.get("n_jobs", -1)
            }
        else:
            rf_base_params = {
                "random_state": random_state,
                "n_jobs": -1
            }
            cv_folds = cv
            n_iter = 10
            
        rf_base = RandomForestClassifier(**rf_base_params)
        
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("smote", smote),
            ("classifier", rf_base)
        ])
        
        if quick_mode:
            logger.info("Executing fast champion training with baseline hyperparameters...")
            if config is not None:
                # Set baseline parameters
                baseline_rf_params = {
                    f"classifier__{k}": v for k, v in config.models.champion.params.items()
                    if k not in ["n_jobs"]
                }
                pipeline.set_params(**baseline_rf_params)
            else:
                pipeline.set_params(
                    classifier__n_estimators=100,
                    classifier__max_depth=10,
                    classifier__min_samples_split=5
                )
            
            pipeline.fit(X_train, y_train)
            logger.info("Champion Random Forest model fast training complete.")
            return pipeline
            
        else:
            logger.info(f"Starting Randomized Hyperparameter Search over stratified {cv_folds}-Fold CV...")
            
            # Map search space
            if config is not None and config.models.champion.tuning is not None:
                param_dist = {
                    f"classifier__{k}": v for k, v in config.models.champion.tuning.get("param_distributions", {}).items()
                }
            else:
                param_dist = {
                    "classifier__n_estimators": [50, 100, 200],
                    "classifier__max_depth": [5, 10, 15, None],
                    "classifier__min_samples_split": [2, 5, 10],
                    "classifier__min_samples_leaf": [1, 2, 4]
                }
                
            cv_strategy = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
            
            search = RandomizedSearchCV(
                estimator=pipeline,
                param_distributions=param_dist,
                n_iter=n_iter,
                scoring="f1",
                cv=cv_strategy,
                random_state=random_state,
                n_jobs=-1,
                verbose=1
            )
            
            search.fit(X_train, y_train)
            logger.info(f"Hyperparameter optimization complete. Best parameters: {search.best_params_}")
            logger.info(f"Best cross-validation F1-score: {search.best_score_:.4f}")
            return search.best_estimator_
            
    except Exception as e:
        logger.error(f"Error during champion model training: {e}")
        raise ModelTrainingError(f"Champion training failed: {str(e)}")


def evaluate_with_cross_val_score(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: int = 5,
    scoring: str = "f1",
    random_state: int = 42
) -> np.ndarray:
    """
    Computes cross-validation scores using cross_val_score across stratified folds.
    
    Args:
        pipeline: Target evaluation Pipeline.
        X_train: Training features DataFrame.
        y_train: Training target Series.
        cv: Split folds.
        scoring: Scikit-learn scoring string.
        random_state: Random state seed.
        
    Returns:
        np.ndarray: Computed scores across folds.
        
    Raises:
        ModelTrainingError: If cross-validation scoring fails.
    """
    logger.info(f"Computing cross-validation scores (Folds: {cv}, Metric: {scoring})...")
    
    try:
        cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
        scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=cv_strategy,
            scoring=scoring,
            n_jobs=-1
        )
        
        logger.info(f"Cross-validation {scoring} scores across folds: {scores}")
        logger.info(f"Mean {scoring} score: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        return scores
    except Exception as e:
        logger.error(f"Cross-validation evaluation failed: {e}")
        raise ModelTrainingError(f"Cross-validation score computation failed: {str(e)}")
