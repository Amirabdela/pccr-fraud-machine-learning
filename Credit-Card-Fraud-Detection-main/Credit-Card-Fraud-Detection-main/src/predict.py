"""
Production inference wrapper module.
Provides FraudPredictor for batch and real-time single transaction scoring with robust validation.
"""

from typing import Dict, Any, Tuple, Union, List
import pandas as pd
import numpy as np

from src.utils import load_model, setup_logger
from src.exceptions import InferenceError

# Initialize structured logger
logger = setup_logger()


class FraudPredictor:
    """
    Inference interface that loads a saved model pipeline and manages
    preprocessing, schema validation, and prediction scoring.
    """
    
    def __init__(self, model_path: str):
        """
        Args:
            model_path: Relative or absolute path to the serialized model file.
            
        Raises:
            InferenceError: If the model cannot be loaded.
        """
        logger.info(f"Initializing FraudPredictor with model binary: {model_path}")
        try:
            self.model = load_model(model_path)
            # Define expected PCA + metadata feature schema (Time, V1-V28, Amount)
            self.expected_features: List[str] = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
        except Exception as e:
            logger.error(f"FraudPredictor failed to initialize: {e}")
            raise InferenceError(f"Predictor initialization failed: {str(e)}")
            
    def _validate_and_reorder(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validates presence of columns and reorders them to match training schema.
        Fills missing V columns with 0.0 default.
        """
        df_processed = df.copy()
        
        # Remove target column if present
        if "Class" in df_processed.columns:
            df_processed = df_processed.drop(columns=["Class"])
            
        # Impute missing PCA components or columns
        for col in self.expected_features:
            if col not in df_processed.columns:
                logger.warning(f"Inference record missing feature '{col}'. Defaulting to 0.0.")
                df_processed[col] = 0.0
                
        # Reorder to match exact feature training sequence
        return df_processed[self.expected_features]

    def predict_batch(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Runs predictions on a DataFrame batch of raw credit card transactions.
        
        Args:
            df: Input DataFrame containing transactional features.
            
        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - Class predictions (0 = Legit, 1 = Fraud)
                - Fraud probability estimates
                
        Raises:
            InferenceError: If prediction operations fail.
        """
        if not isinstance(df, pd.DataFrame):
            raise InferenceError("Input for batch prediction must be a pandas DataFrame.")
            
        logger.info(f"Executing batch prediction on {df.shape[0]} transactions...")
        
        try:
            features = self._validate_and_reorder(df)
            preds = self.model.predict(features)
            
            if hasattr(self.model, "predict_proba"):
                probs = self.model.predict_proba(features)[:, 1]
            else:
                logger.warning("Loaded pipeline does not support probability output.")
                probs = preds.astype(float)
                
            return preds, probs
            
        except Exception as e:
            logger.error(f"Batch prediction failed: {e}")
            raise InferenceError(f"Batch inference failed: {str(e)}")

    def predict_single(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predicts classification outcomes for a single transaction.
        
        Args:
            record: Dictionary containing feature mapping keys.
            
        Returns:
            Dict[str, Any]: Dict containing predictions, labels, and probabilities.
            
        Raises:
            InferenceError: If single-instance prediction fails.
        """
        if not isinstance(record, dict):
            raise InferenceError("Input for single-instance prediction must be a dictionary.")
            
        try:
            # Convert single dictionary row to DataFrame
            df_row = pd.DataFrame([record])
            preds, probs = self.predict_batch(df_row)
            
            pred_class = int(preds[0])
            confidence = float(probs[0])
            
            result = {
                "prediction_class": pred_class,
                "prediction_label": "Fraudulent" if pred_class == 1 else "Legitimate",
                "fraud_probability": confidence,
                "legitimate_probability": 1.0 - confidence
            }
            
            logger.info(
                f"Single transaction inference successful. Decision: {result['prediction_label']} "
                f"(Fraud Prob: {confidence:.4f})"
            )
            return result
            
        except Exception as e:
            logger.error(f"Single prediction failed: {e}")
            raise InferenceError(f"Single inference failed: {str(e)}")
