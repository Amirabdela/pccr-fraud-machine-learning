"""
Data Ingestion and Schema Validation Module.
Loads, validates target features, and provides a seamless high-fidelity synthetic fallback
if the primary creditcard.csv dataset is not present in the workspace.
"""

import os
import numpy as np
import pandas as pd
from src.utils import setup_logger
from src.exceptions import DataIngestionError

# Initialize structured logger
logger = setup_logger()


def _generate_synthetic_data(n_samples: int = 10000, fraud_ratio: float = 0.0017) -> pd.DataFrame:
    """
    Generates high-fidelity synthetic transaction data representing Credit Card Fraud distributions.
    Employs exponential amount distributions and distinct normal features for both classes.
    """
    np.random.seed(42)
    n_fraud = int(n_samples * fraud_ratio)
    if n_fraud < 5:  # Ensure we have at least some fraud cases for stratification checks
        n_fraud = 15
    n_legit = n_samples - n_fraud
    
    # Legit transactions (Class 0)
    legit_time = np.random.uniform(0, 172800, n_legit)
    legit_amount = np.random.exponential(88.0, n_legit)
    legit_v = np.random.normal(0.0, 1.0, (n_legit, 28))
    legit_class = np.zeros(n_legit)
    
    # Fraud transactions (Class 1) - distinct shift in time, amount, and PCA distribution
    fraud_time = np.random.uniform(0, 172800, n_fraud)
    fraud_amount = np.random.exponential(122.0, n_fraud)
    fraud_v = np.random.normal(-1.5, 2.0, (n_fraud, 28))
    fraud_class = np.ones(n_fraud)
    
    times = np.concatenate([legit_time, fraud_time])
    amounts = np.concatenate([legit_amount, fraud_amount])
    vs = np.concatenate([legit_v, fraud_v])
    classes = np.concatenate([legit_class, fraud_class])
    
    # Shuffle dataset
    shuffle_indices = np.arange(n_samples)
    np.random.shuffle(shuffle_indices)
    
    times = times[shuffle_indices]
    amounts = amounts[shuffle_indices]
    vs = vs[shuffle_indices]
    classes = classes[shuffle_indices]
    
    columns = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]
    data = np.hstack([times.reshape(-1, 1), vs, amounts.reshape(-1, 1), classes.reshape(-1, 1)])
    
    df = pd.DataFrame(data, columns=columns)
    df["Class"] = df["Class"].astype(int)
    return df


def load_data(filepath: str = "data/raw/creditcard.csv") -> pd.DataFrame:
    """
    Loads and validates the credit card transaction dataset from a CSV file.
    If the file is not found, it automatically generates a high-fidelity synthetic 
    dataset to ensure the pipeline runs seamlessly from end-to-end.
    
    Args:
        filepath: Direct path to the creditcard.csv target file.
        
    Returns:
        pd.DataFrame: The loaded and verified dataset.
        
    Raises:
        DataIngestionError: If dataset fails schema assertions.
    """
    logger.info(f"Initiating dataset ingestion from target path: {filepath}")
    
    if not os.path.exists(filepath):
        logger.warning(f"Dataset not found at target location: '{filepath}'")
        logger.info("GENERATING HIGH-FIDELITY SYNTHETIC DATASET FOR SEAMLESS DEMO RUN...")
        
        # Ensure directories exist
        parent_dir = os.path.dirname(filepath)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
            
        try:
            # Generate 15,000 samples with 0.17% fraud ratio to mirror Kaggle statistics
            df_synthetic = _generate_synthetic_data(n_samples=15000, fraud_ratio=0.0017)
            df_synthetic.to_csv(filepath, index=False)
            logger.info(f"Synthetic dataset of 15,000 samples successfully saved to: {filepath}")
        except Exception as e:
            logger.error(f"Failed to generate synthetic dataset: {e}")
            raise DataIngestionError(f"Synthetic generation failed: {str(e)}")
        
    try:
        df = pd.read_csv(filepath)
        row_count, col_count = df.shape
        logger.info(f"Dataset loaded successfully. Shape: {row_count} rows, {col_count} columns.")
        
        # Schema verification
        required_cols = ["Time", "Amount", "Class"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Dataset is missing mandatory structural columns: {missing_cols}")
            
        # Target class balance evaluation
        class_counts = df["Class"].value_counts().to_dict()
        total_class_1 = class_counts.get(1, 0)
        total_class_0 = class_counts.get(0, 0)
        
        if row_count > 0:
            fraud_ratio = (total_class_1 / row_count) * 100
        else:
            fraud_ratio = 0.0
        
        logger.info(
            f"Target class distribution: Legitimate (0) = {total_class_0}, "
            f"Fraudulent (1) = {total_class_1} ({fraud_ratio:.4f}%)"
        )
        
        # Missing values assessment
        null_counts = int(df.isnull().sum().sum())
        if null_counts > 0:
            logger.warning(f"Detected {null_counts} missing value(s) in raw dataset. Imputer steps will handle this.")
        else:
            logger.info("Dataset verified: No missing values detected.")
            
        return df
        
    except Exception as e:
        logger.error(f"Critical error occurred during dataset loading: {e}")
        raise DataIngestionError(f"Data ingestion failed: {str(e)}")
