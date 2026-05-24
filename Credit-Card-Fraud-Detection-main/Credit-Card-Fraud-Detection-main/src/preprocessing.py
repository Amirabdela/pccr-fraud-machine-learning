"""
Data Preprocessing Pipeline Module.
Implements missing value imputation, winsorization for outlier handling, and column transformations.
"""

from typing import Tuple, List, Union
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.utils import setup_logger
from src.exceptions import PreprocessingError

# Initialize structured logger
logger = setup_logger()


def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handles missing values in features. Production environments require resilience
    to missing fields, even if the research dataset is complete.
    
    Args:
        df: Input pandas DataFrame.
        
    Returns:
        pd.DataFrame: Imputed DataFrame.
        
    Raises:
        PreprocessingError: If input is not a DataFrame or imputation fails.
    """
    if not isinstance(df, pd.DataFrame):
        raise PreprocessingError("Input to impute_missing_values must be a pandas DataFrame.")
        
    logger.info("Evaluating missing values across features...")
    null_count = int(df.isnull().sum().sum())
    
    try:
        if null_count > 0:
            logger.warning(f"Detected {null_count} missing value(s) in data. Performing median imputation.")
            df_imputed = df.copy()
            numeric_cols = df_imputed.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df_imputed[col].isnull().any():
                    median_val = df_imputed[col].median()
                    df_imputed[col] = df_imputed[col].fillna(median_val)
            logger.info("Missing value imputation completed.")
            return df_imputed
        else:
            logger.info("No missing values detected. Skipping imputation.")
            return df
    except Exception as e:
        logger.error(f"Error during missing value imputation: {e}")
        raise PreprocessingError(f"Imputation failed: {str(e)}")


def handle_outliers(
    df: pd.DataFrame, 
    columns: List[str] = ["Amount"], 
    threshold_percentile: float = 0.999
) -> pd.DataFrame:
    """
    Caps extreme feature values using percentile thresholds (Winsorization).
    Prevents large outlier variances from compressing scaled features.
    
    Args:
        df: Input pandas DataFrame.
        columns: Target columns for Winsorization.
        threshold_percentile: Upper percentile threshold for outlier capping.
        
    Returns:
        pd.DataFrame: DataFrame with capped outliers.
        
    Raises:
        PreprocessingError: If capping operation fails.
    """
    if not isinstance(df, pd.DataFrame):
        raise PreprocessingError("Input to handle_outliers must be a pandas DataFrame.")
        
    logger.info(f"Checking outliers for columns: {columns} at {threshold_percentile} threshold.")
    
    try:
        df_capped = df.copy()
        for col in columns:
            if col in df_capped.columns:
                cap_val = float(df_capped[col].quantile(threshold_percentile))
                outlier_mask = df_capped[col] > cap_val
                outlier_count = int(outlier_mask.sum())
                
                if outlier_count > 0:
                    logger.info(
                        f"Capping {outlier_count} extreme outlier(s) in '{col}' at "
                        f"{threshold_percentile} percentile value: {cap_val:.2f}"
                    )
                    df_capped.loc[outlier_mask, col] = cap_val
            else:
                logger.warning(f"Capping requested for missing column: '{col}'")
                
        return df_capped
    except Exception as e:
        logger.error(f"Error capping outliers: {e}")
        raise PreprocessingError(f"Outlier handling failed: {str(e)}")


def split_data(
    df: pd.DataFrame,
    target_column: str = "Class",
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Extracts features and target labels and partitions them into train and test splits.
    
    Args:
        df: Input transaction dataset.
        target_column: Target class label.
        test_size: Split ratio for testing.
        random_state: Random state seed.
        stratify: Whether to stratify target balances.
        
    Returns:
        Tuple: X_train, X_test, y_train, y_test.
        
    Raises:
        PreprocessingError: If target_column is missing or split operations fail.
    """
    if target_column not in df.columns:
        raise PreprocessingError(f"Target column '{target_column}' not found in DataFrame columns.")
        
    logger.info(f"Executing train/test split (test_size={test_size}, stratify={stratify})")
    
    try:
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        stratify_array = y if stratify else None
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            stratify=stratify_array,
            random_state=random_state
        )
        
        logger.info(f"Split completed. Train shape: {X_train.shape}, Test shape: {X_test.shape}")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logger.error(f"Failed to split data: {e}")
        raise PreprocessingError(f"Data partitioning failed: {str(e)}")


def get_preprocessor(scaling_cols: List[str] = ["Time", "Amount"]) -> ColumnTransformer:
    """
    Constructs a ColumnTransformer mapping imputation and scaling transforms onto specified columns.
    
    Args:
        scaling_cols: Column names requiring imputation and standard scaling.
        
    Returns:
        ColumnTransformer: Preprocessing transformer pipeline.
    """
    logger.info(f"Configuring preprocessor pipeline targeting scaling columns: {scaling_cols}")
    
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, scaling_cols)
        ],
        remainder="passthrough"
    )
    
    return preprocessor
