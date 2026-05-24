"""
pytest unit tests for data splitting, outlier capping, and feature scaling preprocessors.
"""

import pandas as pd
import numpy as np
import pytest

from tests.test_data_loader import generate_mock_data
from src.preprocessing import split_data, handle_outliers, impute_missing_values, get_preprocessor


def test_split_data_stratification():
    """Asserts train/test split maintains exact target balance proportions."""
    df = generate_mock_data(n_samples=1000, fraud_ratio=0.02)  # 20 fraud cases
    X_train, X_test, y_train, y_test = split_data(
        df, 
        target_column="Class", 
        test_size=0.2, 
        random_state=42
    )
    
    assert X_train.shape[0] == 800
    assert X_test.shape[0] == 200
    
    # 20 * 0.8 = 16 frauds in train, 20 * 0.2 = 4 frauds in test
    assert int(y_train.sum()) == 16
    assert int(y_test.sum()) == 4


def test_handle_outliers_capping():
    """Verifies that extreme values are capped at the 99.9th percentile."""
    data = {"Amount": [1.0] * 999 + [10000.0]}  # 99.9th percentile is 1.0
    df = pd.DataFrame(data)
    
    df_capped = handle_outliers(df, columns=["Amount"], threshold_percentile=0.999)
    assert float(df_capped["Amount"].max()) < 10000.0
    assert float(df_capped["Amount"].max()) == float(df["Amount"].quantile(0.999))


def test_preprocessing_transformer():
    """Confirms scaler standardizes targeted attributes and leaves rest untouched."""
    df = generate_mock_data(n_samples=500, fraud_ratio=0.04)
    X = df.drop(columns=["Class"])
    
    preprocessor = get_preprocessor(scaling_cols=["Time", "Amount"])
    X_trans = preprocessor.fit_transform(X)
    
    assert X_trans.shape == X.shape
    # Standard remainder "passthrough" columns start at index 2 (V1 is index 2)
    # Check that V1 values are identical in both raw and transformed states
    assert np.allclose(X_trans[:, 2], X["V1"].values)
