"""
pytest unit tests for data loader schema checks and warnings.
"""

import os
import pandas as pd
import numpy as np
import pytest

from src.data_loader import load_data
from src.exceptions import DataIngestionError


def generate_mock_data(n_samples: int = 100, fraud_ratio: float = 0.05) -> pd.DataFrame:
    """Helper to generate synthetic credit card dataset shapes."""
    np.random.seed(42)
    n_fraud = int(n_samples * fraud_ratio)
    n_legit = n_samples - n_fraud
    
    times = np.random.uniform(0, 100000, n_samples)
    amounts = np.random.exponential(50, n_samples)
    classes = np.concatenate([np.zeros(n_legit), np.ones(n_fraud)])
    np.random.shuffle(classes)
    
    vs = np.random.normal(0, 1, (n_samples, 28))
    columns = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]
    data = np.hstack([times.reshape(-1, 1), vs, amounts.reshape(-1, 1), classes.reshape(-1, 1)])
    
    df = pd.DataFrame(data, columns=columns)
    df["Class"] = df["Class"].astype(int)
    return df


def test_data_loader_valid(tmp_path):
    """Verifies successful load of a schema-compliant CSV."""
    df_mock = generate_mock_data(200, 0.1)
    file_path = tmp_path / "creditcard.csv"
    df_mock.to_csv(file_path, index=False)
    
    df = load_data(str(file_path))
    assert df.shape == (200, 31)
    assert "Class" in df.columns
    assert int(df["Class"].sum()) == 20


def test_data_loader_missing_file(tmp_path):
    """Asserts that calling load_data on a non-existent path generates the synthetic data file."""
    file_path = tmp_path / "synthetic_creditcard.csv"
    assert not os.path.exists(file_path)
    
    df = load_data(str(file_path))
    assert os.path.exists(file_path)
    assert df.shape == (15000, 31)


def test_data_loader_malformed_schema(tmp_path):
    """Verifies that missing structural columns raise ValueError inside load."""
    df_bad = pd.DataFrame({"V1": [1.0, 2.0], "Class": [0, 1]})
    file_path = tmp_path / "bad_schema.csv"
    df_bad.to_csv(file_path, index=False)
    
    with pytest.raises(DataIngestionError) as exc_info:
        load_data(str(file_path))
    assert "missing" in str(exc_info.value)
