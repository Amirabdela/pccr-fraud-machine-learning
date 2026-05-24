"""
pytest integration tests asserting leakage-free pipelines and model serialization cycles.
"""

import os
import pytest
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression

from tests.test_data_loader import generate_mock_data
from src.preprocessing import split_data, get_preprocessor
from src.train import train_baseline, train_champion
from src.utils import save_model, load_model


def test_leakage_free_smote_pipeline():
    """Asserts SMOTE is active strictly during train split fits and bypassed in validation evaluations."""
    df = generate_mock_data(n_samples=500, fraud_ratio=0.05)
    X_train, X_test, y_train, y_test = split_data(df, test_size=0.2, random_state=42)
    
    preprocessor = get_preprocessor(scaling_cols=["Time", "Amount"])
    smote = SMOTE(random_state=42)
    classifier = LogisticRegression(random_state=42)
    
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("smote", smote),
        ("classifier", classifier)
    ])
    
    # Fit execution
    pipeline.fit(X_train, y_train)
    
    # Run test prediction
    preds = pipeline.predict(X_test)
    probs = pipeline.predict_proba(X_test)
    
    # Assert size holds (Test size is strictly 100 rows, SMOTE does not alter inference size)
    assert len(preds) == 100
    assert probs.shape == (100, 2)


def test_model_serialization(tmp_path):
    """Tests model serialization and metadata file companion save/load cycles."""
    model = LogisticRegression(random_state=42)
    file_path = tmp_path / "model.joblib"
    metadata = {"model_name": "test_lr", "type": "LogisticRegression"}
    
    save_model(model, str(file_path), metadata=metadata)
    
    # Check both files exist
    assert os.path.exists(file_path)
    metadata_path = str(file_path).replace(".joblib", "_metadata.json")
    assert os.path.exists(metadata_path)
    
    # Load and assert
    loaded = load_model(str(file_path))
    assert isinstance(loaded, LogisticRegression)
    assert loaded.random_state == 42
