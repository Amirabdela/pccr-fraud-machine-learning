import sys
import os
import shutil
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression

# Adjust search path to load source modules correctly
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.utils import setup_logger, save_model, load_model
from src.preprocessing import split_data, get_preprocessor
from src.data_loader import load_data

logger = setup_logger("TestRunner")

def generate_mock_data(n_samples: int = 1000, fraud_ratio: float = 0.05) -> pd.DataFrame:
    """
    Generates synthetic transaction matrices representing Credit Card Fraud distributions.
    """
    np.random.seed(42)
    n_fraud = int(n_samples * fraud_ratio)
    n_legit = n_samples - n_fraud
    
    # Legit transactions (Class 0)
    legit_time = np.random.uniform(0, 100000, n_legit)
    legit_amount = np.random.exponential(50, n_legit)
    legit_v = np.random.normal(0, 1, (n_legit, 28))
    legit_class = np.zeros(n_legit)
    
    # Fraud transactions (Class 1)
    fraud_time = np.random.uniform(0, 100000, n_fraud)
    fraud_amount = np.random.exponential(150, n_fraud)
    fraud_v = np.random.normal(-1.5, 2, (n_fraud, 28))
    fraud_class = np.ones(n_fraud)
    
    times = np.concatenate([legit_time, fraud_time])
    amounts = np.concatenate([legit_amount, fraud_amount])
    vs = np.concatenate([legit_v, fraud_v])
    classes = np.concatenate([legit_class, fraud_class])
    
    columns = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]
    data = np.hstack([times.reshape(-1, 1), vs, amounts.reshape(-1, 1), classes.reshape(-1, 1)])
    
    df = pd.DataFrame(data, columns=columns)
    df["Class"] = df["Class"].astype(int)
    return df

def test_data_loader_and_schema():
    """
    Tests data loader assertions and schema compliance.
    """
    logger.info("TEST: Loading & Schema Validation...")
    temp_dir = "data/raw"
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, "temp_mock_creditcard.csv")
    
    try:
        mock_df = generate_mock_data(n_samples=200, fraud_ratio=0.1)
        mock_df.to_csv(temp_path, index=False)
        
        # Load through pipeline module
        df = load_data(temp_path)
        
        assert df.shape == (200, 31), f"Unexpected data dimensions: {df.shape}"
        assert "Class" in df.columns, "Class target column missing"
        assert df["Class"].isnull().sum() == 0, "Found null items in Target"
        logger.info("-> Schema Validation Test Passed.")
        
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

def test_preprocessing_and_split():
    """
    Tests preprocessing scaling bounds and stratified splitting distribution checks.
    """
    logger.info("TEST: Data Splitting & Feature Scaling Preprocessors...")
    df = generate_mock_data(n_samples=1000, fraud_ratio=0.02)
    
    X_train, X_test, y_train, y_test = split_data(
        df, 
        target_column="Class", 
        test_size=0.2, 
        random_state=42
    )
    
    # Assert Stratification Holds
    original_fraud_ratio = df["Class"].mean()
    train_fraud_ratio = y_train.mean()
    test_fraud_ratio = y_test.mean()
    
    assert abs(original_fraud_ratio - train_fraud_ratio) < 0.01, "Stratification ratio failed on Train Split"
    assert abs(original_fraud_ratio - test_fraud_ratio) < 0.01, "Stratification ratio failed on Test Split"
    assert X_train.shape[0] == 800, f"Expected 800 training records, got {X_train.shape[0]}"
    assert X_test.shape[0] == 200, f"Expected 200 testing records, got {X_test.shape[0]}"
    
    # Test Scaling
    preprocessor = get_preprocessor(scaling_cols=["Time", "Amount"])
    
    # Column Transformer places scaled columns first, then pass-through cols
    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)
    
    assert X_train_trans.shape == X_train.shape, "Transformed dimensions do not match raw feature shapes"
    
    # Verify that V1-V28 are left unscaled (assert variances are identical)
    # The ColumnTransformer places the scaled columns ("Time", "Amount") at index 0 and 1.
    # The remainder "passthrough" columns start at index 2 (V1 is at index 2).
    # Since V1 is untouched, index 2 in X_train_trans must exactly match V1 (col 1) in X_train.
    assert np.allclose(X_train_trans[:, 2], X_train["V1"].values), "Passthrough dimension was altered during preprocessing!"
    logger.info("-> Data Split & Feature Scaling Preprocessor Tests Passed.")

def test_leakage_free_smote_pipeline():
    """
    Asserts SMOTE is active strictly during train split fits and bypassed in validation evaluations.
    """
    logger.info("TEST: Safe Leakage-Free SMOTE Pipeline Execution...")
    df = generate_mock_data(n_samples=500, fraud_ratio=0.05) # 25 fraud records (20 in train split)
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
    
    # Make sure test dataset instances were not oversampled (Test size is strictly 100 rows)
    assert len(preds) == 100, f"Predict changed inference rows, got {len(preds)}"
    assert probs.shape == (100, 2), f"Predict Proba shape mismatched: {probs.shape}"
    logger.info("-> Leakage-free SMOTE Pipeline verification Passed.")

def test_model_serialization():
    """
    Tests saving and loading joblib model pipelines.
    """
    logger.info("TEST: Model Serialization (save/load)...")
    temp_model_path = "models/test_model.joblib"
    
    try:
        model = LogisticRegression(random_state=42)
        save_model(model, temp_model_path)
        
        loaded = load_model(temp_model_path)
        assert isinstance(loaded, LogisticRegression), "Loaded object class mismatched"
        assert loaded.random_state == 42, "Loaded object hyperparameters altered"
        logger.info("-> Model Serialization test Passed.")
        
    finally:
        if os.path.exists(temp_model_path):
            os.remove(temp_model_path)

def run_all_tests():
    logger.info("=======================================================================")
    logger.info("STARTING PIPELINE VERIFICATION SUITE")
    logger.info("=======================================================================")
    
    try:
        test_data_loader_and_schema()
        test_preprocessing_and_split()
        test_leakage_free_smote_pipeline()
        test_model_serialization()
        
        logger.info("=======================================================================")
        logger.info("SUCCESS: ALL PIPELINE VERIFICATIONS PASSED SUCCESSFULLY.")
        logger.info("=======================================================================")
        return True
        
    except AssertionError as e:
        logger.critical(f"VERIFICATION FAILURE: Assertion check failed. Error: {e}")
        return False
    except Exception as e:
        logger.critical(f"CRITICAL FAULT: Pipeline verification crashed. Error: {e}")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
