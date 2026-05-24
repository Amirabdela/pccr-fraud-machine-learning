"""
FastAPI high-performance inference microservice.
Offers standard HTTP endpoints for health-checks, single transaction scoring, and batch transaction processing.
"""

import os
import json
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from src.predict import FraudPredictor
from src.exceptions import InferenceError
from src.utils import setup_logger

# Initialize structured logger
logger = setup_logger()

# Instantiate FastAPI application
app = FastAPI(
    title="💳 Credit Card Fraud Detection Real-Time API",
    description="Production-grade FastAPI inference endpoints for fraud risk scoring.",
    version="1.0.0"
)

# Load model path from environment variable with fallback defaults
model_path = os.getenv("MODEL_PATH", "models/random_forest_champion/random_forest_champion.joblib")
predictor: Optional[FraudPredictor] = None

@app.on_event("startup")
def startup_event():
    """Loads the champion model binary on application startup."""
    global predictor
    logger.info("FastAPI service starting up. Ingestion model binary...")
    
    if not os.path.exists(model_path):
        logger.error(f"Cannot initialize API. Serialized model not found at path: {model_path}")
        # We don't crash startup instantly to allow recovery or health checks
        return
        
    try:
        predictor = FraudPredictor(model_path)
        logger.info("Model binary successfully loaded into memory.")
    except Exception as e:
        logger.critical(f"Startup model load crash: {e}")


# --- Pydantic Data Input Schemas ---

class TransactionRecord(BaseModel):
    Time: float = Field(..., description="Seconds elapsed since first transaction.", example=0.0)
    Amount: float = Field(..., description="Transaction amount in Euros.", example=99.99)
    # Optional PCA dimension features with standard defaults (mean=0.0)
    V1: Optional[float] = Field(0.0, description="PCA Component V1")
    V2: Optional[float] = Field(0.0)
    V3: Optional[float] = Field(0.0)
    V4: Optional[float] = Field(0.0)
    V5: Optional[float] = Field(0.0)
    V6: Optional[float] = Field(0.0)
    V7: Optional[float] = Field(0.0)
    V8: Optional[float] = Field(0.0)
    V9: Optional[float] = Field(0.0)
    V10: Optional[float] = Field(0.0)
    V11: Optional[float] = Field(0.0)
    V12: Optional[float] = Field(0.0)
    V13: Optional[float] = Field(0.0)
    V14: Optional[float] = Field(0.0)
    V15: Optional[float] = Field(0.0)
    V16: Optional[float] = Field(0.0)
    V17: Optional[float] = Field(0.0)
    V18: Optional[float] = Field(0.0)
    V19: Optional[float] = Field(0.0)
    V20: Optional[float] = Field(0.0)
    V21: Optional[float] = Field(0.0)
    V22: Optional[float] = Field(0.0)
    V23: Optional[float] = Field(0.0)
    V24: Optional[float] = Field(0.0)
    V25: Optional[float] = Field(0.0)
    V26: Optional[float] = Field(0.0)
    V27: Optional[float] = Field(0.0)
    V28: Optional[float] = Field(0.0)

    class Config:
        schema_extra = {
            "example": {
                "Time": 8432.0,
                "Amount": 120.50,
                "V1": -1.359807,
                "V2": -0.072781,
                "V3": 2.536347,
                "V4": 1.378155,
                "V5": -0.338321,
                "V6": 0.462388,
                "V7": 0.239599,
                "V8": 0.098698,
                "V9": 0.363787,
                "V10": 0.090794,
                "V11": -0.551600,
                "V12": -0.617801,
                "V13": -0.991390,
                "V14": -0.311169,
                "V15": 1.468177,
                "V16": -0.470401,
                "V17": 0.207971,
                "V18": 0.025791,
                "V19": 0.403993,
                "V20": 0.251412,
                "V21": -0.018307,
                "V22": 0.277838,
                "V23": -0.110474,
                "V24": 0.066928,
                "V25": 0.128539,
                "V26": -0.189115,
                "V27": 0.133558,
                "V28": -0.021053
            }
        }


class BatchTransactionRequest(BaseModel):
    transactions: List[TransactionRecord]


# --- API Endpoints ---

@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    """Simple API health check endpoint."""
    if predictor is None:
        return {
            "status": "unhealthy", 
            "message": "Service is running but model binary is not loaded.",
            "model_path": model_path
        }
    return {
        "status": "healthy", 
        "message": "Real-time inference API is fully functional.",
        "model_loaded": True
    }


@app.get("/model/info", status_code=status.HTTP_200_OK)
def model_info():
    """Fetches serialization metadata of the loaded champion model."""
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
            detail="Model is currently not loaded."
        )
        
    metadata_path = model_path.replace(".joblib", "_metadata.json")
    if not os.path.exists(metadata_path):
        return {
            "model_path": model_path,
            "metadata_status": "not_found",
            "message": "Serialized joblib is available but JSON metadata companion is missing."
        }
        
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        return {
            "model_path": model_path,
            "metadata_status": "loaded",
            "metadata": metadata
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to read metadata file: {str(e)}"
        )


@app.post("/predict/single", status_code=status.HTTP_200_OK)
def predict_single_transaction(record: TransactionRecord):
    """Scores a single raw transaction dictionary for fraud risk."""
    global predictor
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference engine model not initialized."
        )
        
    try:
        # Convert Pydantic object to native dict
        tx_dict = record.dict()
        prediction = predictor.predict_single(tx_dict)
        return prediction
    except InferenceError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Inference processing failed: {e.message}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected internal API crash: {str(e)}"
        )


@app.post("/predict/batch", status_code=status.HTTP_200_OK)
def predict_batch_transactions(request: BatchTransactionRequest):
    """Processes bulk list of transactional records for parallel batch scoring."""
    global predictor
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference engine model not initialized."
        )
        
    try:
        # Construct Pandas DataFrame from the list of dict records
        records_list = [tx.dict() for tx in request.transactions]
        df_batch = pd.DataFrame(records_list)
        
        preds, probs = predictor.predict_batch(df_batch)
        
        results = []
        for idx in range(len(preds)):
            results.append({
                "index": idx,
                "prediction_class": int(preds[idx]),
                "prediction_label": "Fraudulent" if preds[idx] == 1 else "Legitimate",
                "fraud_probability": float(probs[idx])
            })
            
        return {
            "total_transactions": len(preds),
            "total_flagged_frauds": int(np.sum(preds)),
            "results": results
        }
    except InferenceError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Batch inference processing failed: {e.message}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected internal API crash: {str(e)}"
        )
