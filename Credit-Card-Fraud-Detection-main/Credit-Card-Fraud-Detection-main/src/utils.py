"""
Utility functions for logging, directory management, and model serialization.
Provides industry-grade logging via logging.config and metadata-accompanied serialization.
"""

import os
import json
import logging
import logging.config
from typing import Any, Dict, Optional
import joblib
import yaml
from datetime import datetime

from src.exceptions import ConfigurationError, InferenceError

def setup_logger(config_path: str = "config/logging_config.yaml") -> logging.Logger:
    """
    Sets up logger configuration using dictConfig from a YAML file.
    Falls back to a standard stream logger if setup fails.
    
    Args:
        config_path: Path to the logging YAML config file.
        
    Returns:
        logging.Logger: Root or specific configured logger instance.
    """
    # Create logs directory first to prevent RotatingFileHandler errors
    os.makedirs("logs", exist_ok=True)
    
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            logging.config.dictConfig(config)
            logger = logging.getLogger("ml_pipeline")
            logger.debug("Successfully configured logging using YAML config.")
            return logger
        except Exception as e:
            # Fallback to basic configuration
            print(f"Warning: Failed to load logging configuration from {config_path}. Error: {e}")
            
    # Basic streaming fallback configuration
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s [%(name)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    logger = logging.getLogger("ml_pipeline_fallback")
    return logger


# Instantiate standard module-level logger
logger = setup_logger()


def ensure_directories(config: Optional[Any] = None) -> None:
    """
    Ensures all necessary directory structures exist in the workspace.
    
    Args:
        config: Optional PipelineConfig to dynamically fetch directories.
    """
    if config is not None:
        try:
            dirs = [
                config.directories.raw_data_dir,
                config.directories.processed_data_dir,
                config.directories.models_dir,
                config.directories.visuals_dir,
                config.directories.reports_dir,
                config.directories.logs_dir
            ]
        except AttributeError as e:
            logger.warning(f"Invalid configuration passed to ensure_directories: {e}. Falling back to default list.")
            dirs = ["data/raw", "data/processed", "models", "visuals", "reports", "logs"]
    else:
        dirs = ["data/raw", "data/processed", "models", "visuals", "reports", "logs"]
        
    for directory in dirs:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Created structural directory: {directory}")


def save_model(model: Any, filepath: str, metadata: Optional[Dict[str, Any]] = None) -> None:
    """
    Serializes a machine learning model/pipeline to disk, saving companion metadata.
    
    Args:
        model: Trained scikit-learn or imblearn pipeline.
        filepath: Target filepath for the serialized joblib file.
        metadata: Optional dictionary of performance metrics, parameters, and version details.
        
    Raises:
        IOError: If serialization fails.
    """
    parent_dir = os.path.dirname(filepath)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
        
    try:
        # Save model pipeline
        joblib.dump(model, filepath)
        logger.info(f"Successfully serialized model pipeline to: {filepath}")
        
        # Save metadata if provided
        if metadata is not None:
            metadata_filepath = os.path.join(
                parent_dir, 
                os.path.basename(filepath).replace(".joblib", "_metadata.json")
            )
            # Add timestamp to metadata
            metadata["saved_at"] = datetime.utcnow().isoformat() + "Z"
            
            with open(metadata_filepath, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=4)
            logger.info(f"Successfully saved model metadata companion to: {metadata_filepath}")
            
    except Exception as e:
        logger.error(f"Failed to serialize model components to {filepath}. Error: {e}")
        raise IOError(f"Model serialization failed: {str(e)}")


def load_model(filepath: str) -> Any:
    """
    Loads a serialized model pipeline from disk.
    
    Args:
        filepath: Target path to the serialized joblib file.
        
    Returns:
        Any: The loaded scikit-learn or imblearn pipeline.
        
    Raises:
        InferenceError: If file not found or loading fails.
    """
    if not os.path.exists(filepath):
        logger.error(f"Target model file not found at: {filepath}")
        raise InferenceError(f"No model found at specified path: {filepath}")
        
    try:
        model = joblib.load(filepath)
        logger.info(f"Loaded serialized model pipeline successfully from: {filepath}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model from {filepath}. Error: {e}")
        raise InferenceError(f"Failed to load serialized model pipeline: {str(e)}")
