"""
Configuration loader and validator for the Credit Card Fraud Detection pipeline.
Integrates environment variables (.env) and yaml configurations cleanly.
"""

import os
import yaml
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from dotenv import load_dotenv
from src.exceptions import ConfigurationError

# Load environment variables from .env
load_dotenv()


class ProjectConfig(BaseModel):
    name: str
    version: str
    seed: int


class DirectoriesConfig(BaseModel):
    raw_data_dir: str
    processed_data_dir: str
    models_dir: str
    visuals_dir: str
    reports_dir: str
    logs_dir: str


class DataConfig(BaseModel):
    raw_filename: str
    target_column: str
    test_size: float
    stratify: bool
    scaling_columns: List[str]
    outlier_capping_columns: List[str]
    winsorize_quantile: float


class ModelParamsConfig(BaseModel):
    name: str
    type: str
    params: Dict[str, Any]
    tuning: Optional[Dict[str, Any]] = None


class ModelsConfig(BaseModel):
    baseline: ModelParamsConfig
    champion: ModelParamsConfig


class PipelineConfig(BaseModel):
    project: ProjectConfig
    directories: DirectoriesConfig
    data: DataConfig
    models: ModelsConfig
    
    # Environment overrides
    env: str = "production"
    data_path: str = "data/raw/creditcard.csv"
    api_port: int = 8000
    dashboard_port: int = 8501
    log_level: str = "INFO"
    model_path: str = "models/random_forest_champion.joblib"


def load_config(config_path: str = "config/config.yaml") -> PipelineConfig:
    """
    Loads config.yaml and overrides fields with .env/environment variables.
    
    Args:
        config_path: Relative or absolute path to the config YAML file.
        
    Returns:
        PipelineConfig: The validated configuration object.
        
    Raises:
        ConfigurationError: If loading or validation fails.
    """
    if not os.path.exists(config_path):
        raise ConfigurationError(f"Configuration file not found at: {config_path}")
        
    try:
        with open(config_path, "r") as f:
            raw_yaml = yaml.safe_load(f)
    except Exception as e:
        raise ConfigurationError(f"Failed to parse YAML configuration: {str(e)}")
        
    # Get raw directories and data properties for defaults
    dirs = raw_yaml.get("directories", {})
    data = raw_yaml.get("data", {})
    models = raw_yaml.get("models", {})
    
    raw_data_dir = dirs.get("raw_data_dir", "data/raw")
    raw_filename = data.get("raw_filename", "creditcard.csv")
    models_dir = dirs.get("models_dir", "models")
    champion_name = models.get("champion", {}).get("name", "random_forest_champion")
    
    # Inject env variables
    env_vars = {
        "env": os.getenv("ENV", "production"),
        "data_path": os.getenv("DATA_PATH", f"{raw_data_dir}/{raw_filename}"),
        "api_port": int(os.getenv("API_PORT", "8000")),
        "dashboard_port": int(os.getenv("DASHBOARD_PORT", "8501")),
        "log_level": os.getenv("LOG_LEVEL", "INFO"),
        "model_path": os.getenv("MODEL_PATH", f"{models_dir}/{champion_name}/{champion_name}.joblib"),
    }
    
    # Merge and initialize
    try:
        config = PipelineConfig(
            project=raw_yaml["project"],
            directories=raw_yaml["directories"],
            data=raw_yaml["data"],
            models=raw_yaml["models"],
            **env_vars
        )
        return config
    except Exception as e:
        raise ConfigurationError(f"Validation of configuration settings failed: {str(e)}")
