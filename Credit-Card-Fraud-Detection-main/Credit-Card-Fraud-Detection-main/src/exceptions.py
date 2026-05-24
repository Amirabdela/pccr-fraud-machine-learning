"""
Custom exceptions for the Credit Card Fraud Detection pipeline.
Allows granular error handling and reporting across the codebase.
"""

class FraudPipelineError(Exception):
    """Base exception class for all fraud detection pipeline errors."""
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class ConfigurationError(FraudPipelineError):
    """Raised when there is an issue loading environment variables or YAML configs."""
    pass


class DataIngestionError(FraudPipelineError):
    """Raised when data loading, schema checking, or ingestion fails."""
    pass


class PreprocessingError(FraudPipelineError):
    """Raised when scaling, Winsorization, split, or other pipeline processing steps fail."""
    pass


class ModelTrainingError(FraudPipelineError):
    """Raised when model training, CV search, or baseline/champion fitting fails."""
    pass


class InferenceError(FraudPipelineError):
    """Raised when model loading or batch/single prediction scoring fails."""
    pass
