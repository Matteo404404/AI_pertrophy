"""
Logging Utilities for ML Module

Centralized logging system for model training, predictions, and debugging
"""

import logging
import os
from datetime import datetime
from typing import Optional

def setup_logger(name: str, 
                 log_file: Optional[str] = None, 
                 level: str = 'INFO') -> logging.Logger:
    """
    Set up logger for ML module
    
    Args:
        name: Logger name (usually __name__)
        log_file: Optional log file path
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    
    # Don't add handlers if already configured
    if logger.handlers:
        return logger
        
    logger.setLevel(getattr(logging, level.upper()))
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        # Create log directory if it doesn't exist
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    return logger


def log_model_training(logger: logging.Logger, 
                      model_name: str, 
                      training_data_shape: tuple,
                      metrics: dict) -> None:
    """
    Log model training information
    
    Args:
        logger: Logger instance
        model_name: Name of model being trained
        training_data_shape: Shape of training data (samples, features)
        metrics: Training metrics dictionary
    """
    logger.info(f"🚂 Training {model_name}")
    logger.info(f"📊 Training data shape: {training_data_shape}")
    
    for metric, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"📈 {metric}: {value:.4f}")
        else:
            logger.info(f"📈 {metric}: {value}")
            

def log_prediction(logger: logging.Logger,
                  model_name: str,
                  input_shape: tuple,
                  predictions: dict) -> None:
    """
    Log prediction information
    
    Args:
        logger: Logger instance
        model_name: Name of model making prediction
        input_shape: Shape of input data
        predictions: Predictions dictionary
    """
    logger.info(f"🔮 Making predictions with {model_name}")
    logger.info(f"📊 Input shape: {input_shape}")
    
    if 'predictions' in predictions:
        preds = predictions['predictions']
        logger.info(f"📈 Predictions: {preds}")
        
    if 'confidence_intervals' in predictions:
        ci = predictions['confidence_intervals']
        logger.info(f"📈 Confidence intervals: {ci}")


def log_feature_importance(logger: logging.Logger,
                          model_name: str,
                          feature_importance: dict,
                          top_n: int = 10) -> None:
    """
    Log feature importance information
    
    Args:
        logger: Logger instance
        model_name: Name of model
        feature_importance: Feature importance dictionary
        top_n: Number of top features to log
    """
    logger.info(f"🎯 Feature importance for {model_name}:")
    
    # Sort by importance
    sorted_features = sorted(
        feature_importance.items(), 
        key=lambda x: x[1], 
        reverse=True
    )
    
    for i, (feature, importance) in enumerate(sorted_features[:top_n]):
        logger.info(f"   {i+1:2d}. {feature:<25}: {importance:.4f}")


def log_error(logger: logging.Logger, 
              error: Exception, 
              context: str = "") -> None:
    """
    Log error with context
    
    Args:
        logger: Logger instance
        error: Exception that occurred
        context: Additional context about when error occurred
    """
    error_msg = f"❌ Error"
    if context:
        error_msg += f" in {context}"
    error_msg += f": {type(error).__name__}: {str(error)}"
    
    logger.error(error_msg)


class ModelLogger:
    """
    Context manager for logging model operations
    
    Example:
        with ModelLogger("XGBoost Training") as log:
            # Training code here
            log.info("Training completed")
    """
    
    def __init__(self, operation_name: str, logger_name: str = "ml_models"):
        self.operation_name = operation_name
        self.logger = setup_logger(logger_name)
        self.start_time = None
        
    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.info(f"🚀 Starting {self.operation_name}")
        return self.logger
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = datetime.now() - self.start_time
        
        if exc_type is None:
            self.logger.info(
                f"✅ Completed {self.operation_name} in {duration.total_seconds():.2f}s"
            )
        else:
            self.logger.error(
                f"❌ Failed {self.operation_name} after {duration.total_seconds():.2f}s"
            )
            log_error(self.logger, exc_val, self.operation_name)


# Create default logger for ML module
ml_logger = setup_logger('ml_models', 'ml/logs/ml.log')

# Export commonly used functions
__all__ = [
    'setup_logger',
    'log_model_training',
    'log_prediction', 
    'log_feature_importance',
    'log_error',
    'ModelLogger',
    'ml_logger'
]
