"""
Abstract Base Model Class

Defines interface that all ML models must implement
Ensures consistency across XGBoost, ARIMA, and ensemble models
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import pandas as pd
import joblib
import os
from datetime import datetime

class BaseModel(ABC):
    """
    Abstract base class for all hypertrophy prediction models
    
    All models (XGBoost, ARIMA, Ensemble) inherit from this class
    and must implement the abstract methods.
    """
    
    def __init__(self, model_name: str):
        """
        Initialize base model
        
        Args:
            model_name: Human-readable model name (e.g., "XGBoost Predictor")
        """
        self.model_name = model_name
        self.model = None
        self.is_trained = False
        self.feature_names = []
        self.scaler = None
        self.training_history = {}
        self.last_trained = None
        
    @abstractmethod
    def fit(self, 
            X: Union[pd.DataFrame, np.ndarray], 
            y: Union[pd.Series, np.ndarray],
            validation_data: Optional[Tuple] = None) -> Dict:
        """
        Train the model on provided data
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target values (muscle gain kg/week)
            validation_data: Optional (X_val, y_val) tuple
            
        Returns:
            Dictionary with training metrics
        """
        pass
        
    @abstractmethod
    def predict(self, 
                X: Union[pd.DataFrame, np.ndarray],
                weeks_ahead: int = 4) -> Dict:
        """
        Generate predictions for given input
        
        Args:
            X: Feature matrix for prediction
            weeks_ahead: Number of weeks to predict (4, 8, or 12)
            
        Returns:
            Dictionary with predictions and confidence intervals
        """
        pass
        
    @abstractmethod
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        pass
        
    def save_model(self, filepath: str) -> None:
        """
        Save trained model to disk
        
        Args:
            filepath: Path to save model (without extension)
        """
        if not self.is_trained:
            raise ValueError(f"{self.model_name} is not trained yet")
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model and metadata
        model_data = {
            'model': self.model,
            'model_name': self.model_name,
            'feature_names': self.feature_names,
            'scaler': self.scaler,
            'training_history': self.training_history,
            'last_trained': self.last_trained,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, f"{filepath}.pkl")
        print(f"✅ Saved {self.model_name} to {filepath}.pkl")
        
    def load_model(self, filepath: str) -> None:
        """
        Load trained model from disk
        
        Args:
            filepath: Path to model file (without extension)
        """
        try:
            model_data = joblib.load(f"{filepath}.pkl")
            
            self.model = model_data['model']
            self.model_name = model_data.get('model_name', self.model_name)
            self.feature_names = model_data.get('feature_names', [])
            self.scaler = model_data.get('scaler', None)
            self.training_history = model_data.get('training_history', {})
            self.last_trained = model_data.get('last_trained', None)
            self.is_trained = model_data.get('is_trained', False)
            
            print(f"✅ Loaded {self.model_name} from {filepath}.pkl")
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file not found: {filepath}.pkl")
        except Exception as e:
            raise RuntimeError(f"Error loading model: {str(e)}")
    
    def get_model_info(self) -> Dict:
        """
        Get model metadata and performance info
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_name': self.model_name,
            'is_trained': self.is_trained,
            'n_features': len(self.feature_names),
            'feature_names': self.feature_names,
            'last_trained': self.last_trained,
            'training_history': self.training_history
        }
    
    def validate_input(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Validate and preprocess input data
        
        Args:
            X: Input features
            
        Returns:
            Processed numpy array
            
        Raises:
            ValueError: If input is invalid
        """
        if not self.is_trained:
            raise ValueError(f"{self.model_name} is not trained yet")
            
        # Convert to numpy if pandas DataFrame
        if isinstance(X, pd.DataFrame):
            X_processed = X.values
        else:
            X_processed = np.array(X)
            
        # Check dimensions
        if len(X_processed.shape) == 1:
            X_processed = X_processed.reshape(1, -1)
            
        # Check feature count
        expected_features = len(self.feature_names)
        if X_processed.shape[1] != expected_features:
            raise ValueError(
                f"Expected {expected_features} features, got {X_processed.shape[1]}"
            )
            
        # Apply scaling if scaler exists
        if self.scaler is not None:
            X_processed = self.scaler.transform(X_processed)
            
        return X_processed
    
    def calculate_confidence_intervals(self, 
                                     predictions: np.ndarray,
                                     confidence_level: float = 0.8) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate confidence intervals for predictions (default implementation)
        
        Args:
            predictions: Point predictions
            confidence_level: Confidence level (0.8 = 80%)
            
        Returns:
            Tuple of (lower_bounds, upper_bounds)
        """
        # Simple implementation - subclasses can override
        std_error = np.std(predictions) * 0.1  # Rough estimate
        z_score = 1.28 if confidence_level == 0.8 else 1.96  # 80% or 95%
        
        margin = z_score * std_error
        lower_bounds = predictions - margin
        upper_bounds = predictions + margin
        
        return lower_bounds, upper_bounds
        
    def __str__(self) -> str:
        """String representation of model"""
        status = "trained" if self.is_trained else "untrained"
        return f"{self.model_name} ({status})"
        
    def __repr__(self) -> str:
        """Detailed string representation"""
        return (f"{self.__class__.__name__}("
                f"name='{self.model_name}', "
                f"trained={self.is_trained}, "
                f"n_features={len(self.feature_names)})")


class ModelNotTrainedError(Exception):
    """Raised when trying to use an untrained model"""
    pass


class InsufficientDataError(Exception):
    """Raised when there's not enough data for training"""
    pass


class FeatureMismatchError(Exception):
    """Raised when input features don't match expected features"""
    pass
