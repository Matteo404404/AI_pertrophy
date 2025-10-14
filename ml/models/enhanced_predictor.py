"""
Enhanced Lifting Predictor with Error Handling
Production-ready version with input validation and error recovery.
"""
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
import logging
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PredictionResult:
    success: bool
    prediction: Optional[str] = None
    confidence: Optional[float] = None
    rir_recommendation: Optional[str] = None
    rest_seconds: Optional[int] = None
    reason: Optional[str] = None
    warning: Optional[str] = None
    error: Optional[str] = None

class EnhancedLiftingPredictor:
    """
    Production-ready lifting predictor with robust error handling.
    
    Features:
    - Input validation and sanitization
    - Missing data handling
    - Graceful error recovery  
    - Detailed logging
    - Flexible data requirements
    """
    
    def __init__(self):
        self.required_columns = ['total_sets']
        self.optional_columns = ['average_rpe', 'sleep_quality_1_10', 'sleep_duration_hours', 
                                'hrv_rmssd', 'perceived_stress_1_10']
        
    def predict(self, data: Any) -> PredictionResult:
        """
        Main prediction method with comprehensive error handling.
        
        Args:
            data: Training data (pandas DataFrame expected)
            
        Returns:
            PredictionResult with success status and prediction details
        """
        try:
            # Step 1: Input validation
            logger.info("Starting prediction process")
            validation_result = self._validate_input(data)
            if not validation_result.success:
                logger.warning(f"Input validation failed: {validation_result.error}")
                return validation_result
            
            # Step 2: Data cleaning and preparation
            logger.info("Cleaning and preparing data")
            clean_data = self._clean_data(data)
            
            # Step 3: Quality checks
            quality_warnings = self._check_data_quality(clean_data)
            
            # Step 4: Make prediction using core predictor
            logger.info("Making prediction")
            from ml.models.simple_lifting_predictor import SimpleLiftingPredictor
            predictor = SimpleLiftingPredictor()
            forecast = predictor.predict_lifting_gains(clean_data)
            
            # Combine warnings
            all_warnings = []
            if quality_warnings:
                all_warnings.extend(quality_warnings)
            if forecast.warning:
                all_warnings.append(forecast.warning)
            
            final_warning = "; ".join(all_warnings) if all_warnings else None
            
            logger.info(f"Prediction successful: {forecast.strength_gain_next_week} ({forecast.strength_confidence:.0f}%)")
            
            return PredictionResult(
                success=True,
                prediction=forecast.strength_gain_next_week,
                confidence=forecast.strength_confidence,
                rir_recommendation=forecast.recommended_rir,
                rest_seconds=forecast.recommended_rest_seconds,
                reason=forecast.main_reason,
                warning=final_warning
            )
            
        except ImportError as e:
            error_msg = f"Missing required components: {str(e)}"
            logger.error(error_msg)
            return PredictionResult(success=False, error=error_msg)
            
        except Exception as e:
            error_msg = f"Prediction failed: {str(e)}"
            logger.error(error_msg)
            return PredictionResult(success=False, error=error_msg)
    
    def _validate_input(self, data: Any) -> PredictionResult:
        """Comprehensive input validation."""
        
        # Check if data exists
        if data is None:
            return PredictionResult(success=False, error="No data provided")
        
        # Check data type
        if not isinstance(data, pd.DataFrame):
            return PredictionResult(
                success=False, 
                error=f"Data must be a pandas DataFrame, got {type(data).__name__}"
            )
        
        # Check if empty
        if len(data) == 0:
            return PredictionResult(success=False, error="Empty dataset provided")
        
        # Check minimum data requirements
        if len(data) < 3:
            return PredictionResult(
                success=False, 
                error=f"Need at least 3 days of data for prediction, got {len(data)} days"
            )
        
        # Check for required columns
        missing_required = [col for col in self.required_columns if col not in data.columns]
        if missing_required:
            return PredictionResult(
                success=False, 
                error=f"Missing required columns: {missing_required}. Available columns: {list(data.columns)}"
            )
        
        # Check for completely null columns
        null_required = [col for col in self.required_columns if data[col].isna().all()]
        if null_required:
            return PredictionResult(
                success=False,
                error=f"Required columns contain only null values: {null_required}"
            )
        
        logger.info(f"Input validation passed: {len(data)} rows, {len(data.columns)} columns")
        return PredictionResult(success=True)
    
    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare data with sensible defaults."""
        
        clean = data.copy()
        
        # Add missing optional columns with reasonable defaults
        defaults = {
            'average_rpe': 7.0,           # Moderate intensity
            'sleep_quality_1_10': 7.0,   # Good sleep
            'sleep_duration_hours': 7.5, # Adequate sleep
            'hrv_rmssd': 45.0,           # Average HRV
            'perceived_stress_1_10': 5.0  # Moderate stress
        }
        
        for col, default_val in defaults.items():
            if col not in clean.columns:
                clean[col] = default_val
                logger.info(f"Added missing column '{col}' with default value {default_val}")
            else:
                # Fill missing values in existing columns
                missing_count = clean[col].isna().sum()
                if missing_count > 0:
                    clean[col].fillna(default_val, inplace=True)
                    logger.info(f"Filled {missing_count} missing values in '{col}' with {default_val}")
        
        # Clip values to reasonable physiological ranges
        ranges = {
            'total_sets': (0, 50),
            'average_rpe': (1, 10),
            'sleep_quality_1_10': (1, 10),
            'sleep_duration_hours': (3, 12),
            'hrv_rmssd': (10, 100),
            'perceived_stress_1_10': (1, 10)
        }
        
        for col, (min_val, max_val) in ranges.items():
            if col in clean.columns:
                original_range = (clean[col].min(), clean[col].max())
                clean[col] = np.clip(clean[col], min_val, max_val)
                new_range = (clean[col].min(), clean[col].max())
                
                if original_range != new_range:
                    logger.info(f"Clipped '{col}' from range {original_range} to {new_range}")
        
        # Handle infinite values
        clean.replace([np.inf, -np.inf], np.nan, inplace=True)
        clean.fillna(method='forward', inplace=True)
        clean.fillna(method='backward', inplace=True)
        
        return clean
    
    def _check_data_quality(self, data: pd.DataFrame) -> list:
        """Check data quality and return warnings."""
        warnings = []
        
        # Check for insufficient recent data
        if len(data) < 7:
            warnings.append(f"Only {len(data)} days of data - predictions more reliable with 7+ days")
        
        # Check for training consistency
        if 'total_sets' in data.columns:
            zero_training_days = (data['total_sets'] == 0).sum()
            training_percentage = (1 - zero_training_days / len(data)) * 100
            
            if training_percentage < 30:
                warnings.append(f"Very low training frequency ({training_percentage:.0f}% of days)")
            elif training_percentage < 50:
                warnings.append(f"Low training frequency ({training_percentage:.0f}% of days)")
        
        # Check for extreme values
        if 'average_rpe' in data.columns:
            avg_rpe = data['average_rpe'].mean()
            if avg_rpe > 9:
                warnings.append("Very high average RPE - may indicate overreaching")
            elif avg_rpe < 6:
                warnings.append("Very low average RPE - may not be training with enough intensity")
        
        # Check sleep patterns
        if 'sleep_duration_hours' in data.columns:
            avg_sleep = data['sleep_duration_hours'].mean()
            if avg_sleep < 6:
                warnings.append("Insufficient sleep duration may impact recovery")
        
        # Check for data staleness (if date column exists)
        if 'date' in data.columns:
            try:
                latest_date = pd.to_datetime(data['date']).max()
                days_old = (pd.Timestamp.now() - latest_date).days
                if days_old > 7:
                    warnings.append(f"Data is {days_old} days old - predictions more accurate with recent data")
            except:
                pass  # Date parsing failed, skip this check
        
        if warnings:
            logger.info(f"Data quality warnings: {len(warnings)} issues found")
        
        return warnings
    
    def get_feature_importance(self) -> Dict[str, str]:
        """Get information about feature importance for predictions."""
        return {
            'total_sets': 'Training volume - most important for fatigue assessment',
            'average_rpe': 'Training intensity - key for performance trends',
            'sleep_quality_1_10': 'Recovery indicator - affects prediction confidence',
            'sleep_duration_hours': 'Recovery duration - impacts fatigue levels',
            'hrv_rmssd': 'Autonomic recovery - sensitive recovery marker',
            'perceived_stress_1_10': 'Stress levels - affects recovery capacity'
        }
    
    def get_prediction_explanation(self, result: PredictionResult) -> Dict[str, str]:
        """Get detailed explanation of prediction factors."""
        if not result.success:
            return {'error': result.error}
        
        explanation = {
            'prediction': f"Strength gains are {result.prediction} this week",
            'confidence': f"{result.confidence:.0f}% confidence based on data quality and patterns",
            'rir_logic': f"Recommended {result.rir_recommendation} based on fatigue and recovery status",
            'rest_logic': f"Rest {result.rest_seconds} seconds to optimize recovery between sets",
            'main_factors': result.reason
        }
        
        if result.warning:
            explanation['warnings'] = result.warning
        
        return explanation


# Convenience function for simple usage
def predict_lifting_gains(data: pd.DataFrame) -> PredictionResult:
    """
    Simple convenience function for making predictions.
    
    Args:
        data: Training data DataFrame
        
    Returns:
        PredictionResult with prediction details
    """
    predictor = EnhancedLiftingPredictor()
    return predictor.predict(data)


if __name__ == "__main__":
    # Example usage
    print("🧪 Testing Enhanced Lifting Predictor")
    
    # Create sample data
    sample_data = pd.DataFrame({
        'total_sets': [12, 14, 13, 15, 16, 14, 0, 13, 15, 17, 16, 14, 18, 0],
        'average_rpe': [7.5, 7.8, 7.2, 8.0, 7.1, 8.2, 0, 7.3, 7.9, 7.4, 8.1, 7.2, 8.3, 0],
        'sleep_quality_1_10': [7, 8, 6, 7, 8, 5, 9, 7, 6, 8, 7, 8, 6, 9]
    })
    
    predictor = EnhancedLiftingPredictor()
    result = predictor.predict(sample_data)
    
    if result.success:
        print("✅ Test successful!")
        print(f"Prediction: {result.prediction}")
        print(f"Confidence: {result.confidence:.0f}%")
        print(f"RIR: {result.rir_recommendation}")
        print(f"Rest: {result.rest_seconds}s")
        if result.warning:
            print(f"Warning: {result.warning}")
    else:
        print(f"❌ Test failed: {result.error}")
