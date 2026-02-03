import os
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any

# Import the two "brains"
from ml_engine.models.simple_lifting_predictor import SimpleLiftingPredictor
# We wrap the torch import in try/except so the app doesn't crash if torch is missing
try:
    import torch
    from ml_engine.inference.predictor import StrengthPredictor
    from ml_engine.inference.preprocessor import InferencePreprocessor
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logger = logging.getLogger(__name__)

class HybridPredictor:
    """
    Intelligent router that decides whether to use:
    1. Heuristic Model (SimpleLiftingPredictor) -> For new users (Cold Start)
    2. Deep Learning Model (LSTM) -> For users with >14 days of data
    """
    
    def __init__(self, db_manager, model_path='ml_engine/models/strength_predictor.pt'):
        self.db = db_manager
        self.heuristic_model = SimpleLiftingPredictor()
        self.lstm_model = None
        self.preprocessor = None
        
        # Try to load the Deep Learning Brain
        if HAS_TORCH and os.path.exists(model_path):
            try:
                self.preprocessor = InferencePreprocessor(
                    normalization_stats_path='ml_engine/data/normalization_stats.json'
                )
                self.lstm_model = StrengthPredictor(model_path)
                logger.info("✅ Deep Learning Engine Loaded")
            except Exception as e:
                logger.error(f"❌ Failed to load Neural Network: {e}")
        else:
            logger.warning("⚠️ Running in Heuristic Mode (Model file not found or PyTorch missing)")

    def predict(self, user_id: int, exercise_name: str) -> Dict[str, Any]:
        """
        Main entry point for the GUI.
        """
        # 1. Fetch Data from DB
        # We need a helper in DB manager to get the last 30 days as a DataFrame
        user_df = self.db.get_user_workout_history_df(user_id, exercise_name)
        
        # 2. Check: Do we use the Big Brain or the Simple Brain?
        # LSTM needs exactly 14 sessions to form a sequence
        if self.lstm_model and len(user_df) >= 14:
            return self._run_deep_learning_inference(user_df)
        else:
            return self._run_heuristic_inference(user_df)

    def _run_heuristic_inference(self, df):
        """Standard rule-based logic for new users"""
        logger.info("📉 Not enough data for LSTM. Using Heuristic logic.")
        forecast = self.heuristic_model.predict_lifting_gains(df)
        
        return {
            'type': 'Heuristic (Rule-Based)',
            'prediction': forecast.strength_gain_next_week, # "likely", "possible"
            'confidence': forecast.strength_confidence,
            'predicted_weight': None, # Heuristics can't predict exact kg
            'uncertainty_range': None,
            'recommendations': {
                'rir': forecast.recommended_rir,
                'rest': forecast.recommended_rest_seconds
            },
            'reason': forecast.main_reason
        }

    def _run_deep_learning_inference(self, df):
        """Advanced LSTM logic for experienced users"""
        logger.info("🧠 Running LSTM Inference...")
        
        # 1. Prepare Tensor (Normalize data)
        input_tensor = self.preprocessor.prepare_user_sequence(df)
        
        # 2. Run Model
        prediction_result = self.lstm_model.predict(input_tensor)
        
        # 3. Extract Horizon 1 (Next Session) details
        h1 = prediction_result['horizon_1_session']
        
        # 4. Logic to interpret the prediction
        gain = h1['weight'] - df.iloc[-1]['weight_kg']
        if gain > 0.5:
            pred_text = "likely"
        elif gain > 0:
            pred_text = "possible"
        else:
            pred_text = "unlikely"

        return {
            'type': 'Deep Learning (LSTM)',
            'prediction': pred_text,
            'confidence': h1['confidence_score'] * 100, # Convert 0.85 -> 85%
            'predicted_weight': h1['weight'],
            'uncertainty_range': h1['weight_uncertainty'], # e.g. 2.5 kg
            'recommendations': {
                'rir': f"{int(h1['rir'])} RIR", 
                'rest': 180 # LSTM doesn't predict rest yet, default to 3m
            },
            'reason': f"Neural network detected {gain:.1f}kg strength trend based on volume & sleep patterns."
        }