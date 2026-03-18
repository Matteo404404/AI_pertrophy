"""
Hybrid Predictor — Intelligent router between heuristic and LSTM inference.

Cold-start users (< 14 sessions) get the Banister Fitness-Fatigue heuristic.
Experienced users (>= 14 sessions) get full LSTM + Bayesian uncertainty.
"""

import os
import logging
import torch
import numpy as np
import pandas as pd
from typing import Dict, Any

from ml_engine.models.simple_lifting_predictor import SimpleLiftingPredictor

try:
    from ml_engine.inference.predictor import StrengthPredictor
    from ml_engine.inference.preprocessor import InferencePreprocessor
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logger = logging.getLogger(__name__)


class HybridPredictor:
    """
    Routes between:
    1. Heuristic (SimpleLiftingPredictor) for cold-start / sparse data
    2. LSTM (StrengthPredictor) for users with >= 14 sessions
    """

    def __init__(self, db_manager, model_path='ml_engine/models/strength_predictor.pt'):
        self.db = db_manager
        self.heuristic_model = SimpleLiftingPredictor()
        self.lstm_model = None
        self.preprocessor = None

        if HAS_TORCH and os.path.exists(model_path):
            try:
                self.preprocessor = InferencePreprocessor(
                    normalization_stats_path='ml_engine/data/normalization_stats.json'
                )
                self.lstm_model = StrengthPredictor(model_path)
                logger.info("Deep Learning Engine loaded")
            except Exception as e:
                logger.error(f"Failed to load Neural Network: {e}")
        else:
            logger.warning("Running in Heuristic Mode (model file not found or PyTorch missing)")

    def predict(self, user_id: int, exercise_name: str) -> Dict[str, Any]:
        user_df = self.db.get_user_workout_history_df(user_id, exercise_name)
        user_profile = self.db.get_user_ml_profile(user_id)

        if self.lstm_model and len(user_df) >= 14:
            return self._run_deep_learning_inference(user_df, user_profile)
        else:
            return self._run_heuristic_inference(user_df, user_profile)

    def _run_heuristic_inference(self, df: pd.DataFrame,
                                  user_profile: Dict) -> Dict[str, Any]:
        logger.info("Using heuristic (Banister) logic — not enough data for LSTM")
        forecast = self.heuristic_model.predict_lifting_gains(df)

        literacy = user_profile.get('training_literacy_index', 0.5)

        return {
            'type': 'Heuristic (Banister Fitness-Fatigue)',
            'prediction': forecast.strength_gain_next_week,
            'confidence': forecast.strength_confidence,
            'predicted_weight': None,
            'uncertainty_range': None,
            'recommendations': {
                'rir': forecast.recommended_rir,
                'rest': forecast.recommended_rest_seconds,
            },
            'reason': forecast.main_reason,
            'knowledge_tier': round(literacy * 4),
        }

    def _run_deep_learning_inference(self, df: pd.DataFrame,
                                      user_profile: Dict) -> Dict[str, Any]:
        logger.info("Running LSTM inference")

        input_tensor = self.preprocessor.prepare_user_sequence(df, user_profile)
        if input_tensor is None:
            return self._run_heuristic_inference(df, user_profile)

        prediction_result = self.lstm_model.predict(input_tensor)

        h1 = prediction_result.get('horizon_1_session', {})
        predicted_weight = h1.get('weight', 0.0)
        last_weight = float(df.iloc[-1]['weight_kg']) if not df.empty else 0.0

        gain = predicted_weight - last_weight
        if gain > 0.5:
            pred_text = "likely"
        elif gain > 0:
            pred_text = "possible"
        else:
            pred_text = "unlikely"

        literacy = user_profile.get('training_literacy_index', 0.5)
        knowledge_boost = literacy * 10

        rir_val = h1.get('rir', 2)
        if rir_val <= 1:
            rir_text = "0-1 (push hard, high readiness detected)"
        elif rir_val <= 2:
            rir_text = "1-2 (moderate proximity to failure)"
        else:
            rir_text = f"{rir_val} (fatigue accumulation detected — back off)"

        return {
            'type': 'Deep Learning (LSTM + Bayesian)',
            'prediction': pred_text,
            'confidence': round(h1.get('confidence_score', 0.5) * 100, 1),
            'predicted_weight': predicted_weight,
            'uncertainty_range': h1.get('weight_uncertainty', 0),
            'recommendations': {
                'rir': rir_text,
                'rest': 180 if rir_val <= 2 else 240,
            },
            'reason': (
                f"LSTM forecasts {predicted_weight:.1f} kg (current {last_weight:.1f} kg). "
                f"Tier {round(literacy * 4)} knowledge adds {knowledge_boost:.1f}% confidence."
            ),
            'all_horizons': prediction_result,
            'knowledge_tier': round(literacy * 4),
        }
