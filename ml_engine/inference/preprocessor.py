"""
Inference Preprocessor

Converts a user's workout history DataFrame + static profile into
a normalized (1, 14, 35) tensor the LSTM model expects.
Also handles denormalization of model outputs back to real units.
"""

import numpy as np
import pandas as pd
import torch
import json
import os
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    'weight_kg', 'reps', 'rir', 'total_sets', 'total_volume',
    'age', 'weight_kg_user', 'height_cm', 'body_fat_pct',
    'assessment_score', 'training_literacy_index', 'load_management_score',
    'technique_score', 'recovery_knowledge',
    'calories', 'protein_g', 'carbs_g', 'fats_g', 'fiber_g', 'water_ml',
    'sleep_hours', 'sleep_quality', 'stress_level', 'days_since_last_session',
    'creatine', 'protein_powder', 'pre_workout', 'caffeine_mg',
    'soreness_level', 'fatigue_level', 'readiness_score', 'hrv',
    'resting_heart_rate', 'session_rpe', 'recovery_quality',
]

SEQUENCE_LENGTH = 14

DEFAULTS = {
    'weight_kg': 60.0, 'reps': 8, 'rir': 2, 'total_sets': 3,
    'total_volume': 480.0,
    'age': 25, 'weight_kg_user': 75, 'height_cm': 175, 'body_fat_pct': 15,
    'assessment_score': 50, 'training_literacy_index': 0.5,
    'load_management_score': 0.5, 'technique_score': 0.5,
    'recovery_knowledge': 0.5,
    'calories': 2500, 'protein_g': 150, 'carbs_g': 250, 'fats_g': 80,
    'fiber_g': 30, 'water_ml': 3000,
    'sleep_hours': 7.5, 'sleep_quality': 7, 'stress_level': 5,
    'days_since_last_session': 2,
    'creatine': 0, 'protein_powder': 0, 'pre_workout': 0, 'caffeine_mg': 0,
    'soreness_level': 5, 'fatigue_level': 5, 'readiness_score': 5,
    'hrv': 60, 'resting_heart_rate': 60, 'session_rpe': 7,
    'recovery_quality': 7,
}


class InferencePreprocessor:
    """Normalizes user data for LSTM input and denormalizes predictions."""

    def __init__(self, normalization_stats_path: str = None):
        self.stats: Dict[str, Dict[str, float]] = {}

        if normalization_stats_path and os.path.exists(normalization_stats_path):
            with open(normalization_stats_path, 'r') as f:
                self.stats = json.load(f)
            logger.info(f"Loaded normalization stats for {len(self.stats)} features")

    def prepare_user_sequence(self, workout_df: pd.DataFrame,
                               user_profile: Optional[Dict] = None) -> Optional[torch.Tensor]:
        """
        Build a (1, 14, 35) tensor from the user's last 14 workout rows.

        Args:
            workout_df: DataFrame from db.get_user_workout_history_df().
                        Must have at least 14 rows with columns like
                        weight_kg, reps, rir, sleep_duration_hours, etc.
            user_profile: Static user features from db.get_user_ml_profile().

        Returns:
            torch.Tensor of shape (1, 14, 35) or None if data insufficient.
        """
        if workout_df is None or len(workout_df) < SEQUENCE_LENGTH:
            return None

        df = workout_df.tail(SEQUENCE_LENGTH).copy()

        col_map = {
            'sleep_duration_hours': 'sleep_hours',
        }
        df.rename(columns=col_map, inplace=True)

        if 'total_volume' not in df.columns and 'weight_kg' in df.columns:
            sets = df.get('total_sets', pd.Series([3] * len(df)))
            df['total_volume'] = df['weight_kg'] * df['reps'] * sets

        if user_profile:
            for key in ['age', 'weight_kg_user', 'height_cm', 'body_fat_pct',
                        'assessment_score', 'training_literacy_index',
                        'load_management_score', 'technique_score',
                        'recovery_knowledge']:
                if key not in df.columns or df[key].isna().all():
                    df[key] = user_profile.get(key, DEFAULTS.get(key, 0))

        for col in FEATURE_COLUMNS:
            if col not in df.columns:
                df[col] = DEFAULTS.get(col, 0)
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col].fillna(DEFAULTS.get(col, 0), inplace=True)

        features = df[FEATURE_COLUMNS].values.astype(np.float32)

        normalized = self._normalize(features)

        tensor = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0)
        return tensor

    def _normalize(self, arr: np.ndarray) -> np.ndarray:
        """Z-score normalize each feature column using training stats."""
        out = arr.copy()
        for i, col_name in enumerate(FEATURE_COLUMNS):
            if col_name in self.stats:
                mean = self.stats[col_name]['mean']
                std = self.stats[col_name]['std']
                if std < 1e-5:
                    std = 1.0
                out[:, i] = (arr[:, i] - mean) / std
        return out
