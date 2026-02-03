"""
Inference Data Loader - Bridges SQLite Database to LSTM Model
Handles hybrid real/mock data loading with intelligent gap-filling using normalization statistics.
"""

import sqlite3
import json
import numpy as np
import os
from typing import List, Dict

# Exact feature order as defined in your schema
FEATURE_COLUMNS = [
    # Workout Performance (5)
    'weight_kg', 'reps', 'rir', 'total_sets', 'total_volume',
    # User Profile (4)
    'age', 'weight_kg_user', 'height_cm', 'body_fat_pct',
    # Knowledge/Assessment (5)
    'assessment_score', 'training_literacy_index', 'load_management_score', 'technique_score', 'recovery_knowledge',
    # Diet (6)
    'calories', 'protein_g', 'carbs_g', 'fats_g', 'fiber_g', 'water_ml',
    # Sleep/Stress (4)
    'sleep_hours', 'sleep_quality', 'stress_level', 'days_since_last_session',
    # Supplements (4)
    'creatine', 'protein_powder', 'pre_workout', 'caffeine_mg',
    # Recovery (7)
    'soreness_level', 'fatigue_level', 'readiness_score', 'hrv', 'resting_heart_rate', 'session_rpe', 'recovery_quality'
]


class InferenceDataLoader:
    """
    Robustly loads and prepares data for LSTM inference.
    - Fetches real workout data from SQLite
    - Fills missing features with statistical defaults from normalization_stats.json
    - Normalizes and returns ready-to-use Numpy Array
    """
    
    def __init__(self, db_path: str = "database/app.db", stats_path: str = "ml/data/normalization_stats.json"):
        """
        Initialize loader with database path and normalization statistics.
        """
        self.db_path = db_path
        
        if not os.path.exists(stats_path):
            raise FileNotFoundError(f"Normalization stats not found at {stats_path}")
            
        with open(stats_path, 'r') as f:
            self.stats = json.load(f)
            
        # Cache means/stds for quick access
        self.defaults = {k: self.stats[k]['mean'] for k in FEATURE_COLUMNS if k in self.stats}
        self.stds = {k: self.stats[k]['std'] for k in FEATURE_COLUMNS if k in self.stats}
        
        print(f"[DataLoader] Initialized with stats from {stats_path}")

    def _get_db_connection(self):
        """Get SQLite connection."""
        return sqlite3.connect(self.db_path)

    def _fetch_last_workouts(self, user_id: int, exercise_name: str, limit: int = 14) -> List[Dict]:
        """
        Fetch core workout data from SQLite.
        Returns List of dicts with workout data, ordered chronologically (oldest -> newest).
        """
        conn = self._get_db_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        query = """
            SELECT 
                weight_kg, reps, rir, sets as total_sets, date 
            FROM workouts 
            WHERE user_id = ? AND exercise_name = ?
            ORDER BY date DESC 
            LIMIT ?
        """
        
        try:
            cursor.execute(query, (user_id, exercise_name, limit))
            rows = cursor.fetchall()
            data = [dict(row) for row in rows]
            return data[::-1]  # Reverse to chronological order (oldest -> newest)
        except sqlite3.Error as e:
            print(f"[DataLoader] Database Error: {e}")
            return []
        finally:
            conn.close()

    def _fill_missing_features(self, session_data: Dict) -> Dict:
        """
        Fill missing features with sensible defaults from normalization stats.
        """
        filled_data = session_data.copy()

        # Calculate total_volume if missing
        if 'total_volume' not in filled_data:
            w = filled_data.get('weight_kg', 0)
            r = filled_data.get('reps', 0)
            s = filled_data.get('total_sets', 0)
            filled_data['total_volume'] = w * r * s

        # Handle days_since_last_session
        if 'days_since_last_session' not in filled_data:
            filled_data['days_since_last_session'] = self.defaults.get('days_since_last_session', 3)

        # Fill ALL missing columns with Mean from training data
        for col in FEATURE_COLUMNS:
            if col not in filled_data:
                filled_data[col] = self.defaults.get(col, 0.0)
        
        return filled_data

    def prepare_inference_tensor(self, user_id: int, exercise_name: str) -> np.ndarray:
        """
        Orchestrate full data loading, processing, and normalization pipeline.
        
        Returns:
            Numpy array of shape (14, 35) ready for model inference
        """
        # 1. Fetch existing history
        raw_history = self._fetch_last_workouts(user_id, exercise_name)
        
        print(f"[DataLoader] Loaded {len(raw_history)} sessions for user {user_id}, exercise '{exercise_name}'")
        
        # 2. Process sequence
        processed_sequence = []
        
        # If no history exists, create neutral initialization sequence
        if not raw_history:
            print("[DataLoader] WARNING: No history found. Creating neutral initialization.")
            dummy_session = {col: self.defaults.get(col, 0) for col in FEATURE_COLUMNS}
            raw_history = [dummy_session] * 14

        # Iterate through history and fill gaps
        for i, session in enumerate(raw_history):
            full_row = self._fill_missing_features(session)
            
            # Ensure strict ordering per FEATURE_COLUMNS
            row_values = [float(full_row[col]) for col in FEATURE_COLUMNS]
            processed_sequence.append(row_values)

        # 3. Handle padding (if user has < 14 sessions)
        # Pre-pad with first session to represent steady-state before tracking
        while len(processed_sequence) < 14:
            processed_sequence.insert(0, processed_sequence[0])

        # 4. Convert to numpy
        data_array = np.array(processed_sequence, dtype=np.float32)  # Shape: (14, 35)

        # 5. Normalize using (X - Mean) / Std
        for idx, col_name in enumerate(FEATURE_COLUMNS):
            if col_name in self.stats:
                mu = self.stats[col_name]['mean']
                sigma = self.stats[col_name]['std']
                
                # Avoid division by zero
                if sigma == 0:
                    sigma = 1e-7
                
                data_array[:, idx] = (data_array[:, idx] - mu) / sigma

        # 6. Return Numpy Array (Shape: 14, 35) - Predictor will handle batch dimension
        print(f"[DataLoader] Array prepared: shape {data_array.shape}")
        return data_array
