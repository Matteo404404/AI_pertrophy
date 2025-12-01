"""
Data Preprocessor for Inference

Handles normalization stats and denormalization of predictions.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
import json
import os


class InferencePreprocessor:
    """Prepares user data for model inference and denormalizes outputs."""
    
    def __init__(self, normalization_stats_path: str = None):
        """
        Initialize with normalization statistics from training.
        
        Args:
            normalization_stats_path: Path to JSON file with mean/std for each feature
        """
        # Default normalization stats (you'll need to extract these from your training data)
        # These are placeholders - replace with actual values from training
        self.stats = {
            'weight_kg': {'mean': 100.0, 'std': 25.0},
            'reps': {'mean': 8.0, 'std': 3.0},
            'rir': {'mean': 2.0, 'std': 1.5},
            # Add all 35 features here...
        }
        
        # Load from file if provided
        if normalization_stats_path and os.path.exists(normalization_stats_path):
            with open(normalization_stats_path, 'r') as f:
                self.stats = json.load(f)
    
    def denormalize_predictions(self, predictions: Dict) -> Dict:
        """
        Convert normalized predictions back to original scale.
        
        Args:
            predictions: Dictionary with normalized predictions
            
        Returns:
            Dictionary with denormalized predictions in real units
        """
        denormalized = {}
        
        for horizon, values in predictions.items():
            # Denormalize weight (kg)
            weight_denorm = (values['weight'] * self.stats['weight_kg']['std'] + 
                           self.stats['weight_kg']['mean'])
            
            # Denormalize reps
            reps_denorm = (values['reps'] * self.stats['reps']['std'] + 
                         self.stats['reps']['mean'])
            
            # Denormalize RIR
            rir_denorm = (values['rir'] * self.stats['rir']['std'] + 
                        self.stats['rir']['mean'])
            
            # Denormalize uncertainties
            weight_unc_denorm = values['weight_uncertainty'] * self.stats['weight_kg']['std']
            reps_unc_denorm = values['reps_uncertainty'] * self.stats['reps']['std']
            rir_unc_denorm = values['rir_uncertainty'] * self.stats['rir']['std']
            
            denormalized[horizon] = {
                'weight_kg': max(0, weight_denorm),  # Can't be negative
                'reps': max(1, int(round(reps_denorm))),  # Must be at least 1
                'rir': max(0, min(10, int(round(rir_denorm)))),  # Clamp 0-10
                'weight_confidence_interval': (
                    max(0, weight_denorm - 1.96 * weight_unc_denorm),
                    weight_denorm + 1.96 * weight_unc_denorm
                ),
                'reps_confidence_interval': (
                    max(1, int(reps_denorm - 1.96 * reps_unc_denorm)),
                    int(reps_denorm + 1.96 * reps_unc_denorm)
                ),
                'rir_confidence_interval': (
                    max(0, int(rir_denorm - 1.96 * rir_unc_denorm)),
                    min(10, int(rir_denorm + 1.96 * rir_unc_denorm))
                )
            }
        
        return denormalized
    
    def prepare_user_sequence(self, workout_history: list) -> Optional[np.ndarray]:
        """
        Convert user workout history to normalized model input.
        
        Args:
            workout_history: List of 14 workout dictionaries with all features
            
        Returns:
            (14, 35) normalized array or None if insufficient data
        """
        if len(workout_history) < 14:
            return None
        
        # Take last 14 workouts
        recent_workouts = workout_history[-14:]
        
        # Extract features (all 35 features in the correct order)
        sequence = []
        for workout in recent_workouts:
            feature_vec = [
                workout['weight_kg'],
                workout['reps'],
                workout['rir'],
                # ... all 35 features
            ]
            sequence.append(feature_vec)
        
        # Convert to numpy
        sequence = np.array(sequence, dtype=np.float32)
        
        # Normalize
        normalized = self._normalize(sequence)
        
        return normalized
    
    def _normalize(self, sequence: np.ndarray) -> np.ndarray:
        """Normalize features using training statistics."""
        normalized = sequence.copy()
        
        feature_names = list(self.stats.keys())
        for i, feature_name in enumerate(feature_names):
            if i < sequence.shape[1]:
                normalized[:, i] = ((sequence[:, i] - self.stats[feature_name]['mean']) / 
                                  self.stats[feature_name]['std'])
        
        return normalized
