"""
PyTorch Dataset for Strength Prediction

Handles sequence construction, feature normalization, and batch loading.
Each sequence is 14 days of training history → 4 prediction horizons.

Author: AI_PERTROPHY - Task 3
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StrengthPredictionDataset(Dataset):
    """
    PyTorch Dataset for strength prediction.
    
    Input: 14-day sequence of workout/recovery data
    Output: Next 1, 2, 4, 10 sessions predictions (weight, reps, RIR)
    """
    
    def __init__(self, df: pd.DataFrame, sequence_length: int = 14, 
                 predict_horizons: List[int] = None, normalize: bool = True):
        """
        Initialize dataset.
        
        Args:
            df: DataFrame with training data
            sequence_length: Days of history to use
            predict_horizons: Sessions ahead to predict [1, 2, 4, 10]
            normalize: Whether to normalize features
        """
        self.df = df.copy()
        self.sequence_length = sequence_length
        self.predict_horizons = predict_horizons or [1, 2, 4, 10]
        self.normalize = normalize
        
        # Feature columns
        self.feature_columns = self._get_feature_columns()
        
        # Normalization statistics
        self.feature_stats = {}
        if normalize:
            self._compute_normalization_stats()
            self._normalize_features()
        
        # Build sequences
        self.sequences = self._build_sequences()
        
        logger.info(f"Created dataset with {len(self.sequences)} sequences")
    
    def _get_feature_columns(self) -> List[str]:
        """Get list of feature columns to use."""
        features = [
            # Performance
            'weight_kg', 'reps', 'rir', 'total_sets',
            
            # User profile (static)
            'age', 'weight_kg_user', 'height_cm', 'body_fat_pct',
            
            # Assessment (static per user)
            'assessment_score', 'training_literacy_index', 
            'load_management_score', 'technique_score', 'recovery_knowledge',
            
            # Diet
            'calories', 'protein_g', 'carbs_g', 'fats_g', 'hydration_l',
            'meal_timing_score', 'diet_consistency',
            
            # Sleep
            'sleep_duration', 'sleep_quality', 'deep_sleep_pct', 'rem_sleep_pct',
            'sleep_consistency',
            
            # Supplements
            'creatine_taken', 'caffeine_mg', 'beta_alanine_taken', 
            'protein_shake_taken', 'supplement_count',
            
            # Recovery
            'days_since_last_session', 'weekly_volume', 'weekly_frequency',
            'fatigue_index', 'recovery_score',
        ]
        return features
    
    def _compute_normalization_stats(self):
        """Compute mean and std for each feature."""
        for col in self.feature_columns:
            if col in self.df.columns:
                self.feature_stats[col] = {
                    'mean': self.df[col].mean(),
                    'std': self.df[col].std() + 1e-6  # Avoid division by zero
                }
            else:
                logger.warning(f"Column {col} not in dataframe")
    
    def _normalize_features(self):
        """Normalize features using z-score."""
        for col, stats in self.feature_stats.items():
            self.df[col] = (self.df[col] - stats['mean']) / stats['std']
    
    def _build_sequences(self) -> List[Dict]:
        """
        Build training sequences.
        
        Each sequence:
        - 14 days of workout history (input)
        - Prediction for 1, 2, 4, 10 sessions ahead (output)
        """
        sequences = []
        
        # Group by user and exercise
        grouped = self.df.groupby(['user_id', 'exercise_id'])
        
        for (user_id, exercise_id), group_df in grouped:
            group_df = group_df.sort_values('day_offset')
            
            if len(group_df) < self.sequence_length + 10:
                continue  # Skip if not enough data
            
            # Slide window through data
            for i in range(self.sequence_length, len(group_df) - 10, 2):  # Step by 2 for data efficiency
                
                # Input: previous sequence_length days
                input_seq = group_df.iloc[i - self.sequence_length:i]
                
                # Outputs: predictions for each horizon
                future_targets = {}
                for horizon in self.predict_horizons:
                    if i + horizon < len(group_df):
                        future_row = group_df.iloc[i + horizon]
                        future_targets[horizon] = {
                            'weight': future_row['weight_kg'],
                            'reps': future_row['reps'],
                            'rir': future_row['rir'],
                        }
                
                # Only add if we have all horizon targets
                if len(future_targets) == len(self.predict_horizons):
                    sequences.append({
                        'user_id': user_id,
                        'exercise_id': exercise_id,
                        'input_seq': input_seq,
                        'targets': future_targets,
                    })
        
        logger.info(f"Built {len(sequences)} training sequences")
        return sequences
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get one training example.
        
        Returns:
            (input_features, output_targets)
        """
        seq_data = self.sequences[idx]
        input_df = seq_data['input_seq']
        targets = seq_data['targets']
        
        # Extract features
        features = []
        for _, row in input_df.iterrows():
            feature_vec = []
            for col in self.feature_columns:
                if col in row.index:
                    feature_vec.append(float(row[col]))
                else:
                    feature_vec.append(0.0)
            features.append(feature_vec)
        
        # Convert to tensor (sequence_length, num_features)
        input_tensor = torch.tensor(features, dtype=torch.float32)
        
        # Build output tensor (4 horizons × 3 outputs = 12)
        output_list = []
        for horizon in self.predict_horizons:
            target = targets[horizon]
            output_list.extend([
                target['weight'],
                target['reps'],
                target['rir'],
            ])
        
        output_tensor = torch.tensor(output_list, dtype=torch.float32)
        
        return input_tensor, output_tensor
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names."""
        return self.feature_columns
    
    def get_normalization_stats(self) -> Dict:
        """Get normalization statistics for later inference."""
        return self.feature_stats


def create_dataloaders(df: pd.DataFrame, batch_size: int = 32, 
                      train_split: float = 0.8) -> Tuple:
    """
    Create train and validation dataloaders.
    
    Args:
        df: Full dataset
        batch_size: Batch size
        train_split: Fraction for training (rest for validation)
        
    Returns:
        (train_loader, val_loader, dataset)
    """
    # Create dataset
    dataset = StrengthPredictionDataset(df, normalize=True)
    
    # Split by sequences
    n_sequences = len(dataset)
    train_size = int(n_sequences * train_split)
    
    train_indices = np.random.choice(n_sequences, train_size, replace=False)
    val_indices = np.array([i for i in range(n_sequences) if i not in train_indices])
    
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    
    logger.info(f"Created dataloaders: {len(train_dataset)} train, {len(val_dataset)} val")
    
    return train_loader, val_loader, dataset