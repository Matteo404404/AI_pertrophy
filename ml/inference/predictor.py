"""
Inference Engine for Strength Prediction

Loads trained LSTM model and generates predictions for users.
"""

import torch
import numpy as np
from typing import Dict, List
import sys
import os

# Add parent directory to path to import model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.models.pytorch_strength_predictor import StrengthPredictorLSTM


class StrengthPredictor:
    """Production inference engine for strength predictions."""
    
    def __init__(self, model_path: str, device: str = 'cuda'):
        """
        Initialize predictor.
        
        Args:
            model_path: Path to trained model checkpoint
            device: 'cuda' or 'cpu'
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        print(f"Checkpoint keys: {list(checkpoint.keys())}")
        
        # Create model with same architecture as training
        self.model = StrengthPredictorLSTM(
            input_dim=35,
            hidden_dim=128,
            num_layers=2,
            num_heads=8,
            dropout=0.2,
            predict_horizons=4
        )
        
        if 'model_state' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state'])
            print(f"✅ Loaded model from epoch {checkpoint.get('epoch', 'N/A')}")
        elif 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Loaded model from epoch {checkpoint.get('epoch', 'N/A')}")
        else:
            # Checkpoint is just the state dict
            self.model.load_state_dict(checkpoint)
            print("✅ Loaded model state dict")
        
        # Move to device and set to evaluation mode
        self.model.to(self.device)
        self.model.eval()
        
        print(f"   Device: {self.device}")
        print(f"   Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
    def predict(self, user_sequence: np.ndarray) -> Dict:
        """
        Generate predictions for a user.
        
        Args:
            user_sequence: (14, 35) array of 14-day history with 35 features
            
        Returns:
            Dictionary with predictions and uncertainties
        """
        # Validate input shape
        if user_sequence.shape != (14, 35):
            raise ValueError(f"Expected shape (14, 35), got {user_sequence.shape}")
        
        with torch.no_grad():
            # Convert to tensor
            input_tensor = torch.FloatTensor(user_sequence).unsqueeze(0)  # (1, 14, 35)
            input_tensor = input_tensor.to(self.device)
            
            # Get predictions
            predictions, uncertainties = self.model(input_tensor)
            
            # Convert to numpy
            predictions = predictions.cpu().numpy()[0]  # (12,)
            uncertainties = uncertainties.cpu().numpy()[0]  # (12,)
            
            # Unpack predictions (4 horizons × 3 outputs)
            results = {
                'horizon_1_session': {
                    'weight': float(predictions[0]),
                    'reps': float(predictions[1]),
                    'rir': float(predictions[2]),
                    'weight_uncertainty': float(uncertainties[0]),
                    'reps_uncertainty': float(uncertainties[1]),
                    'rir_uncertainty': float(uncertainties[2]),
                },
                'horizon_2_sessions': {
                    'weight': float(predictions[3]),
                    'reps': float(predictions[4]),
                    'rir': float(predictions[5]),
                    'weight_uncertainty': float(uncertainties[3]),
                    'reps_uncertainty': float(uncertainties[4]),
                    'rir_uncertainty': float(uncertainties[5]),
                },
                'horizon_4_sessions': {
                    'weight': float(predictions[6]),
                    'reps': float(predictions[7]),
                    'rir': float(predictions[8]),
                    'weight_uncertainty': float(uncertainties[6]),
                    'reps_uncertainty': float(uncertainties[7]),
                    'rir_uncertainty': float(uncertainties[8]),
                },
                'horizon_10_sessions': {
                    'weight': float(predictions[9]),
                    'reps': float(predictions[10]),
                    'rir': float(predictions[11]),
                    'weight_uncertainty': float(uncertainties[9]),
                    'reps_uncertainty': float(uncertainties[10]),
                    'rir_uncertainty': float(uncertainties[11]),
                }
            }
            
            return results
    
    def predict_batch(self, sequences: np.ndarray) -> List[Dict]:
        """
        Generate predictions for multiple users.
        
        Args:
            sequences: (batch_size, 14, 35) array
            
        Returns:
            List of prediction dictionaries
        """
        with torch.no_grad():
            input_tensor = torch.FloatTensor(sequences).to(self.device)
            predictions, uncertainties = self.model(input_tensor)
            
            results = []
            for i in range(len(sequences)):
                pred = predictions[i].cpu().numpy()
                uncert = uncertainties[i].cpu().numpy()
                
                result = {
                    'horizon_1_session': {
                        'weight': float(pred[0]),
                        'reps': float(pred[1]),
                        'rir': float(pred[2]),
                        'weight_uncertainty': float(uncert[0]),
                        'reps_uncertainty': float(uncert[1]),
                        'rir_uncertainty': float(uncert[2]),
                    },
                    'horizon_2_sessions': {
                        'weight': float(pred[3]),
                        'reps': float(pred[4]),
                        'rir': float(pred[5]),
                        'weight_uncertainty': float(uncert[3]),
                        'reps_uncertainty': float(uncert[4]),
                        'rir_uncertainty': float(uncert[5]),
                    },
                    'horizon_4_sessions': {
                        'weight': float(pred[6]),
                        'reps': float(pred[7]),
                        'rir': float(pred[8]),
                        'weight_uncertainty': float(uncert[6]),
                        'reps_uncertainty': float(uncert[7]),
                        'rir_uncertainty': float(uncert[8]),
                    },
                    'horizon_10_sessions': {
                        'weight': float(pred[9]),
                        'reps': float(pred[10]),
                        'rir': float(pred[11]),
                        'weight_uncertainty': float(uncert[9]),
                        'reps_uncertainty': float(uncert[10]),
                        'rir_uncertainty': float(uncert[11]),
                    }
                }
                results.append(result)
            
            return results
