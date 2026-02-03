"""
PyTorch LSTM Strength Predictor with Attention & Bayesian Uncertainty

State-of-the-art architecture:
- LSTM encoder for sequence processing
- Multi-head attention for feature importance
- Bayesian uncertainty quantification
- Multi-horizon prediction heads (1, 2, 4, 10 sessions)

Author: AI_PERTROPHY - Task 3
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism."""
    
    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, 
                value: torch.Tensor, mask=None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = query.shape[0]
        
        Q = self.query(query)
        K = self.key(key)
        V = self.value(value)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        energy = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float('-1e10'))
        
        attention = torch.softmax(energy, dim=-1)
        attention = self.dropout(attention)
        
        out = torch.matmul(attention, V)
        out = out.transpose(1, 2).contiguous()
        out = out.view(batch_size, -1, self.hidden_dim)
        out = self.fc_out(out)
        
        return out, attention


class BayesianLinear(nn.Module):
    """Bayesian linear layer for uncertainty quantification."""
    
    def __init__(self, in_features: int, out_features: int, num_samples: int = 4):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_samples = num_samples
        
        # Mean parameters
        self.weight_mu = nn.Parameter(torch.empty((out_features, in_features)))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        
        # Log variance parameters
        self.weight_log_sigma = nn.Parameter(torch.empty((out_features, in_features)))
        self.bias_log_sigma = nn.Parameter(torch.empty(out_features))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight_mu)
        nn.init.zeros_(self.bias_mu)
        self.weight_log_sigma.data.fill_(-5.0)
        self.bias_log_sigma.data.fill_(-5.0)

    def forward(self, x: torch.Tensor, sample: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        if sample:
            # Sample weights from distribution
            weight_sigma = torch.exp(self.weight_log_sigma)
            bias_sigma = torch.exp(self.bias_log_sigma)
            
            weight_eps = torch.randn_like(self.weight_mu)
            bias_eps = torch.randn_like(self.bias_mu)
            
            weight = self.weight_mu + weight_sigma * weight_eps
            bias = self.bias_mu + bias_sigma * bias_eps
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        
        mean = F.linear(x, self.weight_mu, self.bias_mu)
        uncertainty = F.linear(x ** 2, torch.exp(self.weight_log_sigma) ** 2, 
                              torch.exp(self.bias_log_sigma) ** 2) ** 0.5
        
        return mean, uncertainty


class StrengthPredictorLSTM(nn.Module):
    """
    LSTM-based strength predictor with attention and Bayesian uncertainty.
    
    Architecture:
    1. LSTM encoder (processes 14-day sequences)
    2. Multi-head attention
    3. Dense layers
    4. Multi-horizon prediction heads
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, 
                 num_layers: int = 2, num_heads: int = 8, dropout: float = 0.2,
                 predict_horizons: int = 4, outputs_per_horizon: int = 3):
        """
        Initialize model.
        
        Args:
            input_dim: Number of input features
            hidden_dim: Hidden dimension size
            num_layers: Number of LSTM layers
            num_heads: Number of attention heads
            dropout: Dropout rate
            predict_horizons: Number of prediction horizons (4: [1,2,4,10])
            outputs_per_horizon: Outputs per horizon (3: weight, reps, rir)
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.predict_horizons = predict_horizons
        self.outputs_per_horizon = outputs_per_horizon
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # LSTM encoder
        self.lstm = nn.LSTM(
            hidden_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention
        self.attention = MultiHeadAttention(hidden_dim, num_heads, dropout)
        
        # Dense layers
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.dropout_layer = nn.Dropout(dropout)
        
        # Prediction heads (one per horizon)
        self.prediction_heads = nn.ModuleList([
            nn.Linear(hidden_dim // 2, outputs_per_horizon)
            for _ in range(predict_horizons)
        ])
        
        # Uncertainty heads (Bayesian)
        self.uncertainty_heads = nn.ModuleList([
            BayesianLinear(hidden_dim // 2, outputs_per_horizon)
            for _ in range(predict_horizons)
        ])
        
        logger.info(f"Created StrengthPredictorLSTM with {self._count_parameters()} parameters")
    
    def _count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch_size, seq_len, input_dim)
            
        Returns:
            (predictions, uncertainties) each shape (batch_size, predict_horizons * outputs_per_horizon)
        """
        # Project input
        x = self.input_proj(x)  # (batch, seq_len, hidden_dim)
        
        # LSTM encoding
        lstm_out, (hidden, cell) = self.lstm(x)  # (batch, seq_len, hidden_dim)
        
        # Attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)  # (batch, seq_len, hidden_dim)
        
        # Use last hidden state + attention output
        final_hidden = hidden[-1]  # (batch, hidden_dim)
        attn_last = attn_out[:, -1, :]  # (batch, hidden_dim)
        combined = final_hidden + attn_last  # (batch, hidden_dim)
        
        # Dense layers
        x = F.relu(self.fc1(combined))
        x = self.dropout_layer(x)
        x = F.relu(self.fc2(x))  # (batch, hidden_dim // 2)
        x = self.dropout_layer(x)
        
        # Multi-horizon predictions
        predictions = []
        uncertainties = []
        
        for i in range(self.predict_horizons):
            # Point prediction
            pred = self.prediction_heads[i](x)  # (batch, outputs_per_horizon)
            predictions.append(pred)
            
            # Uncertainty
            mean, uncert = self.uncertainty_heads[i](x, sample=True)
            uncertainties.append(uncert)
        
        # Concatenate all horizons
        predictions = torch.cat(predictions, dim=1)  # (batch, horizons * outputs)
        uncertainties = torch.cat(uncertainties, dim=1)  # (batch, horizons * outputs)
        
        return predictions, uncertainties
    
    def predict(self, x: torch.Tensor, num_samples: int = 10) -> Dict:
        """
        Generate predictions with uncertainty estimates.
        
        Args:
            x: Input tensor
            num_samples: Number of MC samples for uncertainty
            
        Returns:
            Dict with predictions and confidence intervals
        """
        self.eval()
        with torch.no_grad():
            preds_list = []
            uncert_list = []
            
            # MC dropout sampling
            for _ in range(num_samples):
                pred, uncert = self.forward(x)
                preds_list.append(pred)
                uncert_list.append(uncert)
            
            # Average predictions
            mean_pred = torch.stack(preds_list).mean(dim=0)
            pred_std = torch.stack(preds_list).std(dim=0)
            
            # Average uncertainty
            mean_uncert = torch.stack(uncert_list).mean(dim=0)
        
        return {
            'predictions': mean_pred,
            'aleatoric_uncertainty': mean_uncert,
            'epistemic_uncertainty': pred_std,
            'confidence': 1.0 / (1.0 + mean_uncert),
        }


def create_model(input_dim: int, hidden_dim: int = 128, num_layers: int = 2,
                num_heads: int = 8, predict_horizons: int = 4) -> StrengthPredictorLSTM:
    """Create and return model."""
    model = StrengthPredictorLSTM(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        predict_horizons=predict_horizons,
        outputs_per_horizon=3  # weight, reps, rir
    )
    return model