"""
Training Pipeline for Strength Prediction Model

Handles:
- Training loop with early stopping
- Validation evaluation
- Loss functions (MSE + KL divergence for Bayesian)
- Checkpoint saving
- Learning rate scheduling

Author: AI_PERTROPHY - Task 3
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import logging
from typing import Dict, Tuple
import os
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StrengthPredictorTrainer:
    """Trainer for strength prediction model."""
    
    def __init__(self, model: nn.Module, device: torch.device = None,
                 checkpoint_dir: str = 'ml/checkpoints'):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model
            device: Torch device
            checkpoint_dir: Directory to save checkpoints
        """
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Training state
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_weight_mae': [],
            'val_weight_mae': [],
            'learning_rate': []
        }
        
        logger.info(f"Trainer initialized on device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    def setup_optimizer(self, lr: float = 1e-3, weight_decay: float = 1e-5):
        """Setup optimizer and scheduler."""
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )
            
        self.scheduler = ReduceLROnPlateau(
        self.optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        min_lr=1e-6
        )

        
        logger.info(f"Optimizer setup: AdamW(lr={lr}, weight_decay={weight_decay})")
    
    def _compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor,
                     uncertainties: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Compute total loss: MSE + uncertainty regularization.
        
        Args:
            predictions: Model predictions
            targets: Ground truth
            uncertainties: Predicted uncertainties
            
        Returns:
            (total_loss, loss_dict)
        """
        # Main MSE loss
        mse_loss = nn.MSELoss()(predictions, targets)
        
        # Uncertainty regularization (penalize high uncertainty)
        uncertainty_loss = torch.mean(uncertainties) * 0.01
        
        # Total loss
        total_loss = mse_loss + uncertainty_loss
        
        return total_loss, {
            'mse': mse_loss.item(),
            'uncertainty': uncertainty_loss.item(),
            'total': total_loss.item()
        }
    
    def _compute_metrics(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict:
        """Compute evaluation metrics."""
        # Extract weight predictions (every 3rd output, starting at 0)
        # Format: [w1, r1, rir1, w2, r2, rir2, w4, r4, rir4, w10, r10, rir10]
        weight_pred = predictions[:, [0, 3, 6, 9]]  # Weights for 4 horizons
        weight_true = targets[:, [0, 3, 6, 9]]
        
        # Reps predictions
        reps_pred = predictions[:, [1, 4, 7, 10]]
        reps_true = targets[:, [1, 4, 7, 10]]
        
        # RIR predictions
        rir_pred = predictions[:, [2, 5, 8, 11]]
        rir_true = targets[:, [2, 5, 8, 11]]
        
        # MAE
        weight_mae = torch.abs(weight_pred - weight_true).mean().item()
        reps_mae = torch.abs(reps_pred - reps_true).mean().item()
        rir_mae = torch.abs(rir_pred - rir_true).mean().item()
        
        # MAPE (for weights)
        weight_mape = (torch.abs(weight_pred - weight_true) / (torch.abs(weight_true) + 1e-6)).mean().item() * 100
        
        return {
            'weight_mae': weight_mae,
            'reps_mae': reps_mae,
            'rir_mae': rir_mae,
            'weight_mape': weight_mape,
        }
    
    def train_epoch(self, train_loader) -> Dict:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        total_weight_mae = 0.0
        num_batches = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            predictions, uncertainties = self.model(inputs)
            
            # Compute loss
            loss, loss_dict = self._compute_loss(predictions, targets, uncertainties)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Metrics
            metrics = self._compute_metrics(predictions, targets)
            
            total_loss += loss_dict['total']
            total_weight_mae += metrics['weight_mae']
            num_batches += 1
            
            if (batch_idx + 1) % 50 == 0:
                logger.info(f"  Batch {batch_idx + 1}: Loss={loss_dict['total']:.4f}, "
                           f"Weight MAE={metrics['weight_mae']:.2f}kg")
        
        return {
            'train_loss': total_loss / num_batches,
            'train_weight_mae': total_weight_mae / num_batches,
        }
    
    def validate(self, val_loader) -> Dict:
        """Evaluate on validation set."""
        self.model.eval()
        
        total_loss = 0.0
        total_weight_mae = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                predictions, uncertainties = self.model(inputs)
                loss, loss_dict = self._compute_loss(predictions, targets, uncertainties)
                metrics = self._compute_metrics(predictions, targets)
                
                total_loss += loss_dict['total']
                total_weight_mae += metrics['weight_mae']
                num_batches += 1
        
        return {
            'val_loss': total_loss / num_batches,
            'val_weight_mae': total_weight_mae / num_batches,
        }
    
    def train(self, train_loader, val_loader, num_epochs: int = 100,
             early_stopping_patience: int = 15):
        """
        Full training loop.
        
        Args:
            train_loader: Training dataloader
            val_loader: Validation dataloader
            num_epochs: Maximum epochs
            early_stopping_patience: Epochs without improvement before stopping
        """
        logger.info(f"Starting training for {num_epochs} epochs...")
        logger.info(f"Device: {self.device}")
        logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
        
        for epoch in range(num_epochs):
            logger.info(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            logger.info(f"Train Loss: {train_metrics['train_loss']:.4f}, "
                       f"Weight MAE: {train_metrics['train_weight_mae']:.2f}kg")
            
            # Validate
            val_metrics = self.validate(val_loader)
            logger.info(f"Val Loss: {val_metrics['val_loss']:.4f}, "
                       f"Weight MAE: {val_metrics['val_weight_mae']:.2f}kg")
            
            # Record history
            self.training_history['train_loss'].append(train_metrics['train_loss'])
            self.training_history['val_loss'].append(val_metrics['val_loss'])
            self.training_history['train_weight_mae'].append(train_metrics['train_weight_mae'])
            self.training_history['val_weight_mae'].append(val_metrics['val_weight_mae'])
            self.training_history['learning_rate'].append(
                self.optimizer.param_groups[0]['lr']
            )
            
            # Learning rate scheduling
            self.scheduler.step(val_metrics['val_loss'])
            
            # Early stopping
            if val_metrics['val_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['val_loss']
                self.patience_counter = 0
                self.save_checkpoint(epoch, val_metrics)
                logger.info(f"✅ New best model! Val loss: {self.best_val_loss:.4f}")
            else:
                self.patience_counter += 1
                if self.patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping after {epoch + 1} epochs")
                    break
        
        logger.info(f"\n✅ Training complete!")
        return self.training_history
    
    def save_checkpoint(self, epoch: int, metrics: Dict):
        """Save model checkpoint."""
        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f"model_epoch{epoch + 1}_loss{metrics['val_loss']:.4f}.pt"
        )
        
        torch.save({
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'metrics': metrics,
            'history': self.training_history,
        }, checkpoint_path)
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state'])
        if 'optimizer_state' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        return checkpoint
    
    def save_final_model(self, filepath: str = 'ml/models/strength_predictor.pt'):
        """Save final trained model."""
        torch.save(self.model.state_dict(), filepath)
        logger.info(f"Final model saved to {filepath}")