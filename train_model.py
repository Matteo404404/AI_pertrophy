"""
Train Strength Prediction Model

Complete training pipeline:
1. Generate synthetic data (10K users, 90 days each)
2. Create PyTorch dataloaders
3. Build and train LSTM model
4. Save trained model

Usage:
    python train_model.py
    
    Optional arguments:
    --num_users 10000      Number of synthetic users
    --batch_size 32        Training batch size
    --num_epochs 100       Maximum epochs
    --lr 0.001            Learning rate

Author: AI_PERTROPHY - Task 3
"""

import argparse
import torch
import pandas as pd
import logging
import os
from datetime import datetime

# Add imports
from ml.data.enhanced_synthetic_generator import EnhancedSyntheticDataGenerator
from ml.training.dataset import create_dataloaders
from ml.models.pytorch_strength_predictor import create_model
from ml.training.trainer import StrengthPredictorTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(args):
    """Main training pipeline."""
    
    logger.info("="*80)
    logger.info("STRENGTH PREDICTION MODEL - TRAINING PIPELINE")
    logger.info("="*80)
    
    # Step 1: Generate synthetic data
    logger.info("\n[STEP 1] Generating synthetic training data...")
    logger.info(f"  • Users: {args.num_users}")
    logger.info(f"  • History: 90 days per user")
    logger.info(f"  • Features: diet, sleep, supplements, recovery, assessment")
    
    generator = EnhancedSyntheticDataGenerator(
        num_users=args.num_users,
        days_history=90,
        seed=42
    )
    
    df = generator.generate_dataset()
    training_data_path = 'ml/data/training_data.csv'
    generator.save_dataset(df, training_data_path)
    
    logger.info(f"  ✅ Generated {len(df)} records")
    logger.info(f"     • Unique users: {df['user_id'].nunique()}")
    logger.info(f"     • Unique exercises: {df['exercise_id'].nunique()}")
    
    # Step 2: Create dataloaders
    logger.info("\n[STEP 2] Creating PyTorch dataloaders...")
    
    train_loader, val_loader, dataset = create_dataloaders(
        df,
        batch_size=args.batch_size,
        train_split=0.8
    )
    
    num_features = len(dataset.get_feature_names())
    logger.info(f"  ✅ Created dataloaders")
    logger.info(f"     • Input features: {num_features}")
    logger.info(f"     • Batch size: {args.batch_size}")
    logger.info(f"     • Train batches: {len(train_loader)}")
    logger.info(f"     • Val batches: {len(val_loader)}")
    
    # Step 3: Create model
    logger.info("\n[STEP 3] Creating PyTorch model...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"  • Device: {device}")
    
    model = create_model(
        input_dim=num_features,
        hidden_dim=128,
        num_layers=2,
        num_heads=8,
        predict_horizons=4
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  ✅ Model created")
    logger.info(f"     • Architecture: LSTM + Attention + Bayesian")
    logger.info(f"     • Parameters: {num_params:,}")
    
    # Step 4: Setup trainer
    logger.info("\n[STEP 4] Setting up trainer...")
    
    trainer = StrengthPredictorTrainer(model, device=device)
    trainer.setup_optimizer(lr=args.lr, weight_decay=1e-5)
    
    logger.info(f"  ✅ Trainer ready")
    logger.info(f"     • Optimizer: AdamW(lr={args.lr})")
    logger.info(f"     • Scheduler: ReduceLROnPlateau")
    logger.info(f"     • Early stopping: 15 epochs patience")
    
    # Step 5: Train model
    logger.info("\n[STEP 5] Training model...")
    logger.info(f"  • Max epochs: {args.num_epochs}")
    logger.info(f"  • Target metric: Validation Weight MAE")
    logger.info("")
    
    start_time = datetime.now()
    history = trainer.train(
        train_loader,
        val_loader,
        num_epochs=args.num_epochs,
        early_stopping_patience=15
    )
    end_time = datetime.now()
    
    training_time = (end_time - start_time).total_seconds() / 60
    logger.info(f"  ✅ Training completed in {training_time:.1f} minutes")
    
    # Step 6: Save final model
    logger.info("\n[STEP 6] Saving trained model...")
    
    os.makedirs('ml/models', exist_ok=True)
    model_path = 'ml/models/strength_predictor.pt'
    trainer.save_final_model(model_path)
    
    logger.info(f"  ✅ Model saved to {model_path}")
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("TRAINING SUMMARY")
    logger.info("="*80)
    logger.info(f"Best Validation Loss: {trainer.best_val_loss:.4f}")
    logger.info(f"Best Weight MAE: {min(history['val_weight_mae']):.2f} kg")
    logger.info(f"Epochs trained: {len(history['train_loss'])}")
    logger.info(f"Total time: {training_time:.1f} minutes")
    logger.info(f"Model size: {os.path.getsize(model_path) / 1024 / 1024:.1f} MB")
    logger.info("")
    logger.info("✅ Ready for production!")
    logger.info("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train strength prediction model")
    
    parser.add_argument('--num_users', type=int, default=10000,
                       help='Number of synthetic users')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Training batch size')
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='Maximum number of epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    
    args = parser.parse_args()
    main(args)