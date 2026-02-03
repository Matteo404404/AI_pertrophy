"""
Train Strength Prediction Model

Complete training pipeline:
1. Generate synthetic data (5K users)
2. Create PyTorch dataloaders
3. Build and train LSTM model
4. Save trained model

Usage:
    python train_model.py --num_users 5000 --num_epochs 50

Author: AI_PERTROPHY - Task 3
"""

import argparse
import torch
import pandas as pd
import logging
import os
import sys
from datetime import datetime

# --- IMPORT FIXES ---
# We need to add the current directory to sys.path to ensure imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 1. Generator is in research_lab
try:
    from research_lab.generator.enhanced_synthetic_generator import EnhancedSyntheticDataGenerator
    from research_lab.training.dataset import create_dataloaders
    from research_lab.training.trainer import StrengthPredictorTrainer
except ImportError:
    # Fallback if files were moved differently
    from ml_engine.data.enhanced_synthetic_generator import EnhancedSyntheticDataGenerator
    # You might need to adjust these if your specific file moves were different, 
    # but based on standard structure, the above should work.

# 2. Model Architecture is in ml_engine (shared with the app)
from ml_engine.models.pytorch_strength_predictor import create_model
# --------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main(args):
    """Main training pipeline."""
    
    logger.info("="*80)
    logger.info("STRENGTH PREDICTION MODEL - TRAINING PIPELINE")
    logger.info("="*80)
    
    # --- Step 1: Generate synthetic data ---
    logger.info("\n[STEP 1] Checking training data...")
    # Save data to ml_engine/data so the app can find normalization stats later
    os.makedirs('ml_engine/data', exist_ok=True)
    training_data_path = 'ml_engine/data/training_data.csv'
    
    if os.path.exists(training_data_path) and not args.regenerate:
        logger.info(f"  • Found existing data at {training_data_path}")
        df = pd.read_csv(training_data_path)
        logger.info(f"  ✅ Loaded {len(df)} records")
    else:
        logger.info("  • Generating NEW synthetic data...")
        logger.info(f"  • Users: {args.num_users}")
        
        generator = EnhancedSyntheticDataGenerator(
            num_users=args.num_users,
            days_history=90,
            seed=42
        )
        
        df = generator.generate_dataset()
        generator.save_dataset(df, training_data_path)
        
        logger.info(f"  ✅ Generated {len(df)} records")
    
    # --- Step 2: Create dataloaders ---
    logger.info("\n[STEP 2] Creating PyTorch dataloaders...")
    
    train_loader, val_loader, dataset = create_dataloaders(
        df,
        batch_size=args.batch_size,
        train_split=0.8
    )
    
    # Save normalization stats immediately so we don't forget
    import json
    stats_path = 'ml_engine/data/normalization_stats.json'
    with open(stats_path, 'w') as f:
        json.dump(dataset.get_normalization_stats(), f, indent=4)
    logger.info(f"  💾 Saved normalization stats to {stats_path}")
    
    # --- Step 3: Create model ---
    logger.info("\n[STEP 3] Creating PyTorch model...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"  🚀 Device: {device}")
    
    model = create_model(
        input_dim=35,       # Explicit 35 features
        hidden_dim=128,
        num_layers=2,
        num_heads=8,
        predict_horizons=4
    )
    
    model = model.to(device)
    
    # --- Step 4: Setup trainer ---
    logger.info("\n[STEP 4] Setting up trainer...")
    
    # Save checkpoints to research_lab, final model to ml_engine
    trainer = StrengthPredictorTrainer(
        model, 
        device=device,
        checkpoint_dir='research_lab/training/checkpoints'
    )
    trainer.setup_optimizer(lr=args.lr)
    
    # --- Step 5: Train model ---
    logger.info("\n[STEP 5] Training model...")
    
    start_time = datetime.now()
    history = trainer.train(
        train_loader,
        val_loader,
        num_epochs=args.num_epochs,
        early_stopping_patience=10
    )
    end_time = datetime.now()
    
    training_time = (end_time - start_time).total_seconds() / 60
    logger.info(f"  ✅ Training completed in {training_time:.1f} minutes")
    
    # --- Step 6: Save final model ---
    logger.info("\n[STEP 6] Saving trained model...")
    
    # Save to ml_engine/models so the APP can use it immediately
    os.makedirs('ml_engine/models', exist_ok=True)
    model_path = 'ml_engine/models/strength_predictor.pt'
    trainer.save_final_model(model_path)
    
    logger.info(f"  ✅ Model saved to {model_path}")
    logger.info("="*80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train strength prediction model")
    
    parser.add_argument('--num_users', type=int, default=2000, help='Number of synthetic users')
    parser.add_argument('--batch_size', type=int, default=128, help='Training batch size')
    parser.add_argument('--num_epochs', type=int, default=100, help='Maximum number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--regenerate', action='store_true', help='Force new data generation')
    
    args = parser.parse_args()
    main(args)