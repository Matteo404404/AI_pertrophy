"""
Train Strength Prediction Model

Complete training pipeline:
1. Generate synthetic data (5K users)
2. Create PyTorch dataloaders
3. Build and train LSTM model
4. Save trained model & stats

Usage:
    python train_model.py --num_users 5000 --num_epochs 50
"""

import argparse
import torch
import pandas as pd
import logging
import os
import sys
import json
from datetime import datetime

# --- PATH SETUP ---
# Get the absolute path to the project root (where this script is)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

# Define output directories relative to root
ML_ENGINE_DIR = os.path.join(BASE_DIR, 'ml_engine')
DATA_DIR = os.path.join(ML_ENGINE_DIR, 'data')
MODELS_DIR = os.path.join(ML_ENGINE_DIR, 'models')

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# --- IMPORTS ---
try:
    # Try importing from research_lab (standard structure)
    from research_lab.generator.enhanced_synthetic_generator import EnhancedSyntheticDataGenerator
    from research_lab.training.dataset import create_dataloaders
    from research_lab.training.trainer import StrengthPredictorTrainer
except ImportError as e:
    raise ImportError(
        f"Could not import from research_lab: {e}. "
        "Ensure research_lab/generator/enhanced_synthetic_generator.py, "
        "research_lab/training/dataset.py, and research_lab/training/trainer.py exist."
    )

from ml_engine.models.pytorch_strength_predictor import create_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main(args):
    """Main training pipeline."""
    
    logger.info("="*80)
    logger.info(f"STRENGTH PREDICTION MODEL - TRAINING PIPELINE")
    logger.info(f"Root Directory: {BASE_DIR}")
    logger.info("="*80)
    
    # --- Step 1: Generate synthetic data ---
    logger.info("\n[STEP 1] Checking training data...")
    
    training_data_path = os.path.join(DATA_DIR, 'training_data.csv')
    
    if os.path.exists(training_data_path) and not args.regenerate:
        logger.info(f"  • Found existing data at: {training_data_path}")
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
    
    # CRITICAL: Save normalization stats so the GUI can use the model later
    stats_path = os.path.join(DATA_DIR, 'normalization_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(dataset.get_normalization_stats(), f, indent=4)
    logger.info(f"  💾 Saved normalization stats to: {stats_path}")
    
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
    
    # Checkpoints go to research lab (temp storage)
    checkpoint_dir = os.path.join(BASE_DIR, 'research_lab', 'training', 'checkpoints')
    
    trainer = StrengthPredictorTrainer(
        model, 
        device=device,
        checkpoint_dir=checkpoint_dir
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
    model_path = os.path.join(MODELS_DIR, 'strength_predictor.pt')
    trainer.save_final_model(model_path)
    
    logger.info(f"  ✅ Model saved to: {model_path}")
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