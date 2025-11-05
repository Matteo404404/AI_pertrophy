"""
Task 3 Testing Suite

Tests the complete deep learning pipeline:
1. Synthetic data generation
2. Dataset creation
3. Model initialization
4. Training loop
5. Inference

Usage:
    python test_task3.py --quick  # Fast test (100 users)
    python test_task3.py          # Full test (1000 users)
"""

import sys
import os
import torch
import logging
import argparse
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project to path
sys.path.insert(0, os.path.dirname(__file__))

from ml.data.enhanced_synthetic_generator import EnhancedSyntheticDataGenerator
from ml.training.dataset import create_dataloaders
from ml.models.pytorch_strength_predictor import create_model
from ml.training.trainer import StrengthPredictorTrainer


def test_synthetic_data_generation(num_users=100):
    """Test synthetic data generation."""
    print("\n" + "="*80)
    print("TEST 1: Synthetic Data Generation")
    print("="*80)
    
    try:
        logger.info(f"Generating synthetic data for {num_users} users...")
        generator = EnhancedSyntheticDataGenerator(num_users=num_users, days_history=90)
        df = generator.generate_dataset()
        
        assert len(df) > 0, "Generated empty dataset"
        assert df['user_id'].nunique() == num_users, "User count mismatch"
        
        logger.info(f"✅ Generated {len(df)} records")
        logger.info(f"   • Users: {df['user_id'].nunique()}")
        logger.info(f"   • Exercises: {df['exercise_id'].nunique()}")
        logger.info(f"   • Weight range: {df['weight_kg'].min():.1f} - {df['weight_kg'].max():.1f} kg")
        logger.info(f"   • Reps range: {df['reps'].min()} - {df['reps'].max()}")
        
        return True, df
    
    except Exception as e:
        logger.error(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_dataset_creation(df):
    """Test PyTorch dataset creation."""
    print("\n" + "="*80)
    print("TEST 2: PyTorch Dataset Creation")
    print("="*80)
    
    try:
        logger.info("Creating dataloaders...")
        train_loader, val_loader, dataset = create_dataloaders(
            df, batch_size=16, train_split=0.8
        )
        
        # Test batch
        batch_x, batch_y = next(iter(train_loader))
        
        logger.info(f"✅ Dataset created successfully")
        logger.info(f"   • Features: {len(dataset.get_feature_names())}")
        logger.info(f"   • Train sequences: {len(train_loader)}")
        logger.info(f"   • Val sequences: {len(val_loader)}")
        logger.info(f"   • Batch input shape: {batch_x.shape}")
        logger.info(f"   • Batch output shape: {batch_y.shape}")
        
        return True, (train_loader, val_loader, dataset)
    
    except Exception as e:
        logger.error(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_model_creation(num_features):
    """Test model creation."""
    print("\n" + "="*80)
    print("TEST 3: Model Architecture")
    print("="*80)
    
    try:
        logger.info("Creating PyTorch model...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        model = create_model(
            input_dim=num_features,
            hidden_dim=128,
            num_layers=2,
            num_heads=8,
            predict_horizons=4
        )
        
        model.to(device)
        num_params = sum(p.numel() for p in model.parameters())
        
        logger.info(f"✅ Model created")
        logger.info(f"   • Device: {device}")
        logger.info(f"   • Parameters: {num_params:,}")
        logger.info(f"   • Architecture: LSTM + Attention + Bayesian")
        
        return True, (model, device)
    
    except Exception as e:
        logger.error(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_forward_pass(model, device, batch):
    """Test forward pass."""
    print("\n" + "="*80)
    print("TEST 4: Forward Pass")
    print("="*80)
    
    try:
        model.eval()
        batch_x, batch_y = batch
        batch_x = batch_x.to(device)
        
        logger.info("Running forward pass...")
        with torch.no_grad():
            predictions, uncertainties = model(batch_x)
        
        logger.info(f"✅ Forward pass successful")
        logger.info(f"   • Predictions shape: {predictions.shape}")
        logger.info(f"   • Uncertainties shape: {uncertainties.shape}")
        logger.info(f"   • Prediction range: [{predictions.min():.2f}, {predictions.max():.2f}]")
        
        return True, (predictions, uncertainties)
    
    except Exception as e:
        logger.error(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_training_loop(model, device, train_loader, val_loader):
    """Test training loop."""
    print("\n" + "="*80)
    print("TEST 5: Training Loop (1 Epoch)")
    print("="*80)
    
    try:
        logger.info("Initializing trainer...")
        trainer = StrengthPredictorTrainer(model, device=device)
        trainer.setup_optimizer(lr=0.001)
        
        logger.info("Training for 1 epoch...")
        train_metrics = trainer.train_epoch(train_loader)
        
        logger.info(f"✅ Training step successful")
        logger.info(f"   • Train loss: {train_metrics['train_loss']:.4f}")
        logger.info(f"   • Weight MAE: {train_metrics['train_weight_mae']:.2f} kg")
        
        logger.info("Validating...")
        val_metrics = trainer.validate(val_loader)
        
        logger.info(f"✅ Validation successful")
        logger.info(f"   • Val loss: {val_metrics['val_loss']:.4f}")
        logger.info(f"   • Weight MAE: {val_metrics['val_weight_mae']:.2f} kg")
        
        return True, trainer
    
    except Exception as e:
        logger.error(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_inference(model, device, batch):
    """Test inference with uncertainty."""
    print("\n" + "="*80)
    print("TEST 6: Inference with Uncertainty")
    print("="*80)
    
    try:
        batch_x, _ = batch
        batch_x = batch_x.to(device)
        
        logger.info("Running inference with MC dropout...")
        results = model.predict(batch_x, num_samples=5)
        
        logger.info(f"✅ Inference successful")
        logger.info(f"   • Predictions: {results['predictions'].shape}")
        logger.info(f"   • Aleatoric uncertainty: {results['aleatoric_uncertainty'].shape}")
        logger.info(f"   • Epistemic uncertainty: {results['epistemic_uncertainty'].shape}")
        logger.info(f"   • Confidence: {results['confidence'].mean():.2%}")
        
        return True, results
    
    except Exception as e:
        logger.error(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def main(args):
    """Run all tests."""
    
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*78 + "║")
    print("║" + "TASK 3: DEEP LEARNING MODEL - TEST SUITE".center(78) + "║")
    print("║" + " "*78 + "║")
    print("╚" + "="*78 + "╝")
    
    num_users = 100 if args.quick else 500
    
    # Test 1: Data generation
    success, df = test_synthetic_data_generation(num_users)
    if not success:
        return
    
    # Test 2: Dataset creation
    success, loaders_data = test_dataset_creation(df)
    if not success:
        return
    train_loader, val_loader, dataset = loaders_data
    
    # Test 3: Model creation
    num_features = len(dataset.get_feature_names())
    success, model_data = test_model_creation(num_features)
    if not success:
        return
    model, device = model_data
    
    # Test 4: Forward pass
    batch = next(iter(train_loader))
    success, _ = test_forward_pass(model, device, batch)
    if not success:
        return
    
    # Test 5: Training loop
    success, trainer = test_training_loop(model, device, train_loader, val_loader)
    if not success:
        return
    
    # Test 6: Inference
    success, results = test_inference(model, device, batch)
    if not success:
        return
    
    # Summary
    print("\n" + "="*80)
    print("✅ ALL TESTS PASSED!")
    print("="*80)
    print("\nTask 3 is complete and ready for full training with:")
    print(f"  • python train_model.py               (train with 10K users)")
    print(f"  • python train_model.py --num_users 5000  (smaller test run)")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task 3 testing suite")
    parser.add_argument('--quick', action='store_true', help='Quick test (100 users)')
    
    args = parser.parse_args()
    main(args)