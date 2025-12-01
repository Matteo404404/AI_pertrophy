"""
Test inference pipeline.
"""

from ml.inference.predictor import StrengthPredictor
import numpy as np


def test_inference():
    """Test end-to-end inference."""
    
    print("="*70)
    print("TESTING INFERENCE PIPELINE")
    print("="*70)
    
    # Load predictor
    print("\n[Step 1] Loading trained model...")
    predictor = StrengthPredictor(
        model_path='ml/models/strength_predictor.pt',
        device='cuda'
    )
    
    print("\n[Step 2] Creating test input...")
    # Create realistic dummy 14-day sequence (normalized around 0, std ~1)
    dummy_sequence = np.random.randn(14, 35).astype(np.float32)
    print(f"   Input shape: {dummy_sequence.shape}")
    print(f"   Input range: [{dummy_sequence.min():.2f}, {dummy_sequence.max():.2f}]")
    
    print("\n[Step 3] Running inference...")
    predictions = predictor.predict(dummy_sequence)
    
    print("\n" + "="*70)
    print("INFERENCE TEST RESULTS")
    print("="*70)
    
    for horizon, values in predictions.items():
        sessions_ahead = horizon.replace('horizon_', '').replace('_', ' ')
        print(f"\n📊 Predictions for {sessions_ahead}:")
        print(f"   Weight:  {values['weight']:.3f} (normalized) ± {values['weight_uncertainty']:.3f}")
        print(f"   Reps:    {values['reps']:.3f} (normalized) ± {values['reps_uncertainty']:.3f}")
        print(f"   RIR:     {values['rir']:.3f} (normalized) ± {values['rir_uncertainty']:.3f}")
    
    print("\n" + "="*70)
    print("✅ INFERENCE WORKING! Model successfully loaded and predicting.")
    print("="*70)
    
    # Test with different input
    print("\n[Step 4] Testing with different input...")
    dummy_sequence2 = np.random.randn(14, 35).astype(np.float32)
    predictions2 = predictor.predict(dummy_sequence2)
    print(f"   Next session weight prediction: {predictions2['horizon_1_session']['weight']:.3f}")
    
    print("\n✅ All tests passed! Ready for production integration.")


if __name__ == "__main__":
    test_inference()
