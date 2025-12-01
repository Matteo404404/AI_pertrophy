"""Test denormalization of predictions."""

from ml.inference.predictor import StrengthPredictor
from ml.inference.preprocessor import InferencePreprocessor
import numpy as np


def test_full_pipeline():
    """Test complete inference + denormalization pipeline."""
    
    print("="*70)
    print("TESTING FULL PREDICTION PIPELINE")
    print("="*70)
    
    # Load predictor
    print("\n[Step 1] Loading model...")
    predictor = StrengthPredictor(
        model_path='ml/models/strength_predictor.pt',
        device='cuda'
    )
    
    # Load preprocessor
    print("\n[Step 2] Loading preprocessor...")
    preprocessor = InferencePreprocessor()
    
    # Create test input
    print("\n[Step 3] Running inference...")
    test_sequence = np.random.randn(14, 35).astype(np.float32)
    predictions_normalized = predictor.predict(test_sequence)
    
    # Denormalize
    print("\n[Step 4] Denormalizing predictions...")
    predictions_real = preprocessor.denormalize_predictions(predictions_normalized)
    
    print("\n" + "="*70)
    print("REAL-WORLD PREDICTIONS")
    print("="*70)
    
    for horizon, values in predictions_real.items():
        sessions = horizon.replace('horizon_', '').replace('_', ' ')
        print(f"\n📊 {sessions.title()}:")
        print(f"   Weight:  {values['weight_kg']:.1f} kg")
        print(f"            95% CI: [{values['weight_confidence_interval'][0]:.1f}, "
              f"{values['weight_confidence_interval'][1]:.1f}] kg")
        print(f"   Reps:    {values['reps']} reps")
        print(f"            95% CI: [{values['reps_confidence_interval'][0]}, "
              f"{values['reps_confidence_interval'][1]}] reps")
        print(f"   RIR:     {values['rir']}")
        print(f"            95% CI: [{values['rir_confidence_interval'][0]}, "
              f"{values['rir_confidence_interval'][1]}]")
    
    print("\n" + "="*70)
    print("✅ FULL PIPELINE WORKING!")
    print("   - Model loads and predicts")
    print("   - Denormalization converts to real units")
    print("   - Confidence intervals calculated")
    print("   - Ready for GUI integration!")
    print("="*70)


if __name__ == "__main__":
    test_full_pipeline()
