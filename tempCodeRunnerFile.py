"""
Test inference pipeline.
"""

from ml.inference.predictor import StrengthPredictor
from ml.inference.preprocessor import InferencePreprocessor
import numpy as np


def test_inference():
    """Test end-to-end inference."""
    
    # Load predictor
    predictor = StrengthPredictor(
        model_path='ml/models/strength_predictor.pt',
        device='cuda'
    )
    
    # Create dummy 14-day sequence (normalized)
    dummy_sequence = np.random.randn(14, 35)
    
    # Get predictions
    predictions = predictor.predict(dummy_sequence)
    
    print("Inference Test Results:")
    print("="*60)
    
    for horizon, values in predictions.items():
        print(f"\n{horizon}:")
        print(f"  Weight: {values['weight']:.2f} ± {values['weight_uncertainty']:.2f}")
        print(f"  Reps: {values['reps']:.0f}")
        print(f"  RIR: {values['rir']:.0f}")
    
    print("\n✅ Inference working!")


if __name__ == "__main__":
    test_inference()
