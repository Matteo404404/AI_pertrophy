"""
Test Component 3: Simple Lifting Gains Predictor Validation

Testing the simplified lifting gains predictor that focuses on:
1. "Will I get stronger this week?"
2. "What RIR should I use?"
3. "How long should I rest?"

Author: Scientific Hypertrophy Trainer ML Team
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports  
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("=" * 80)
print("🧪 COMPONENT 3 TEST: SIMPLE LIFTING GAINS PREDICTOR")
print("=" * 80)

# Import the simple predictor
try:
    from ml.models.simple_lifting_predictor import SimpleLiftingPredictor, LiftingForecast, format_simple_recommendation
    print("✅ Simple Lifting Predictor imported successfully")
except ImportError as e:
    print(f"❌ Simple Lifting Predictor import failed: {e}")
    
    # Try alternative import
    try:
        import importlib.util
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        module_path = os.path.join(parent_dir, "ml", "models", "simple_lifting_predictor.py")
        
        spec = importlib.util.spec_from_file_location("simple_lifting_predictor", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        SimpleLiftingPredictor = module.SimpleLiftingPredictor
        LiftingForecast = module.LiftingForecast
        format_simple_recommendation = module.format_simple_recommendation
        print("✅ Simple Lifting Predictor imported successfully (direct path)")
        
    except Exception as e2:
        print(f"❌ All import methods failed: {e2}")
        print("Make sure ml/models/simple_lifting_predictor.py exists")
        sys.exit(1)

# Test data import
try:
    if os.path.exists("ml/data/synthetic_hypertrophy_data.csv"):
        df = pd.read_csv("ml/data/synthetic_hypertrophy_data.csv")
        df['date'] = pd.to_datetime(df['date'])
        print(f"✅ Loaded synthetic data: {len(df)} samples, {len(df.columns)} features")
    else:
        print("⚠️  Synthetic data not found - generating sample data")
        # Generate minimal test data
        dates = pd.date_range("2024-01-01", periods=90, freq='D')
        df = pd.DataFrame({
            'date': dates,
            'user_name': np.repeat(['Alex_Beginner', 'Sarah_Intermediate', 'Mike_Advanced'], 30),
            'total_sets': np.random.normal(12, 3, 90),
            'average_rpe': np.random.normal(7.5, 1, 90),
            'sleep_quality_1_10': np.random.normal(7, 1.5, 90),
            'sleep_duration_hours': np.random.normal(7.5, 1, 90),
            'hrv_rmssd': np.random.normal(45, 10, 90),
            'perceived_stress_1_10': np.random.normal(5, 1.5, 90)
        })
        # Clean up data
        df['total_sets'] = np.clip(df['total_sets'], 0, None)
        df['average_rpe'] = np.clip(df['average_rpe'], 1, 10)
        df['sleep_quality_1_10'] = np.clip(df['sleep_quality_1_10'], 1, 10)
        df['sleep_duration_hours'] = np.clip(df['sleep_duration_hours'], 4, 11)
        df['hrv_rmssd'] = np.clip(df['hrv_rmssd'], 20, 80)
        df['perceived_stress_1_10'] = np.clip(df['perceived_stress_1_10'], 1, 10)
        
except Exception as e:
    print(f"❌ Data loading failed: {e}")
    sys.exit(1)


def test_predictor_basic_functionality():
    """Test basic predictor functionality."""
    print("\n" + "="*60)
    print("🔧 TESTING BASIC FUNCTIONALITY")
    print("="*60)
    
    predictor = SimpleLiftingPredictor()
    
    # Test 1: Prediction Structure
    print("\n1️⃣ Testing Prediction Structure...")
    
    user_data = df[df['user_name'] == 'Alex_Beginner'].copy()
    forecast = predictor.predict_lifting_gains(user_data)
    
    print(f"✅ Forecast generated:")
    print(f"   Strength likelihood: {forecast.strength_gain_next_week}")
    print(f"   Confidence: {forecast.strength_confidence:.0f}%")
    print(f"   RIR recommendation: {forecast.recommended_rir}")
    print(f"   Rest time: {forecast.recommended_rest_seconds}s")
    
    # Validate forecast structure
    assert forecast.strength_gain_next_week in ['likely', 'possible', 'unlikely'], "Invalid strength prediction"
    assert 0 <= forecast.strength_confidence <= 100, "Confidence out of range"
    assert forecast.recommended_rir in ['0-1 RIR', '2-3 RIR', '3-4 RIR'], "Invalid RIR recommendation"
    assert forecast.recommended_rest_seconds in [120, 150, 180, 240], "Invalid rest time"
    assert isinstance(forecast.main_reason, str), "Reason must be string"
    
    # Test 2: Insufficient Data Handling
    print("\n2️⃣ Testing Insufficient Data Handling...")
    
    small_data = user_data.head(3)  # Only 3 days
    fallback_forecast = predictor.predict_lifting_gains(small_data)
    
    print(f"✅ Fallback prediction:")
    print(f"   Strength: {fallback_forecast.strength_gain_next_week}")
    print(f"   Warning: {fallback_forecast.warning}")
    
    assert fallback_forecast.strength_gain_next_week == "possible", "Fallback should be 'possible'"
    assert fallback_forecast.warning is not None, "Should have warning for insufficient data"
    
    print("✅ Basic functionality: ALL TESTS PASSED")
    return True


def test_prediction_logic():
    """Test prediction logic with different scenarios."""
    print("\n" + "="*60)
    print("🧠 TESTING PREDICTION LOGIC")
    print("="*60)
    
    predictor = SimpleLiftingPredictor()
    
    # Test 1: High Recovery Scenario
    print("\n1️⃣ Testing High Recovery Scenario...")
    
    # Create good recovery scenario
    good_recovery_data = pd.DataFrame({
        'total_sets': [12, 14, 13, 15, 16, 14, 0, 13, 15, 17, 16, 14, 18, 0],
        'average_rpe': [7.0, 7.2, 6.8, 7.5, 7.3, 7.1, 0, 6.9, 7.4, 7.6, 7.2, 7.0, 7.8, 0],
        'sleep_quality_1_10': [8, 9, 8, 8, 7, 9, 9, 8, 8, 7, 8, 9, 8, 9],
        'sleep_duration_hours': [8.2, 8.5, 8.0, 7.8, 8.1, 8.4, 9.0, 8.3, 8.0, 7.9, 8.2, 8.6, 8.1, 8.8],
        'hrv_rmssd': [52, 48, 50, 47, 49, 51, 55, 48, 50, 46, 49, 52, 48, 54],
        'perceived_stress_1_10': [3, 2, 4, 3, 2, 3, 2, 3, 4, 3, 2, 3, 4, 2]
    })
    
    good_forecast = predictor.predict_lifting_gains(good_recovery_data)
    
    print(f"✅ Good recovery prediction:")
    print(f"   Strength: {good_forecast.strength_gain_next_week}")
    print(f"   RIR: {good_forecast.recommended_rir}")
    print(f"   Rest: {good_forecast.recommended_rest_seconds}s")
    print(f"   Reason: {good_forecast.main_reason}")
    
    # Should predict gains are likely or possible with good recovery
    assert good_forecast.strength_gain_next_week in ['likely', 'possible'], "Good recovery should predict gains"
    
    # Test 2: Poor Recovery Scenario
    print("\n2️⃣ Testing Poor Recovery Scenario...")
    
    # Create poor recovery scenario
    poor_recovery_data = pd.DataFrame({
        'total_sets': [12, 14, 13, 15, 16, 14, 0, 13, 15, 17, 16, 14, 18, 0],
        'average_rpe': [8.5, 8.8, 8.2, 8.9, 8.6, 8.4, 0, 8.7, 8.9, 9.0, 8.8, 8.5, 9.2, 0],
        'sleep_quality_1_10': [4, 3, 5, 4, 3, 4, 5, 3, 4, 3, 4, 5, 3, 4],
        'sleep_duration_hours': [5.5, 5.8, 5.2, 5.9, 5.6, 5.4, 6.0, 5.7, 5.3, 5.8, 5.5, 5.9, 5.4, 5.7],
        'hrv_rmssd': [32, 28, 35, 31, 29, 33, 36, 30, 28, 32, 31, 34, 29, 33],
        'perceived_stress_1_10': [8, 9, 7, 8, 9, 8, 7, 8, 9, 8, 7, 8, 9, 7]
    })
    
    poor_forecast = predictor.predict_lifting_gains(poor_recovery_data)
    
    print(f"✅ Poor recovery prediction:")
    print(f"   Strength: {poor_forecast.strength_gain_next_week}")
    print(f"   RIR: {poor_forecast.recommended_rir}")
    print(f"   Rest: {poor_forecast.recommended_rest_seconds}s")
    print(f"   Warning: {poor_forecast.warning}")
    
    # Should recommend higher RIR and longer rest with poor recovery - FIXED TEST
    assert poor_forecast.recommended_rir in ['2-3 RIR', '3-4 RIR'], "Poor recovery should recommend higher RIR"
    assert poor_forecast.recommended_rest_seconds >= 240, "Poor recovery should recommend longer rest (240s)"  # FIXED
    assert poor_forecast.warning is not None, "Poor recovery should have warnings"
    
    print("✅ Prediction logic: ALL TESTS PASSED")
    return True


def test_all_users_predictions():
    """Test predictions for all user types."""
    print("\n" + "="*60)
    print("👥 TESTING ALL USERS PREDICTIONS")
    print("="*60)
    
    predictor = SimpleLiftingPredictor()
    
    users = df['user_name'].unique()
    
    for user in users:
        print(f"\n📊 Testing predictions for {user}...")
        
        user_data = df[df['user_name'] == user].copy()
        forecast = predictor.predict_lifting_gains(user_data)
        
        print(f"   Strength: {forecast.strength_gain_next_week} ({forecast.strength_confidence:.0f}%)")
        print(f"   RIR: {forecast.recommended_rir}")
        print(f"   Rest: {forecast.recommended_rest_seconds}s")
        
        if forecast.warning:
            print(f"   Warning: {forecast.warning}")
        
        # Validate each prediction
        assert forecast.strength_gain_next_week in ['likely', 'possible', 'unlikely'], f"Invalid prediction for {user}"
        assert 30 <= forecast.strength_confidence <= 90, f"Confidence out of range for {user}"
        assert forecast.recommended_rir in ['0-1 RIR', '2-3 RIR', '3-4 RIR'], f"Invalid RIR for {user}"
        assert forecast.recommended_rest_seconds in [120, 150, 180, 240], f"Invalid rest for {user}"
    
    print("✅ All users predictions: VALID")
    return True


def test_recommendation_formatting():
    """Test recommendation formatting for users."""
    print("\n" + "="*60)
    print("📱 TESTING RECOMMENDATION FORMATTING")
    print("="*60)
    
    predictor = SimpleLiftingPredictor()
    
    # Get sample prediction
    user_data = df[df['user_name'] == 'Alex_Beginner'].copy()
    forecast = predictor.predict_lifting_gains(user_data)
    
    # Format for user
    formatted = format_simple_recommendation(forecast)
    
    print("✅ Example formatted recommendation:")
    print("-" * 40)
    print(formatted)
    print("-" * 40)
    
    # Validate formatting
    assert "STRENGTH GAINS" in formatted, "Missing strength prediction"
    assert "RIR:" in formatted, "Missing RIR recommendation"
    assert "Rest:" in formatted, "Missing rest recommendation"
    assert "Why:" in formatted, "Missing reasoning"
    
    # Check emojis are present
    assert any(emoji in formatted for emoji in ["💪", "🤔", "😐"]), "Missing strength emoji"
    assert "📋" in formatted, "Missing recommendations emoji"
    assert "🧠" in formatted, "Missing reasoning emoji"
    
    print("✅ Recommendation formatting: PASSED")
    return True


def test_edge_cases():
    """Test edge cases and error handling."""
    print("\n" + "="*60)
    print("⚠️ TESTING EDGE CASES")
    print("="*60)
    
    predictor = SimpleLiftingPredictor()
    
    # Test 1: Empty DataFrame
    print("\n1️⃣ Testing Empty DataFrame...")
    
    empty_df = pd.DataFrame()
    empty_forecast = predictor.predict_lifting_gains(empty_df)
    
    assert empty_forecast.strength_gain_next_week == "possible", "Empty data should default to 'possible'"
    assert empty_forecast.warning is not None, "Empty data should have warning"
    print("✅ Empty DataFrame handled correctly")
    
    # Test 2: Missing Columns
    print("\n2️⃣ Testing Missing Columns...")
    
    minimal_data = pd.DataFrame({
        'total_sets': [12, 14, 13, 15, 16, 14, 0, 13, 15, 17, 16, 14, 18, 0]
    })
    
    minimal_forecast = predictor.predict_lifting_gains(minimal_data)
    
    assert minimal_forecast is not None, "Should handle missing columns gracefully"
    print("✅ Missing columns handled correctly")
    
    # Test 3: Extreme Values
    print("\n3️⃣ Testing Extreme Values...")
    
    extreme_data = pd.DataFrame({
        'total_sets': [0, 0, 0, 50, 45, 40, 0, 35, 30, 25, 0, 0, 0, 0],
        'average_rpe': [10, 10, 10, 1, 2, 3, 0, 4, 5, 6, 10, 10, 10, 0],
        'sleep_quality_1_10': [1, 1, 2, 10, 10, 9, 10, 8, 7, 6, 1, 1, 2, 1],
        'sleep_duration_hours': [3, 3, 4, 12, 11, 10, 11, 9, 8, 7, 3, 4, 3, 4],
        'hrv_rmssd': [15, 18, 20, 80, 75, 70, 75, 65, 60, 55, 15, 18, 20, 22],
        'perceived_stress_1_10': [10, 10, 9, 1, 1, 2, 1, 3, 4, 5, 10, 9, 10, 9]
    })
    
    extreme_forecast = predictor.predict_lifting_gains(extreme_data)
    
    assert extreme_forecast is not None, "Should handle extreme values"
    print("✅ Extreme values handled correctly")
    
    print("✅ Edge cases: ALL TESTS PASSED")
    return True


def main():
    """Run all Component 3 tests."""
    print("🚀 Starting Component 3 Simple Lifting Predictor tests...\n")
    
    results = []
    
    try:
        # Test 1: Basic Functionality
        results.append(test_predictor_basic_functionality())
        
        # Test 2: Prediction Logic
        results.append(test_prediction_logic())
        
        # Test 3: All Users
        results.append(test_all_users_predictions())
        
        # Test 4: Formatting
        results.append(test_recommendation_formatting())
        
        # Test 5: Edge Cases
        results.append(test_edge_cases())
        
        # Final Results
        print("\n" + "=" * 80)
        
        passed_tests = sum(results)
        total_tests = len(results)
        
        if passed_tests == total_tests:
            print("✅ ALL COMPONENT 3 TESTS PASSED!")
            print("✅ Basic functionality: Working correctly")
            print("✅ Prediction logic: Sound reasoning")
            print("✅ All users: Valid predictions")
            print("✅ Formatting: User-friendly output")
            print("✅ Edge cases: Robust error handling")
            
            print(f"\n🎯 COMPONENT 3 COMPLETE!")
            print("🚀 Simple Lifting Gains Predictor is PRODUCTION READY!")
            
            # Demo prediction
            print(f"\n📱 EXAMPLE PREDICTION:")
            predictor = SimpleLiftingPredictor()
            user_data = df[df['user_name'] == 'Alex_Beginner'].copy()
            forecast = predictor.predict_lifting_gains(user_data)
            formatted = format_simple_recommendation(forecast)
            print("-" * 50)
            print(formatted)
            print("-" * 50)
            
            print(f"\n✅ SCIENTIFIC HYPERTROPHY TRAINER COMPLETE!")
            print("💪 Ready to help gym bros get stronger!")
            
        else:
            print(f"❌ COMPONENT 3 TESTS: {passed_tests}/{total_tests} PASSED")
            for i, result in enumerate(results):
                status = "✅" if result else "❌"
                test_names = ["Basic Functionality", "Prediction Logic", "All Users", "Formatting", "Edge Cases"]
                print(f"{status} {test_names[i]}")
        
        print("=" * 80)
        return passed_tests == total_tests
        
    except Exception as e:
        print(f"\n❌ COMPONENT 3 TESTS FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    if success:
        print("\n🏆 ALL COMPONENTS COMPLETE!")
        print("📊 Component 1: TDEE Calculator")
        print("🏃‍♂️ Component 2: Synthetic Data Generator")  
        print("💪 Component 3: Simple Lifting Gains Predictor")
        print("\n✅ SCIENTIFIC HYPERTROPHY TRAINER: PRODUCTION READY!")
    else:
        print("\n❌ Fix issues before deployment")
        sys.exit(1)
