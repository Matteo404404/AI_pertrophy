"""
Test Task 2: Collaborative Filtering System

This script tests the user embedding and collaborative filtering implementation.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from database.db_manager import DatabaseManager
from ml.models.user_embedding import UserEmbeddingCalculator
from ml.models.collaborative_filter import CollaborativeFilteringEngine
import numpy as np


def test_user_embedding():
    """Test user embedding calculation"""
    print("=" * 60)
    print("TEST 1: User Embedding Calculation")
    print("=" * 60)
    
    db = DatabaseManager()
    embedding_calc = UserEmbeddingCalculator(db)
    
    # Get first user
    users = db.get_all_users()
    if len(users) == 0:
        print("❌ No users found in database")
        return False
    
    user_id = users[0]['id']
    print(f"\n✅ Testing with user: {users[0]['username']} (ID: {user_id})")
    
    # Compute embedding
    try:
        embedding, components, completeness = embedding_calc.compute_user_embedding(user_id)
        
        print(f"\n📊 Embedding computed successfully!")
        print(f"   • Vector shape: {embedding.shape}")
        print(f"   • Data completeness: {completeness:.1%}")
        print(f"\n   Component breakdown:")
        print(f"   • Demographics (4): {components['demographics']}")
        print(f"   • Diet patterns (5): {components['diet_patterns'][:3]}...")  # Show first 3
        print(f"   • Sleep patterns (3): {components['sleep_patterns']}")
        print(f"   • Training style (6): {components['training_style'][:3]}...")
        print(f"   • Knowledge (2): {components['knowledge']}")
        
        # Store in database
        embedding_calc.update_user_embedding(user_id)
        print(f"\n✅ Embedding stored in database")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to compute embedding: {e}")
        return False


def test_similarity_matching():
    """Test finding similar users"""
    print("\n" + "=" * 60)
    print("TEST 2: User Similarity Matching")
    print("=" * 60)
    
    db = DatabaseManager()
    embedding_calc = UserEmbeddingCalculator(db)
    cf_engine = CollaborativeFilteringEngine(db, embedding_calc)
    
    users = db.get_all_users()
    if len(users) < 2:
        print("❌ Need at least 2 users for similarity testing")
        return False
    
    user_id = users[0]['id']
    print(f"\n✅ Finding similar users to: {users[0]['username']}")
    
    try:
        # Update embeddings for all users first
        print("\n   Computing embeddings for all users...")
        for user in users:
            embedding_calc.update_user_embedding(user['id'])
        
        # Find similar users
        similar_users = cf_engine.find_similar_users(user_id, top_k=5, min_completeness=0.0)
        
        print(f"\n📊 Found {len(similar_users)} similar users:")
        for i, similar in enumerate(similar_users, 1):
            similar_user_data = db.get_user_by_id(similar['user_id'])
            print(f"\n   {i}. {similar_user_data['username']}")
            print(f"      • Similarity: {similar['similarity']:.3f}")
            print(f"      • Data completeness: {similar['completeness']:.1%}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to find similar users: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_predictions():
    """Test generating predictions"""
    print("\n" + "=" * 60)
    print("TEST 3: Collaborative Filtering Predictions")
    print("=" * 60)
    
    db = DatabaseManager()
    embedding_calc = UserEmbeddingCalculator(db)
    cf_engine = CollaborativeFilteringEngine(db, embedding_calc)
    
    users = db.get_all_users()
    user_id = users[0]['id']
    
    print(f"\n✅ Generating predictions for: {users[0]['username']}")
    
    # Get an exercise to predict
    exercises = db.get_all_exercises()
    if len(exercises) == 0:
        print("❌ No exercises found")
        return False
    
    exercise = exercises[0]
    exercise_id = exercise['id']
    
    print(f"   Exercise: {exercise['name']}")
    
    try:
        # Generate predictions
        predictions = cf_engine.predict_for_exercise(
            user_id, exercise_id, sessions_ahead=[1, 2, 4, 10]
        )
        
        print(f"\n📊 Predictions generated:")
        
        horizons = {
            1: "Next workout (3 days)",
            2: "2 workouts ahead (6 days)",
            4: "4 workouts ahead (12 days)",
            10: "1 month ahead (30 days)"
        }
        
        for horizon, pred in predictions.items():
            print(f"\n   {horizons.get(horizon, f'{horizon} sessions ahead')}:")
            print(f"      • Weight: {pred['weight_kg']}kg")
            print(f"      • Reps: {pred['reps']}")
            print(f"      • RIR: {pred['rir']}")
            print(f"      • Confidence: {pred['confidence']:.0%}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to generate predictions: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_workflow():
    """Test complete workflow"""
    print("\n" + "=" * 60)
    print("TEST 4: Complete Prediction Workflow")
    print("=" * 60)
    
    db = DatabaseManager()
    
    users = db.get_all_users()
    user_id = users[0]['id']
    
    print(f"\n✅ Running full prediction workflow for: {users[0]['username']}")
    
    try:
        from ml.models.collaborative_filter import generate_predictions_for_new_user
        
        # Generate predictions for all exercises
        all_predictions = generate_predictions_for_new_user(db, user_id)
        
        print(f"\n📊 Generated predictions for {len(all_predictions)} exercises:")
        
        # Show first 3 exercises
        for i, (exercise_id, horizons) in enumerate(list(all_predictions.items())[:3], 1):
            exercise = db.get_exercise_by_id(exercise_id)
            print(f"\n   {i}. {exercise['name']}:")
            
            # Show just next workout prediction
            if 1 in horizons:
                pred = horizons[1]
                print(f"      Next workout: {pred['weight_kg']}kg × {pred['reps']} reps @ RIR {pred['rir']}")
                print(f"      Confidence: {pred['confidence']:.0%}")
        
        if len(all_predictions) > 3:
            print(f"\n   ... and {len(all_predictions) - 3} more exercises")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed full workflow: {e}")
        import traceback
        traceback.print_exc()
        return False


def add_sample_workout_data():
    """Add sample workout data for testing"""
    print("\n" + "=" * 60)
    print("SETUP: Adding Sample Workout Data")
    print("=" * 60)
    
    db = DatabaseManager()
    
    users = db.get_all_users()
    exercises = db.get_all_exercises()
    
    if len(users) < 2 or len(exercises) < 3:
        print("❌ Need at least 2 users and 3 exercises")
        return False
    
    print("\n✅ Adding sample workout data...")
    
    # Add some performance history for each user
    from datetime import datetime, timedelta
    
    for user in users[:3]:  # First 3 users
        user_id = user['id']
        print(f"\n   Adding data for {user['username']}...")
        
        # Add 5 workout sessions
        for day in range(5):
            session_date = (datetime.now() - timedelta(days=day*3)).strftime('%Y-%m-%d')
            
            # Add performance for 3 exercises
            for ex_idx, exercise in enumerate(exercises[:3]):
                exercise_id = exercise['id']
                
                # Simulate progressive overload
                base_weight = 50 + (user_id * 10) + (ex_idx * 5)
                weight = base_weight + (5 - day) * 2.5  # More recent = heavier
                reps = 10 - day  # More recent = fewer reps (harder sets)
                rir = 2 + (day % 3)  # Vary RIR
                
                db.store_exercise_performance(
                    user_id=user_id,
                    exercise_id=exercise_id,
                    session_date=session_date,
                    weight_kg=weight,
                    reps=reps,
                    rir=rir,
                    total_sets=3,
                    exercise_order=ex_idx + 1
                )
    
    print("\n✅ Sample data added successfully!")
    return True


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🧪 TASK 2: COLLABORATIVE FILTERING TEST SUITE")
    print("=" * 60)
    
    # Setup: Add sample data
    if not add_sample_workout_data():
        print("\n❌ Setup failed - cannot continue tests")
        sys.exit(1)
    
    # Run tests
    tests = [
        ("User Embedding", test_user_embedding),
        ("Similarity Matching", test_similarity_matching),
        ("Predictions", test_predictions),
        ("Full Workflow", test_full_workflow)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n❌ Test '{test_name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    total = len(results)
    passed = sum(1 for _, success in results if success)
    
    print(f"\n🎯 Results: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
    
    if passed == total:
        print("\n🎉 All tests passed! Task 2 is complete!")
    else:
        print("\n⚠️  Some tests failed - check errors above")
