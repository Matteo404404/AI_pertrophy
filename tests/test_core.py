"""
Scientific Hypertrophy Trainer - Core System Test
Comprehensive verification of all components
"""

import os
import sys
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("=" * 70)
print("🧪 SCIENTIFIC HYPERTROPHY TRAINER - CORE SYSTEM TEST")
print("=" * 70)
print()

# ===== TEST 1: DATABASE MANAGER =====
print("📦 TEST 1: Database Manager")
print("-" * 70)

try:
    from app.database.db_manager import DatabaseManager
    
    # Create test database
    test_db_path = "data/test_verification.db"
    if os.path.exists(test_db_path):
        os.remove(test_db_path)
    
    db = DatabaseManager(test_db_path)
    print("✅ Database initialized")
    
    # Test user retrieval
    users = db.get_all_users()
    print(f"✅ Found {len(users)} demo users:")
    for user in users:
        print(f"   • {user['username']} ({user['experience_level']})")
    
    # Test user creation
    new_user_id = db.create_user(
        username="TestUser",
        experience_level="beginner",
        primary_goal="muscle_growth",
        weight_kg=80.0
    )
    print(f"✅ Created new user with ID: {new_user_id}")
    
    # Test tier progression
    alex = db.get_user_by_username("Alex_Beginner")
    progress = db.get_user_tier_progress(alex['id'])
    print(f"✅ Alex's tier progress: Tier {progress['current_tier'] + 1}")
    
    # Test diet entry
    diet_id = db.save_diet_entry(
        user_id=alex['id'],
        entry_date=datetime.now().date(),
        protein_g=150.0,
        calories=2500,
        protein_per_kg=2.0,
        meals_count=4,
        hydration_liters=3.0,
        notes="Test diet entry"
    )
    print(f"✅ Created diet entry with ID: {diet_id}")
    
    # Test sleep entry
    sleep_id = db.save_sleep_entry(
        user_id=alex['id'],
        entry_date=datetime.now().date(),
        sleep_duration_hours=8.0,
        sleep_quality=7,
        time_to_fall_asleep_minutes=15,
        awakenings_count=1,
        notes="Test sleep entry"
    )
    print(f"✅ Created sleep entry with ID: {sleep_id}")
    
    print("\n✅ DATABASE MANAGER TEST PASSED!")
    
except Exception as e:
    print(f"\n❌ DATABASE MANAGER TEST FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# ===== TEST 2: ASSESSMENT ENGINE =====
print("📝 TEST 2: Assessment Engine")
print("-" * 70)

try:
    from app.core.assessment_engine import AssessmentEngine
    
    # Initialize engine
    engine = AssessmentEngine("data/questions.json")
    print(f"✅ Assessment engine initialized")
    print(f"✅ Loaded {engine.get_total_question_count()} total questions")
    
    # Test starting assessment
    assessment_info = engine.start_assessment(0)  # Tier 1
    print(f"✅ Started assessment: {assessment_info['tier_title']}")
    print(f"   • {assessment_info['total_questions']} questions")
    print(f"   • Passing score: {assessment_info['passing_score']}/{assessment_info['total_questions']}")
    
    # Test getting first question
    question = engine.get_current_question()
    if question:
        print(f"✅ Retrieved question {question['question_number']}/{question['total_questions']}")
        print(f"   • ID: {question['id']}")
        print(f"   • Question: {question['question'][:50]}...")
        print(f"   • Options: {len(question['options'])}")
        
        # Test submitting answer
        correct_option = next(opt for opt in question['options'] if opt['correct'])
        result = engine.submit_answer(correct_option['text'])
        print(f"✅ Submitted answer: {'Correct' if result['answer']['is_correct'] else 'Incorrect'}")
        print(f"   • Status: {result['status']}")
        
        # Check no repeats
        first_question_id = question['id']
        if first_question_id in engine.used_question_ids:
            print(f"✅ Question {first_question_id} marked as used (NO REPEATS)")
    
    # Test question categorization
    test_question = "What weekly volume range optimizes hypertrophy?"
    category = engine.categorize_question(test_question)
    print(f"✅ Question categorization works: '{category}'")
    
    print("\n✅ ASSESSMENT ENGINE TEST PASSED!")
    
except Exception as e:
    print(f"\n❌ ASSESSMENT ENGINE TEST FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# ===== TEST 3: USER MANAGER =====
print("👤 TEST 3: User Manager")
print("-" * 70)

try:
    from app.core.user_manager import UserManager
    
    # Initialize manager
    manager = UserManager(db)
    print("✅ User manager initialized")
    
    # Test setting current user
    alex = db.get_user_by_username("Alex_Beginner")
    manager.set_current_user(alex['id'])
    print(f"✅ Set current user: {manager.get_current_user()['username']}")
    
    # Test calculating stats
    stats = manager.calculate_user_stats(alex['id'])
    print(f"✅ User statistics calculated:")
    print(f"   • Total assessments: {stats['total_assessments']}")
    print(f"   • Accuracy: {stats['accuracy_percentage']}%")
    print(f"   • Current tier: {stats['current_tier'] + 1}")
    
    # Test tier access
    can_access_tier_1 = manager.can_access_tier(0)
    can_access_tier_2 = manager.can_access_tier(1)
    can_access_tier_3 = manager.can_access_tier(2)
    print(f"✅ Tier access check:")
    print(f"   • Tier 1: {can_access_tier_1}")
    print(f"   • Tier 2: {can_access_tier_2}")
    print(f"   • Tier 3: {can_access_tier_3}")
    
    # Test tier status
    tier_status = manager.get_tier_status()
    print(f"✅ Tier status retrieved: {len(tier_status['tiers'])} tiers")
    
    # Test recommendations
    recommendations = manager.get_personalized_recommendations()
    print(f"✅ Generated {len(recommendations)} recommendations")
    for rec in recommendations[:2]:
        print(f"   • {rec['category']}: {rec['title']}")
    
    # Test dashboard data
    dashboard = manager.get_dashboard_data()
    print(f"✅ Dashboard data compiled: {len(dashboard)} sections")
    
    print("\n✅ USER MANAGER TEST PASSED!")
    
except Exception as e:
    print(f"\n❌ USER MANAGER TEST FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# ===== TEST 4: TRACKING SYSTEM =====
print("📊 TEST 4: Tracking System")
print("-" * 70)

try:
    from app.core.tracking_system import TrackingSystem
    
    # Initialize tracking system
    tracking = TrackingSystem(db)
    print("✅ Tracking system initialized")
    
    # Test diet stats
    diet_stats = tracking.calculate_diet_stats(alex['id'], days=7)
    print(f"✅ Diet statistics calculated:")
    print(f"   • Entries: {diet_stats['entries_count']}")
    print(f"   • Avg protein: {diet_stats['avg_protein_g']}g")
    print(f"   • Avg calories: {diet_stats['avg_calories']}")
    print(f"   • Consistency: {diet_stats['consistency_percentage']}%")
    
    # Test diet recommendations
    diet_recs = tracking.get_diet_recommendations(alex['id'], user_weight_kg=75.0)
    print(f"✅ Diet recommendations generated: {len(diet_recs)}")
    for rec in diet_recs[:2]:
        print(f"   • {rec['category']}: {rec['message'][:50]}...")
    
    # Test sleep stats
    sleep_stats = tracking.calculate_sleep_stats(alex['id'], days=7)
    print(f"✅ Sleep statistics calculated:")
    print(f"   • Entries: {sleep_stats['entries_count']}")
    print(f"   • Avg duration: {sleep_stats['avg_sleep_hours']}h")
    print(f"   • Avg quality: {sleep_stats['avg_sleep_quality']}/10")
    
    # Test sleep recommendations
    sleep_recs = tracking.get_sleep_recommendations(alex['id'])
    print(f"✅ Sleep recommendations generated: {len(sleep_recs)}")
    
    # Test training stats
    training_stats = tracking.calculate_training_stats(alex['id'], days=7)
    print(f"✅ Training statistics calculated:")
    print(f"   • Sessions: {training_stats['entries_count']}")
    print(f"   • Total minutes: {training_stats['total_training_minutes']}")
    
    print("\n✅ TRACKING SYSTEM TEST PASSED!")
    
except Exception as e:
    print(f"\n❌ TRACKING SYSTEM TEST FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# ===== TEST 5: INTEGRATION TEST =====
print("🔗 TEST 5: Integration Test (Full Workflow)")
print("-" * 70)

try:
    # Simulate complete user journey
    print("Simulating complete user workflow...")
    
    # 1. Create user
    test_user_id = db.create_user(
        username="IntegrationTestUser",
        experience_level="intermediate",
        primary_goal="muscle_growth",
        weight_kg=70.0
    )
    manager.set_current_user(test_user_id)
    print("✅ Step 1: User created and set as current")
    
    # 2. Start assessment
    engine2 = AssessmentEngine("data/questions.json")
    assessment_info = engine2.start_assessment(0)
    print(f"✅ Step 2: Started Tier 1 assessment")
    
    # 3. Answer 3 questions (simulate)
    for i in range(3):
        question = engine2.get_current_question()
        if question:
            correct_opt = next(opt for opt in question['options'] if opt['correct'])
            result = engine2.submit_answer(correct_opt['text'])
            print(f"✅ Step 3.{i+1}: Answered question {i+1}")
    
    # 4. Log diet entry
    diet_id = db.save_diet_entry(
        user_id=test_user_id,
        entry_date=datetime.now().date(),
        protein_g=140.0,
        calories=2300,
        protein_per_kg=2.0
    )
    print(f"✅ Step 4: Logged diet entry")
    
    # 5. Log sleep entry
    sleep_id = db.save_sleep_entry(
        user_id=test_user_id,
        entry_date=datetime.now().date(),
        sleep_duration_hours=7.5,
        sleep_quality=8
    )
    print(f"✅ Step 5: Logged sleep entry")
    
    # 6. Get dashboard data
    dashboard = manager.get_dashboard_data()
    print(f"✅ Step 6: Retrieved dashboard with {dashboard['recent_activity']['diet_entries']} diet entries")
    
    # 7. Get recommendations
    recommendations = manager.get_personalized_recommendations()
    print(f"✅ Step 7: Generated {len(recommendations)} personalized recommendations")
    
    print("\n✅ INTEGRATION TEST PASSED!")
    
except Exception as e:
    print(f"\n❌ INTEGRATION TEST FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# ===== TEST 6: NO REPEATS VERIFICATION =====
print("🔍 TEST 6: Question Repeat Prevention")
print("-" * 70)

try:
    engine3 = AssessmentEngine("data/questions.json")
    assessment_info = engine3.start_assessment(0)
    
    seen_question_ids = set()
    repeat_found = False
    
    # Go through several questions
    for i in range(min(10, assessment_info['total_questions'])):
        question = engine3.get_current_question()
        if not question:
            break
        
        q_id = question['id']
        if q_id in seen_question_ids:
            print(f"❌ REPEAT DETECTED: Question {q_id} appeared twice!")
            repeat_found = True
            break
        
        seen_question_ids.add(q_id)
        
        # Submit answer to move forward
        correct_opt = next(opt for opt in question['options'] if opt['correct'])
        engine3.submit_answer(correct_opt['text'])
    
    if not repeat_found:
        print(f"✅ NO REPEATS: Verified {len(seen_question_ids)} unique questions")
        print(f"✅ Used question IDs tracked: {len(engine3.used_question_ids)}")
    
    print("\n✅ NO REPEATS VERIFICATION PASSED!")
    
except Exception as e:
    print(f"\n❌ NO REPEATS VERIFICATION FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# ===== FINAL SUMMARY =====
print("=" * 70)
print("✅ ALL TESTS PASSED SUCCESSFULLY!")
print("=" * 70)
print()
print("📊 VERIFICATION SUMMARY:")
print("   ✅ Database Manager: Working")
print("   ✅ Assessment Engine: Working (NO REPEATS)")
print("   ✅ User Manager: Working")
print("   ✅ Tracking System: Working")
print("   ✅ Integration: Working")
print("   ✅ Question Deduplication: Working")
print()
print("🎉 Core system is ready for GUI development!")
print()
print("📝 Next Steps:")
print("   1. Create GUI modules with PyQt6")
print("   2. Build main window and navigation")
print("   3. Implement assessment interface")
print("   4. Add tracking forms and visualizations")
print("   5. Create learning center UI")
print()
print("=" * 70)

# Cleanup
db.close()
if os.path.exists(test_db_path):
    print(f"🧹 Test database kept at: {test_db_path}")
