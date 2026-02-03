"""
Enhanced Synthetic Data Generator for Deep Learning

Generates 10,000 realistic users with:
- Full 90-day workout histories (3-4 sessions/week)
- Complete diet tracking (daily macros, consistency, patterns)
- Sleep data (duration, quality, recovery trends)
- Supplement usage (creatine, caffeine, recovery stacks)
- Body measurements (weekly tracking)
- Assessment profiles (one-time knowledge baseline)
- Progressive overload patterns (realistic strength gains)

Author: AI_PERTROPHY - Task 3
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedSyntheticDataGenerator:
    """Generate realistic synthetic training data for LSTM model."""
    
    def __init__(self, num_users=10000, days_history=90, seed=42):
        """
        Initialize generator.
        
        Args:
            num_users: Number of synthetic users to generate
            days_history: Days of historical data per user
            seed: Random seed for reproducibility
        """
        self.num_users = num_users
        self.days_history = days_history
        self.seed = seed
        np.random.seed(seed)
        
        # Exercise database
        self.exercises = self._create_exercise_db()
        
        logger.info(f"Initialized generator for {num_users} users, {days_history} days history")
    
    def _create_exercise_db(self) -> List[Dict]:
        """Create realistic exercise database."""
        exercises = [
            # Chest
            {'id': 1, 'name': 'Barbell Bench Press', 'muscle': 'chest', 'compound': True, 'base_weight': 60},
            {'id': 2, 'name': 'Incline Dumbbell Press', 'muscle': 'chest', 'compound': True, 'base_weight': 30},
            {'id': 3, 'name': 'Cable Chest Fly', 'muscle': 'chest', 'compound': False, 'base_weight': 20},
            
            # Back
            {'id': 4, 'name': 'Barbell Row', 'muscle': 'back', 'compound': True, 'base_weight': 65},
            {'id': 5, 'name': 'Pull-ups', 'muscle': 'back', 'compound': True, 'base_weight': 0},
            {'id': 6, 'name': 'Lat Pulldown', 'muscle': 'back', 'compound': False, 'base_weight': 45},
            
            # Legs
            {'id': 7, 'name': 'Barbell Squat', 'muscle': 'legs', 'compound': True, 'base_weight': 80},
            {'id': 8, 'name': 'Romanian Deadlift', 'muscle': 'legs', 'compound': True, 'base_weight': 70},
            {'id': 9, 'name': 'Leg Press', 'muscle': 'legs', 'compound': True, 'base_weight': 120},
            
            # Shoulders
            {'id': 10, 'name': 'Overhead Press', 'muscle': 'shoulders', 'compound': True, 'base_weight': 40},
            {'id': 11, 'name': 'Lateral Raise', 'muscle': 'shoulders', 'compound': False, 'base_weight': 15},
            
            # Arms
            {'id': 12, 'name': 'Barbell Curl', 'muscle': 'biceps', 'compound': False, 'base_weight': 25},
            {'id': 13, 'name': 'Tricep Pushdown', 'muscle': 'triceps', 'compound': False, 'base_weight': 20},
        ]
        return exercises
    
    def generate_dataset(self) -> pd.DataFrame:
        """
        Generate complete synthetic dataset.
        
        Returns:
            DataFrame with all training data
        """
        all_data = []
        
        for user_id in range(1, self.num_users + 1):
            if user_id % 1000 == 0:
                logger.info(f"Generating user {user_id}/{self.num_users}")
            
            user_data = self._generate_user_data(user_id)
            all_data.extend(user_data)
        
        df = pd.DataFrame(all_data)
        logger.info(f"Generated {len(df)} records for {self.num_users} users")
        
        return df
    
    def _generate_user_data(self, user_id: int) -> List[Dict]:
        """Generate complete data for one user."""
        # User profile
        age = int(np.random.randint(18, 65))
        user_weight_kg = float(np.random.normal(75, 15))
        height_cm = float(np.random.normal(175, 10))
        body_fat_pct = float(np.random.uniform(10, 30))
        experience = np.random.choice(['beginner', 'intermediate', 'advanced'], p=[0.33, 0.34, 0.33])

        # Assessment profile (one-time)
        assessment_score = np.random.randint(50, 100)
        knowledge_features = self._generate_knowledge_features(assessment_score)
        last_workout_day_offset = None

        # Generate workout history
        workout_data = []
        
        for day_offset in range(self.days_history, 0, -1):
            session_date = (datetime.now() - timedelta(days=day_offset)).strftime('%Y-%m-%d')
            
            # Decide if workout day (3-4 per week probability)
            if np.random.random() > 0.5:
                continue
            if last_workout_day_offset is None:
                days_since_last_session = 2.0
            else:
                days_since_last_session = float(last_workout_day_offset - day_offset)

            last_workout_day_offset = day_offset

            # Select 3-5 exercises for this session
            num_exercises = np.random.randint(5, 12)
            session_exercises = np.random.choice(self.exercises, num_exercises, replace=False)
            
            for ex_idx, exercise in enumerate(session_exercises):
                # Progressive overload (slight improvement over 90 days)
                days_ago = self.days_history - day_offset
                progression_factor = 1.0 + (days_ago / self.days_history) * 0.15  # 15% gain over 90 days
                
                weight_kg = exercise['base_weight'] * progression_factor
                weight_kg *= np.random.normal(1.0, 0.1)  # Add noise
                
                reps = np.random.randint(6, 13)
                rir = np.random.randint(1, 4)
                total_sets = np.random.randint(2, 5)
                
                # Generate diet data
                diet_data = self._generate_diet_data(weight_kg)
                
                # Generate sleep data
                sleep_data = self._generate_sleep_data()
                
                # Generate supplement data
                supplement_data = self._generate_supplement_data()
                
                # Generate recovery metrics
                recovery_data = self._generate_recovery_data()
                
                
                record = {
                    # Workout (5)
                    'weight_kg': weight_kg,
                    'reps': reps,
                    'rir': rir,
                    'total_sets': total_sets,
                    'total_volume': weight_kg * reps * total_sets,
                    
                    # User (4)
                    'age': age,
                    'weight_kg_user': weight_kg,
                    'height_cm': height_cm,
                    'body_fat_pct': body_fat_pct,
                    
                    # Knowledge (5)
                    'assessment_score': assessment_score,
                    'training_literacy_index': knowledge_features['literacy_index'],
                    'load_management_score': knowledge_features['load_management'],
                    'technique_score': knowledge_features['technique'],
                    'recovery_knowledge': knowledge_features['recovery_knowledge'],
                    
                    # Diet (6)
                    'calories': diet_data['calories'],
                    'protein_g': diet_data['protein_g'],
                    'carbs_g': diet_data['carbs_g'],
                    'fats_g': diet_data['fats_g'],
                    'fiber_g': diet_data['fiber_g'],
                    'water_ml': diet_data['water_ml'],
                    
                    # Sleep/stress (4)
                    'sleep_hours': sleep_data['hours'],
                    'sleep_quality': sleep_data['quality'],
                    'stress_level': sleep_data['stress'],
                    'days_since_last_session': days_since_last_session,
                    
                    # Supplements (4)
                    'creatine': supplement_data['creatine'],
                    'protein_powder': supplement_data['protein_powder'],
                    'pre_workout': supplement_data['pre_workout'],
                    'caffeine_mg': supplement_data['caffeine_mg'],
                    
                    # Recovery (7)
                    'soreness_level': recovery_data['soreness_level'],
                    'fatigue_level': recovery_data['fatigue_level'],
                    'readiness_score': recovery_data['readiness_score'],
                    'hrv': recovery_data['hrv'],
                    'resting_heart_rate': recovery_data['resting_heart_rate'],
                    'session_rpe': recovery_data['session_rpe'],
                    'recovery_quality': recovery_data['recovery_quality'],
                    
                    # Keep your existing fields
                    'user_id': int(user_id),
                    'exercise_id': int(exercise['id']),   # <--- ADD THIS
                    'exercise_name': str(exercise['name']), # <--- ADD THIS
                    'date': str(session_date),
                    'day_offset': int(day_offset)           # <--- ADD THIS
                }

                
                workout_data.append(record)
        
        return workout_data
    
    def _generate_knowledge_features(self, score: int) -> Dict:
        """Generate knowledge-based features from assessment score."""
        literacy_index = score / 100.0
        
        return {
            'literacy_index': literacy_index,
            'load_management': min(1.0, literacy_index + np.random.normal(0, 0.1)),
            'technique': min(1.0, literacy_index + np.random.normal(0, 0.1)),
            'recovery_knowledge': min(1.0, literacy_index + np.random.normal(0, 0.1)),
        }
    
    def _generate_diet_data(self, weight_kg):
        """Generate diet data - ADD fiber_g and water_ml"""
        return {
        'calories': np.random.randint(1500, 3500),
        'protein_g': weight_kg * np.random.uniform(1.6, 2.2),
        'carbs_g': np.random.randint(200, 400),
        'fats_g': np.random.randint(50, 150),
        'fiber_g': np.random.randint(20, 45),        # ADD THIS
        'water_ml': np.random.randint(2000, 4000),}   # ADD THIS}

    def _generate_sleep_data(self):
        """Generate sleep data - ADD hours and stress"""
        return {
            'hours': np.random.uniform(5.0, 9.0),        # ADD THIS
            'quality': np.random.randint(1, 11),
            'stress': np.random.randint(1, 11),          # ADD THIS
        }

    def _generate_supplement_data(self):
        """Generate supplement data - ADD all 4 fields"""
        return {
            'creatine': float(np.random.choice([0, 1], p=[0.6, 0.4])),       # 0 or 1
            'protein_powder': float(np.random.choice([0, 1], p=[0.5, 0.5])), # 0 or 1
            'pre_workout': float(np.random.choice([0, 1], p=[0.7, 0.3])),    # 0 or 1
            'caffeine_mg': np.random.randint(0, 400),
        }

    def _generate_recovery_data(self):
        """Generate recovery data - ADD all 7 fields"""
        return {
            'soreness_level': np.random.randint(1, 11),
            'fatigue_level': np.random.randint(1, 11),
            'readiness_score': np.random.randint(1, 11),
            'hrv': np.random.randint(40, 100),
            'resting_heart_rate': np.random.randint(50, 80),
            'session_rpe': np.random.randint(1, 11),
            'recovery_quality': np.random.randint(1, 11),
        }

    
    def save_dataset(self, df: pd.DataFrame, filepath: str = 'ml/data/training_data.csv'):
        """Save dataset to CSV."""
        df.to_csv(filepath, index=False)
        logger.info(f"Saved dataset to {filepath}")
        return filepath


def generate_and_save(num_users: int = 10000, save_path: str = 'ml/data/training_data.csv'):
    """Convenience function to generate and save dataset."""
    logger.info(f"Starting synthetic data generation for {num_users} users...")
    
    generator = EnhancedSyntheticDataGenerator(num_users=num_users)
    df = generator.generate_dataset()
    
    # Display statistics
    logger.info(f"\nDataset Statistics:")
    logger.info(f"  Total records: {len(df)}")
    logger.info(f"  Unique users: {df['user_id'].nunique()}")
    logger.info(f"  Unique exercises: {df['exercise_id'].nunique()}")
    logger.info(f"  Date range: {df['session_date'].min()} to {df['session_date'].max()}")
    logger.info(f"  Weight range: {df['weight_kg'].min():.1f} - {df['weight_kg'].max():.1f} kg")
    logger.info(f"  Reps range: {df['reps'].min()} - {df['reps'].max()}")
    logger.info(f"  RIR range: {df['rir'].min()} - {df['rir'].max()}")
    
    generator.save_dataset(df, save_path)
    logger.info(f"\n✅ Dataset generation complete!")
    
    return df


if __name__ == "__main__":
    # Generate 10K users (takes ~2-3 minutes)
    df = generate_and_save(num_users=10000)
    
    # Show sample
    print("\n" + "="*80)
    print("SAMPLE DATA:")
    print("="*80)
    print(df.head(10))
    print("\nColumns:", df.columns.tolist())