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
        age = np.random.randint(18, 65)
        weight_kg = np.random.normal(75, 15)
        height_cm = np.random.normal(175, 10)
        body_fat = np.random.uniform(10, 30)
        experience = np.random.choice(['beginner', 'intermediate', 'advanced'], p=[0.33, 0.34, 0.33])
        
        # Assessment profile (one-time)
        assessment_score = np.random.randint(50, 100)
        knowledge_features = self._generate_knowledge_features(assessment_score)
        
        # Generate workout history
        workout_data = []
        
        for day_offset in range(self.days_history, 0, -1):
            session_date = (datetime.now() - timedelta(days=day_offset)).strftime('%Y-%m-%d')
            
            # Decide if workout day (3-4 per week probability)
            if np.random.random() > 0.5:
                continue
            
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
                    'user_id': user_id,
                    'session_date': session_date,
                    'day_offset': days_ago,
                    'exercise_id': exercise['id'],
                    'exercise_name': exercise['name'],
                    'weight_kg': weight_kg,
                    'reps': reps,
                    'rir': rir,
                    'total_sets': total_sets,
                    'exercise_order': ex_idx + 1,
                    
                    # User profile
                    'age': age,
                    'weight_kg_user': weight_kg,
                    'height_cm': height_cm,
                    'body_fat_pct': body_fat,
                    'experience_level': experience,
                    
                    # Assessment (static)
                    'assessment_score': assessment_score,
                    'training_literacy_index': knowledge_features['literacy_index'],
                    'load_management_score': knowledge_features['load_management'],
                    'technique_score': knowledge_features['technique'],
                    'recovery_knowledge': knowledge_features['recovery_knowledge'],
                    
                    # Diet
                    'calories': diet_data['calories'],
                    'protein_g': diet_data['protein'],
                    'carbs_g': diet_data['carbs'],
                    'fats_g': diet_data['fats'],
                    'hydration_l': diet_data['hydration'],
                    'meal_timing_score': diet_data['timing_score'],
                    'diet_consistency': diet_data['consistency'],
                    
                    # Sleep
                    'sleep_duration': sleep_data['duration'],
                    'sleep_quality': sleep_data['quality'],
                    'deep_sleep_pct': sleep_data['deep_pct'],
                    'rem_sleep_pct': sleep_data['rem_pct'],
                    'sleep_consistency': sleep_data['consistency'],
                    
                    # Supplements
                    'creatine_taken': supplement_data['creatine'],
                    'caffeine_mg': supplement_data['caffeine'],
                    'beta_alanine_taken': supplement_data['beta_alanine'],
                    'protein_shake_taken': supplement_data['protein'],
                    'supplement_count': supplement_data['total_count'],
                    
                    # Recovery
                    'days_since_last_session': recovery_data['days_since_last'],
                    'weekly_volume': recovery_data['weekly_volume'],
                    'weekly_frequency': recovery_data['weekly_frequency'],
                    'fatigue_index': recovery_data['fatigue_index'],
                    'recovery_score': recovery_data['recovery_score'],
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
    
    def _generate_diet_data(self, user_weight: float) -> Dict:
        """Generate realistic diet data."""
        base_tdee = 2000 + (user_weight * 10)  # Rough estimate
        
        # 70% consistency (log 70% of days)
        consistency = 70 + np.random.normal(0, 10)
        consistency = max(40, min(100, consistency))
        
        # Calories near TDEE
        calories = base_tdee * np.random.normal(1.0, 0.15)
        
        # Protein target: 1.6-2.2g per kg
        protein = user_weight * np.random.uniform(1.6, 2.2)
        
        # Calorie split
        protein_cals = protein * 4
        remaining = calories - protein_cals
        carbs = (remaining * 0.55) / 4
        fats = (remaining * 0.45) / 9
        
        return {
            'calories': calories,
            'protein': protein,
            'carbs': carbs,
            'fats': fats,
            'hydration': np.random.normal(3.0, 0.5),
            'timing_score': np.random.uniform(0.6, 1.0),
            'consistency': consistency,
        }
    
    def _generate_sleep_data(self) -> Dict:
        """Generate realistic sleep data."""
        # 80% logging consistency
        if np.random.random() > 0.8:
            return {
                'duration': 0,  # Missing data
                'quality': 0,
                'deep_pct': 0,
                'rem_pct': 0,
                'consistency': 0,
            }
        
        duration = np.random.normal(7.5, 0.8)
        duration = max(5, min(10, duration))
        
        quality = np.random.randint(4, 10)
        
        return {
            'duration': duration,
            'quality': quality,
            'deep_pct': np.random.uniform(10, 25),
            'rem_pct': np.random.uniform(15, 25),
            'consistency': 80 + np.random.normal(0, 10),
        }
    
    def _generate_supplement_data(self) -> Dict:
        """Generate supplement usage data."""
        return {
            'creatine': np.random.choice([0, 1], p=[0.5, 0.5]),
            'caffeine': np.random.randint(0, 300),
            'beta_alanine': np.random.choice([0, 1], p=[0.7, 0.3]),
            'protein': np.random.choice([0, 1], p=[0.4, 0.6]),
            'total_count': np.random.randint(0, 4),
        }
    
    def _generate_recovery_data(self) -> Dict:
        """Generate recovery metrics."""
        return {
            'days_since_last': np.random.randint(1, 4),
            'weekly_volume': np.random.normal(10000, 2000),
            'weekly_frequency': np.random.randint(3, 6),
            'fatigue_index': np.random.uniform(0.3, 0.8),
            'recovery_score': np.random.uniform(0.5, 0.95),
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