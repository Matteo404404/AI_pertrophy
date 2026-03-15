"""
Advanced Biomathematical Data Generator for Deep Learning
Uses the Banister Fitness-Fatigue model and Henneman Size Principle to simulate 
realistic human adaptation to mechanical tension.

Author: AI_PERTROPHY - Thesis Level Implementation
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import os
from typing import Dict, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedSyntheticDataGenerator:
    def __init__(self, num_users=2000, days_history=90, seed=42):
        self.num_users = num_users
        self.days_history = days_history
        np.random.seed(seed)
        
        # Exercise DB with scientific profiles
        self.exercises =[
            {'id': 1, 'name': 'Barbell Squat', 'compound': True, 'base_w': 80, 'cns_cost': 2.5},
            {'id': 2, 'name': 'Leg Extension', 'compound': False, 'base_w': 40, 'cns_cost': 0.8},
            {'id': 3, 'name': 'Incline DB Press', 'compound': True, 'base_w': 30, 'cns_cost': 1.5},
            {'id': 4, 'name': 'Cable Fly', 'compound': False, 'base_w': 15, 'cns_cost': 0.5},
            {'id': 5, 'name': 'Romanian Deadlift', 'compound': True, 'base_w': 90, 'cns_cost': 3.0},
        ]
        
    def generate_dataset(self) -> pd.DataFrame:
        all_data =[]
        
        for user_id in range(1, self.num_users + 1):
            if user_id % 500 == 0: 
                logger.info(f"Simulating physiology for user {user_id}/{self.num_users}")
            all_data.extend(self._simulate_user_physiology(user_id))
            
        return pd.DataFrame(all_data)
        
    def _simulate_user_physiology(self, user_id: int) -> List[Dict]:
        """Simulates a user's adaptation over time using differential equations"""
        
        # Genetics / Baseline
        age = np.random.randint(18, 55)
        weight_kg = np.random.normal(80, 10)
        
        # Genetic Time Constants (How fast they recover vs lose muscle)
        tau_fit = np.random.normal(25, 3) # ~25 days to lose adaptations
        tau_fat = np.random.normal(5, 1)  # ~5 days to clear CNS fatigue
        
        # User behavior archetypes
        is_bro_lifter = np.random.random() > 0.7 # High volume, high RIR (suboptimal)
        is_optimal = np.random.random() > 0.8    # Low volume, 0 RIR (optimal)
        
        user_history =[]
        
        # Track continuous state for each exercise
        state = {ex['id']: {'fitness': 0.0, 'fatigue': 0.0, 'base': ex['base_w']} for ex in self.exercises}
        global_cns_fatigue = 0.0
        
        for day_offset in range(self.days_history, 0, -1):
            session_date = (datetime.now() - timedelta(days=day_offset)).strftime('%Y-%m-%d')
            
            # --- 1. DECAY PREVIOUS DAY'S STATE ---
            global_cns_fatigue *= np.exp(-1 / tau_fat)
            for ex_id in state:
                state[ex_id]['fitness'] *= np.exp(-1 / tau_fit)
                state[ex_id]['fatigue'] *= np.exp(-1 / tau_fat)
            
            # Train 3x a week
            if np.random.random() > 0.4:
                continue
                
            # --- 2. GENERATE DAILY VARS ---
            sleep = np.random.normal(7, 1.5)
            protein = weight_kg * np.random.normal(1.8, 0.4)
            cals = np.random.normal(2500, 300)
            
            # Recovery multiplier (Sleep < 7 heavily penalizes recovery)
            recovery_mult = 1.0 if sleep >= 7 else np.exp((sleep - 7) * 0.2)
            
            # Pick exercises for the day (Fix for numpy choice with dicts)
            num_ex = np.random.randint(2, 5)
            indices = np.random.choice(len(self.exercises), num_ex, replace=False)
            session_exercises = [self.exercises[i] for i in indices]
            
            for ex in session_exercises:
                ex_id = ex['id']
                
                # --- 3. DETERMINE BEHAVIOR ---
                if is_optimal:
                    sets, rir, reps = np.random.randint(1, 3), 0, np.random.randint(5, 9)
                elif is_bro_lifter:
                    sets, rir, reps = np.random.randint(4, 7), np.random.randint(2, 5), np.random.randint(10, 15)
                else:
                    sets, rir, reps = np.random.randint(2, 5), np.random.randint(1, 4), np.random.randint(8, 12)
                
                # --- 4. CALCULATE PERFORMANCE (Banister Model) ---
                # Weight = Base + Fitness - Local Fatigue - Global CNS Fatigue
                readiness = state[ex_id]['fitness'] - state[ex_id]['fatigue'] - (global_cns_fatigue * 0.5)
                
                # Apply noise
                current_weight = state[ex_id]['base'] * (1 + (readiness * 0.05)) + np.random.normal(0, 1)
                current_weight = max(10.0, current_weight) # Can't lift negative
                
                # --- 5. CALCULATE NEW STIMULUS & FATIGUE ---
                # Henneman Size Principle: 0 RIR = 1.0. 3 RIR = 0.22.
                stimulus_per_set = np.exp(-0.5 * rir)
                
                # Diminishing returns on sets
                total_stimulus = sum([stimulus_per_set / (1 + 0.25 * i) for i in range(sets)])
                
                # Compound exercises generate drastically more CNS fatigue
                fatigue_generated = (sets * ex['cns_cost']) * (1.2 - (rir*0.1))
                
                # Update continuous state
                state[ex_id]['fitness'] += total_stimulus * recovery_mult
                state[ex_id]['fatigue'] += fatigue_generated
                global_cns_fatigue += fatigue_generated * 0.5
                
                # --- 6. LOG RECORD ---
                user_history.append({
                    'user_id': user_id,
                    'exercise_id': ex_id,
                    'exercise_name': ex['name'],
                    'date': session_date,
                    'day_offset': day_offset,
                    'weight_kg': current_weight,
                    'reps': reps,
                    'rir': rir,
                    'total_sets': sets,
                    'total_volume': current_weight * reps * sets,
                    'age': age,
                    'weight_kg_user': weight_kg,
                    'height_cm': 175,
                    'body_fat_pct': 15,
                    'assessment_score': 80 if is_optimal else 50,
                    'training_literacy_index': 0.8 if is_optimal else 0.5,
                    'load_management_score': 0.8 if is_optimal else 0.4,
                    'technique_score': 0.8,
                    'recovery_knowledge': 0.8 if sleep > 7 else 0.4,
                    'calories': cals,
                    'protein_g': protein,
                    'carbs_g': 250, 'fats_g': 80, 'fiber_g': 30, 'water_ml': 3000,
                    'sleep_hours': sleep,
                    'sleep_quality': min(10, max(1, int(sleep))),
                    'stress_level': 5,
                    'days_since_last_session': 2,
                    'creatine': 1, 'protein_powder': 1, 'pre_workout': 1, 'caffeine_mg': 200,
                    'soreness_level': min(10, max(1, int(state[ex_id]['fatigue']))),
                    'fatigue_level': min(10, max(1, int(global_cns_fatigue))),
                    'readiness_score': max(1, min(10, 10 - int(global_cns_fatigue))),
                    'hrv': max(30.0, min(100.0, 80.0 - (global_cns_fatigue * 2))),
                    'resting_heart_rate': 60 + global_cns_fatigue,
                    'session_rpe': 10 - rir,
                    'recovery_quality': min(10, max(1, int(sleep)))
                })
                
        return user_history

    def save_dataset(self, df: pd.DataFrame, filepath: str):
        """Save dataset to CSV."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath, index=False)
        logger.info(f"Saved dataset to {filepath}")
        return filepath

if __name__ == "__main__":
    generator = EnhancedSyntheticDataGenerator(num_users=10) # Test run
    df = generator.generate_dataset()
    print("✅ Mathematical Simulation Complete.")