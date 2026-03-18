"""
Biomathematical Data Generator for Strength Prediction

Implements realistic human adaptation using:
- Banister Fitness-Fatigue impulse-response model (Calvert 1976, Busso 1994)
- Henneman Size Principle for motor unit recruitment (Henneman 1965)
- Logarithmic dose-response for volume (Schoenfeld et al. 2017 meta-analysis)
- Inter-individual response variance (Hubal et al. 2005: 0-59% strength gains)
- Sleep-mediated recovery (Dattilo et al. 2011, Knowles et al. 2018)
- Protein dose-response (Morton et al. 2018: 1.6 g/kg inflection point)
- Realistic within-person day-to-day variability

Constant-feature problem FIXED: every feature now has realistic variance.
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

        self.exercises = [
            {'id': 1, 'name': 'Barbell Squat',      'compound': True,  'base_w': 80,  'cns_cost': 2.5, 'sfr': 0.75},
            {'id': 2, 'name': 'Leg Extension',       'compound': False, 'base_w': 40,  'cns_cost': 0.6, 'sfr': 1.2},
            {'id': 3, 'name': 'Incline DB Press',    'compound': True,  'base_w': 30,  'cns_cost': 1.3, 'sfr': 0.9},
            {'id': 4, 'name': 'Cable Fly',           'compound': False, 'base_w': 15,  'cns_cost': 0.4, 'sfr': 1.3},
            {'id': 5, 'name': 'Romanian Deadlift',   'compound': True,  'base_w': 90,  'cns_cost': 3.0, 'sfr': 0.6},
            {'id': 6, 'name': 'Seated Leg Curl',     'compound': False, 'base_w': 35,  'cns_cost': 0.5, 'sfr': 1.1},
            {'id': 7, 'name': 'Lat Pulldown',        'compound': True,  'base_w': 55,  'cns_cost': 1.0, 'sfr': 1.0},
            {'id': 8, 'name': 'Cable Lateral Raise',  'compound': False, 'base_w': 10,  'cns_cost': 0.3, 'sfr': 1.4},
        ]

    def generate_dataset(self) -> pd.DataFrame:
        all_data = []
        for user_id in range(1, self.num_users + 1):
            if user_id % 500 == 0:
                logger.info(f"Simulating user {user_id}/{self.num_users}")
            all_data.extend(self._simulate_user(user_id))
        return pd.DataFrame(all_data)

    def _simulate_user(self, user_id: int) -> List[Dict]:
        # --- DEMOGRAPHICS (all with realistic variance) ---
        age = np.random.randint(18, 55)
        weight_kg = np.clip(np.random.normal(80, 12), 50, 130)
        height_cm = np.clip(np.random.normal(175, 8), 155, 200)
        body_fat_pct = np.clip(np.random.normal(18, 6), 6, 40)

        # Genetic time constants (Busso 1994: tau_1 = 11-60d, tau_2 = 2-8d)
        tau_fit = np.clip(np.random.normal(30, 8), 12, 60)
        tau_fat = np.clip(np.random.normal(5, 1.5), 2, 10)

        # Genetic gain responsiveness (Hubal et al. 2005: massive variance)
        gain_factor = np.clip(np.random.normal(1.0, 0.35), 0.2, 2.0)

        # Training archetype
        archetype = np.random.choice(
            ['optimal', 'moderate', 'bro_volume', 'undertrained'],
            p=[0.15, 0.40, 0.30, 0.15]
        )

        # Knowledge/literacy score (correlated with archetype)
        if archetype == 'optimal':
            literacy = np.clip(np.random.normal(0.85, 0.08), 0.6, 1.0)
        elif archetype == 'moderate':
            literacy = np.clip(np.random.normal(0.60, 0.12), 0.3, 0.9)
        elif archetype == 'bro_volume':
            literacy = np.clip(np.random.normal(0.40, 0.10), 0.15, 0.65)
        else:
            literacy = np.clip(np.random.normal(0.30, 0.10), 0.1, 0.55)

        assessment_score = literacy * 100

        # Supplement use probabilities (correlated with knowledge)
        uses_creatine = int(np.random.random() < (0.3 + literacy * 0.4))
        uses_protein_powder = int(np.random.random() < (0.2 + literacy * 0.5))
        uses_pre_workout = int(np.random.random() < 0.35)
        base_caffeine = np.random.choice([0, 100, 200, 300, 400])

        # Lifestyle
        base_stress = np.clip(np.random.normal(5, 2), 1, 10)
        base_sleep = np.clip(np.random.normal(7.0, 1.0), 4.5, 9.5)

        # Nutrition targets (Morton et al. 2018: 1.6 g/kg protein inflection)
        if literacy > 0.7:
            protein_target = weight_kg * np.random.normal(2.0, 0.2)
        elif literacy > 0.4:
            protein_target = weight_kg * np.random.normal(1.6, 0.3)
        else:
            protein_target = weight_kg * np.random.normal(1.2, 0.3)

        tdee = 370 + 21.6 * (weight_kg * (1 - body_fat_pct / 100))
        tdee *= np.random.choice([1.4, 1.55, 1.7, 1.85], p=[0.15, 0.35, 0.35, 0.15])
        cal_target = tdee + np.random.normal(200, 100)

        history = []
        state = {ex['id']: {'fitness': 0.0, 'fatigue': 0.0, 'base': ex['base_w']}
                 for ex in self.exercises}
        global_cns = 0.0
        days_since_session = 2

        for day_offset in range(self.days_history, 0, -1):
            session_date = (datetime.now() - timedelta(days=day_offset)).strftime('%Y-%m-%d')

            # --- DECAY ---
            global_cns *= np.exp(-1 / tau_fat)
            global_cns = max(0, global_cns)
            for eid in state:
                state[eid]['fitness'] *= np.exp(-1 / tau_fit)
                state[eid]['fatigue'] *= np.exp(-1 / tau_fat)
            days_since_session += 1

            # Training frequency: ~3-5x/week depending on archetype
            if archetype == 'optimal':
                trains_today = np.random.random() < 0.50
            elif archetype == 'moderate':
                trains_today = np.random.random() < 0.55
            elif archetype == 'bro_volume':
                trains_today = np.random.random() < 0.65
            else:
                trains_today = np.random.random() < 0.35

            if not trains_today:
                continue

            # --- DAILY BIOMETRICS (all with day-to-day variance) ---
            sleep_hrs = np.clip(np.random.normal(base_sleep, 1.2), 3, 11)
            sleep_qual = np.clip(int(np.random.normal(sleep_hrs, 1.0)), 1, 10)
            stress = np.clip(np.random.normal(base_stress, 1.5), 1, 10)
            protein_g = np.clip(np.random.normal(protein_target, 20), 50, 350)
            calories = np.clip(np.random.normal(cal_target, 250), 1200, 5000)
            carbs_g = np.clip(np.random.normal(calories * 0.45 / 4, 30), 100, 600)
            fats_g = np.clip(np.random.normal(calories * 0.30 / 9, 10), 30, 200)
            fiber_g = np.clip(np.random.normal(28, 8), 5, 60)
            water_ml = np.clip(np.random.normal(2800, 600), 1000, 5000)

            caffeine_mg = base_caffeine + np.random.choice([-50, 0, 0, 50, 100])
            caffeine_mg = max(0, caffeine_mg)

            # HRV and RHR respond to accumulated fatigue (Plews et al. 2013)
            hrv_base = np.clip(np.random.normal(65, 12), 30, 100)
            hrv = np.clip(hrv_base - global_cns * 3 + (sleep_hrs - 7) * 2, 20, 100)
            rhr = np.clip(np.random.normal(62, 6) + global_cns * 1.5, 45, 95)

            # Recovery quality is a composite
            recovery_mult = 1.0
            if sleep_hrs < 6:
                recovery_mult *= np.exp((sleep_hrs - 6) * 0.25)
            if protein_g / weight_kg < 1.2:
                recovery_mult *= 0.85
            if stress > 7:
                recovery_mult *= 0.90

            recovery_quality = np.clip(int(sleep_qual * recovery_mult), 1, 10)
            readiness = np.clip(int(10 - global_cns * 1.5 - (10 - sleep_qual) * 0.3), 1, 10)

            # Pick exercises
            num_ex = np.random.randint(2, 5)
            ex_indices = np.random.choice(len(self.exercises), num_ex, replace=False)
            session_exercises = [self.exercises[i] for i in ex_indices]

            session_cns_load = 0.0

            for order, ex in enumerate(session_exercises, 1):
                eid = ex['id']

                # --- TRAINING BEHAVIOR ---
                if archetype == 'optimal':
                    sets = np.random.randint(2, 4)
                    rir = np.random.choice([0, 0, 1, 1])
                    reps = np.random.randint(5, 10)
                elif archetype == 'moderate':
                    sets = np.random.randint(2, 5)
                    rir = np.random.choice([1, 1, 2, 2, 3])
                    reps = np.random.randint(6, 12)
                elif archetype == 'bro_volume':
                    sets = np.random.randint(4, 8)
                    rir = np.random.choice([2, 3, 3, 4, 4])
                    reps = np.random.randint(10, 16)
                else:
                    sets = np.random.randint(1, 3)
                    rir = np.random.choice([3, 4, 4, 5])
                    reps = np.random.randint(8, 15)

                # --- PERFORMANCE (Banister model) ---
                net_readiness = (state[eid]['fitness'] * gain_factor
                                 - state[eid]['fatigue']
                                 - global_cns * 0.4)
                current_weight = state[eid]['base'] * (1 + net_readiness * 0.04)
                current_weight += np.random.normal(0, 1.5)
                current_weight = max(5.0, current_weight)

                total_volume = current_weight * reps * sets

                # --- STIMULUS & FATIGUE UPDATE ---
                # Henneman: stimulus per set as function of RIR
                stimulus_per_set = np.exp(-0.5 * rir)
                # Diminishing returns (Krieger 2010, Schoenfeld 2017)
                total_stimulus = sum(stimulus_per_set / (1 + 0.25 * s) for s in range(sets))

                fatigue_gen = sets * ex['cns_cost'] * (1.2 - rir * 0.08)
                fatigue_gen *= (1 / recovery_mult)

                state[eid]['fitness'] += total_stimulus * recovery_mult
                state[eid]['fatigue'] += fatigue_gen * 0.7
                session_cns_load += fatigue_gen * 0.3

                soreness = np.clip(int(state[eid]['fatigue'] * 1.2 + np.random.normal(0, 1)), 1, 10)
                fatigue_level = np.clip(int(global_cns * 1.5 + session_cns_load * 0.5 + np.random.normal(0, 1)), 1, 10)
                session_rpe = np.clip(10 - rir + np.random.choice([-1, 0, 0, 0, 1]), 1, 10)

                history.append({
                    'user_id': user_id,
                    'exercise_id': eid,
                    'exercise_name': ex['name'],
                    'date': session_date,
                    'day_offset': day_offset,
                    'weight_kg': round(current_weight, 2),
                    'reps': reps,
                    'rir': rir,
                    'total_sets': sets,
                    'total_volume': round(total_volume, 2),
                    'age': age,
                    'weight_kg_user': round(weight_kg, 1),
                    'height_cm': round(height_cm, 1),
                    'body_fat_pct': round(body_fat_pct, 1),
                    'assessment_score': round(assessment_score, 1),
                    'training_literacy_index': round(literacy, 3),
                    'load_management_score': round(literacy * np.clip(np.random.normal(1.0, 0.1), 0.7, 1.3), 3),
                    'technique_score': round(np.clip(np.random.normal(0.7 + literacy * 0.2, 0.08), 0.3, 1.0), 3),
                    'recovery_knowledge': round(literacy * (0.9 if sleep_hrs > 7 else 0.6), 3),
                    'calories': round(calories),
                    'protein_g': round(protein_g, 1),
                    'carbs_g': round(carbs_g, 1),
                    'fats_g': round(fats_g, 1),
                    'fiber_g': round(fiber_g, 1),
                    'water_ml': round(water_ml),
                    'sleep_hours': round(sleep_hrs, 2),
                    'sleep_quality': sleep_qual,
                    'stress_level': round(stress, 1),
                    'days_since_last_session': days_since_session,
                    'creatine': uses_creatine,
                    'protein_powder': uses_protein_powder,
                    'pre_workout': uses_pre_workout,
                    'caffeine_mg': caffeine_mg,
                    'soreness_level': soreness,
                    'fatigue_level': fatigue_level,
                    'readiness_score': readiness,
                    'hrv': round(hrv, 1),
                    'resting_heart_rate': round(rhr, 1),
                    'session_rpe': session_rpe,
                    'recovery_quality': recovery_quality,
                })

            global_cns += session_cns_load
            global_cns = min(global_cns, 15)
            days_since_session = 0

        return history

    def save_dataset(self, df: pd.DataFrame, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath, index=False)
        logger.info(f"Saved {len(df)} records to {filepath}")
        return filepath


if __name__ == "__main__":
    gen = EnhancedSyntheticDataGenerator(num_users=10)
    df = gen.generate_dataset()
    print(f"Generated {len(df)} records")
    print(f"Feature std check (should all be > 0.01):")
    for col in df.select_dtypes(include=[np.number]).columns:
        print(f"  {col}: std={df[col].std():.4f}")
