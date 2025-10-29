"""
User Embedding System for Collaborative Filtering

Creates 20-dimensional user vectors based on:
- Demographics (age, weight, height, body fat)
- Diet patterns (calories, protein, consistency)
- Sleep patterns (duration, quality)
- Training style (volume, frequency, exercise selection)
- Knowledge level (assessment scores)

Author: Matteo - AI_PERTROPHY
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UserEmbeddingCalculator:
    """
    Calculate user embeddings for collaborative filtering.
    
    20-dimensional embedding breakdown:
    - Demographics (4): age, weight, height, body_fat%
    - Diet patterns (5): avg_calories, protein_per_kg, consistency, calorie_balance, hydration
    - Sleep patterns (3): avg_duration, avg_quality, consistency
    - Training style (6): weekly_volume, frequency, avg_rir, variety, consistency, experience
    - Knowledge (2): tier_completed, avg_score
    """
    
    def __init__(self, db_manager):
        """
        Initialize embedding calculator.
        
        Args:
            db_manager: DatabaseManager instance
        """
        self.db = db_manager
        
        # Default values for missing data
        self.defaults = {
            'age': 25,
            'weight_kg': 75,
            'height_cm': 175,
            'body_fat_percentage': 15,
            'avg_calories': 2500,
            'protein_per_kg': 1.8,
            'diet_consistency': 50,
            'calorie_balance': 0,
            'hydration': 3.0,
            'sleep_duration': 7.5,
            'sleep_quality': 7,
            'sleep_consistency': 50,
            'weekly_volume': 12,
            'frequency': 4,
            'avg_rir': 3,
            'exercise_variety': 10,
            'training_consistency': 50,
            'experience_level': 0,
            'tier_completed': 0,
            'avg_assessment_score': 0
        }
    
    def compute_user_embedding(self, user_id: int) -> Tuple[np.ndarray, Dict, float]:
        """
        Compute complete user embedding.
        
        Args:
            user_id: User ID
            
        Returns:
            Tuple of (embedding_vector, component_breakdown, completeness_score)
        """
        # Get user data
        user = self.db.get_user_by_id(user_id)
        if not user:
            raise ValueError(f"User {user_id} not found")
        
        # Calculate each component
        demographics = self._compute_demographics(user)
        diet_patterns = self._compute_diet_patterns(user_id)
        sleep_patterns = self._compute_sleep_patterns(user_id)
        training_style = self._compute_training_style(user_id, user)
        knowledge = self._compute_knowledge(user_id)
        
        # Combine into single vector
        embedding = np.concatenate([
            demographics,
            diet_patterns,
            sleep_patterns,
            training_style,
            knowledge
        ])
        
        # Calculate data completeness score (0-1)
        completeness = self._calculate_completeness(user_id)
        
        # Component breakdown for storage/debugging
        component_data = {
            'demographics': demographics.tolist(),
            'diet_patterns': diet_patterns.tolist(),
            'sleep_patterns': sleep_patterns.tolist(),
            'training_style': training_style.tolist(),
            'knowledge': knowledge.tolist(),
            'completeness_score': completeness
        }
        
        return embedding, component_data, completeness
    
    def _compute_demographics(self, user: Dict) -> np.ndarray:
        """
        Compute demographic features (4 dimensions).
        
        Returns:
            [age, weight_kg, height_cm, body_fat%]
        """
        age = user.get('age', self.defaults['age'])
        weight = user.get('weight_kg', self.defaults['weight_kg'])
        height = user.get('height_cm', self.defaults['height_cm'])
        body_fat = user.get('body_fat_percentage', self.defaults['body_fat_percentage'])
        
        return np.array([age, weight, height, body_fat], dtype=np.float32)
    
    def _compute_diet_patterns(self, user_id: int) -> np.ndarray:
        """
        Compute diet pattern features (5 dimensions).
        
        Returns:
            [avg_calories, protein_per_kg, consistency%, calorie_balance, hydration_l]
        """
        diet_entries = self.db.get_diet_entries(user_id, days=30)
        
        if len(diet_entries) == 0:
            return np.array([
                self.defaults['avg_calories'],
                self.defaults['protein_per_kg'],
                self.defaults['diet_consistency'],
                self.defaults['calorie_balance'],
                self.defaults['hydration']
            ], dtype=np.float32)
        
        # Average calories
        calories = [e['total_calories'] for e in diet_entries if e['total_calories']]
        avg_calories = np.mean(calories) if calories else self.defaults['avg_calories']
        
        # Protein per kg
        user = self.db.get_user_by_id(user_id)
        weight = user.get('weight_kg', 75)
        proteins = [e['protein_g'] for e in diet_entries if e['protein_g']]
        avg_protein = np.mean(proteins) if proteins else (weight * self.defaults['protein_per_kg'])
        protein_per_kg = avg_protein / weight if weight > 0 else self.defaults['protein_per_kg']
        
        # Consistency (% of days logged in last 30)
        consistency = (len(diet_entries) / 30) * 100
        
        # Calorie balance (estimate surplus/deficit)
        tdee = self._estimate_tdee(user)
        calorie_balance = avg_calories - tdee
        
        # Hydration
        hydrations = [e['hydration_liters'] for e in diet_entries if e['hydration_liters']]
        avg_hydration = np.mean(hydrations) if hydrations else self.defaults['hydration']
        
        return np.array([
            avg_calories,
            protein_per_kg,
            consistency,
            calorie_balance,
            avg_hydration
        ], dtype=np.float32)
    
    def _compute_sleep_patterns(self, user_id: int) -> np.ndarray:
        """
        Compute sleep pattern features (3 dimensions).
        
        Returns:
            [avg_duration_hours, avg_quality_1-10, consistency%]
        """
        sleep_entries = self.db.get_sleep_entries(user_id, days=30)
        
        if len(sleep_entries) == 0:
            return np.array([
                self.defaults['sleep_duration'],
                self.defaults['sleep_quality'],
                self.defaults['sleep_consistency']
            ], dtype=np.float32)
        
        # Average duration
        durations = [e['sleep_duration_hours'] for e in sleep_entries if e['sleep_duration_hours']]
        avg_duration = np.mean(durations) if durations else self.defaults['sleep_duration']
        
        # Average quality
        qualities = [e['sleep_quality'] for e in sleep_entries if e['sleep_quality']]
        avg_quality = np.mean(qualities) if qualities else self.defaults['sleep_quality']
        
        # Consistency
        consistency = (len(sleep_entries) / 30) * 100
        
        return np.array([
            avg_duration,
            avg_quality,
            consistency
        ], dtype=np.float32)
    
    def _compute_training_style(self, user_id: int, user: Dict) -> np.ndarray:
        """
        Compute training style features (6 dimensions).
        
        Returns:
            [weekly_sessions, frequency, avg_rir, exercise_variety, consistency%, experience_level]
        """
        # Get workout data
        workouts = self.db.get_recent_workout_sessions(user_id, days=30)
        performances = self.db.get_exercise_performance_history(user_id, days=30)
        
        if len(performances) == 0:
            # No training data yet
            experience_map = {'beginner': 0, 'intermediate': 1, 'advanced': 2}
            experience = experience_map.get(user.get('experience_level', 'beginner'), 0)
            
            return np.array([
                self.defaults['weekly_volume'],
                self.defaults['frequency'],
                self.defaults['avg_rir'],
                self.defaults['exercise_variety'],
                self.defaults['training_consistency'],
                experience
            ], dtype=np.float32)
        
        # Weekly sessions (sessions per week)
        weekly_sessions = len(workouts) / 4.3  # 30 days ≈ 4.3 weeks
        
        # Frequency (workouts per week)
        frequency = len(set([w['session_date'] for w in workouts])) / 4.3
        
        # Average RIR
        rirs = [p['best_set_rir'] for p in performances if p['best_set_rir'] is not None]
        avg_rir = np.mean(rirs) if rirs else self.defaults['avg_rir']
        
        # Exercise variety (unique exercises trained)
        unique_exercises = len(set([p['exercise_id'] for p in performances]))
        
        # Training consistency (% of weeks with ≥3 workouts)
        consistency = (len(workouts) / 30) * 100
        
        # Experience level
        experience_map = {'beginner': 0, 'intermediate': 1, 'advanced': 2}
        experience = experience_map.get(user.get('experience_level', 'beginner'), 0)
        
        return np.array([
            weekly_sessions,
            frequency,
            avg_rir,
            unique_exercises,
            consistency,
            experience
        ], dtype=np.float32)
    
    def _compute_knowledge(self, user_id: int) -> np.ndarray:
        """
        Compute knowledge features (2 dimensions).
        
        Returns:
            [highest_tier_completed, avg_assessment_score%]
        """
        assessments = self.db.get_user_assessments(user_id)
        
        if len(assessments) == 0:
            return np.array([
                self.defaults['tier_completed'],
                self.defaults['avg_assessment_score']
            ], dtype=np.float32)
        
        # Highest tier passed
        passed = [a for a in assessments if a['passed']]
        tier_completed = max([a['tier_level'] for a in passed]) if passed else 0
        
        # Average assessment score
        scores = [a['percentage'] for a in assessments]
        avg_score = np.mean(scores) if scores else 0
        
        return np.array([
            tier_completed,
            avg_score
        ], dtype=np.float32)
    
    def _calculate_completeness(self, user_id: int) -> float:
        """
        Calculate data completeness score (0-1).
        
        Returns:
            Completeness score where 1.0 = all data available
        """
        # Check what data user has logged
        diet_entries = self.db.get_diet_entries(user_id, days=30)
        sleep_entries = self.db.get_sleep_entries(user_id, days=30)
        workouts = self.db.get_recent_workout_sessions(user_id, days=30)
        assessments = self.db.get_user_assessments(user_id)
        
        # Calculate completeness for each category
        diet_complete = min(len(diet_entries) / 20, 1.0)  # 20+ days = complete
        sleep_complete = min(len(sleep_entries) / 20, 1.0)
        workout_complete = min(len(workouts) / 10, 1.0)  # 10+ sessions = complete
        assessment_complete = min(len(assessments) / 3, 1.0)  # All 3 tiers = complete
        
        # Weighted average (training most important for strength predictions)
        completeness = (
            diet_complete * 0.20 +
            sleep_complete * 0.20 +
            workout_complete * 0.40 +
            assessment_complete * 0.20
        )
        
        return completeness
    
    def _estimate_tdee(self, user: Dict) -> float:
        """
        Estimate TDEE using Katch-McArdle formula.
        
        Args:
            user: User dictionary with weight, height, body_fat
            
        Returns:
            Estimated TDEE in calories
        """
        weight = user.get('weight_kg', 75)
        bf_pct = user.get('body_fat_percentage', 15)
        
        # Calculate lean body mass
        lean_mass = weight * (1 - bf_pct / 100)
        
        # Katch-McArdle BMR
        bmr = 370 + (21.6 * lean_mass)
        
        # Activity multiplier (moderate = 1.55)
        tdee = bmr * 1.55
        
        return tdee
    
    def update_user_embedding(self, user_id: int) -> bool:
        """
        Calculate and store user embedding in database.
        
        Args:
            user_id: User ID
            
        Returns:
            True if successful
        """
        try:
            embedding, component_data, completeness = self.compute_user_embedding(user_id)
            
            # Store in database
            self.db.store_user_embedding(user_id, embedding, component_data)
            
            logger.info(f"Updated embedding for user {user_id} (completeness: {completeness:.1%})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update embedding for user {user_id}: {e}")
            return False
    
    def get_or_compute_embedding(self, user_id: int) -> Optional[np.ndarray]:
        """
        Get user embedding from database, or compute if not exists.
        
        Args:
            user_id: User ID
            
        Returns:
            Embedding vector or None if failed
        """
        # Try to get from database first
        stored = self.db.get_user_embedding(user_id)
        
        if stored:
            import json
            embedding = np.array(json.loads(stored['embedding_json']), dtype=np.float32)
            return embedding
        
        # Compute and store if not found
        try:
            embedding, component_data, _ = self.compute_user_embedding(user_id)
            self.db.store_user_embedding(user_id, embedding, component_data)
            return embedding
        except Exception as e:
            logger.error(f"Failed to compute embedding for user {user_id}: {e}")
            return None
