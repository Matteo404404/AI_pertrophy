"""
Collaborative Filtering System

Finds similar users and generates predictions for new users based on
the performance of similar users (cold-start problem solution).

Author: Matteo - AI_PERTROPHY
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CollaborativeFilteringEngine:
    """
    Collaborative filtering for strength predictions.
    
    Uses cosine similarity to find similar users and generates
    predictions based on their historical performance.
    """
    
    def __init__(self, db_manager, embedding_calculator):
        """
        Initialize collaborative filtering engine.
        
        Args:
            db_manager: DatabaseManager instance
            embedding_calculator: UserEmbeddingCalculator instance
        """
        self.db = db_manager
        self.embedding_calc = embedding_calculator
    
    def find_similar_users(self, target_user_id: int, top_k: int = 10, 
                          min_completeness: float = 0.3) -> List[Dict]:
        """
        Find K most similar users to target user.
        
        Args:
            target_user_id: User ID to find similarities for
            top_k: Number of similar users to return
            min_completeness: Minimum data completeness score (0-1)
            
        Returns:
            List of dicts: [{'user_id': int, 'similarity': float, 'completeness': float}, ...]
        """
        # Get or compute target user embedding
        target_embedding = self.embedding_calc.get_or_compute_embedding(target_user_id)
        
        if target_embedding is None:
            logger.warning(f"Could not compute embedding for user {target_user_id}")
            return []
        
        # Get all other users
        all_users = self.db.get_all_users()
        other_user_ids = [u['id'] for u in all_users if u['id'] != target_user_id]
        
        if len(other_user_ids) == 0:
            logger.info("No other users in database")
            return []
        
        # Compute embeddings for all other users
        other_embeddings = []
        valid_user_ids = []
        completeness_scores = []
        
        for user_id in other_user_ids:
            try:
                embedding = self.embedding_calc.get_or_compute_embedding(user_id)
                if embedding is not None:
                    # Get completeness score
                    stored = self.db.get_user_embedding(user_id)
                    completeness = stored.get('data_completeness_score', 0) if stored else 0
                    
                    # Only include users with minimum data
                    if completeness >= min_completeness:
                        other_embeddings.append(embedding)
                        valid_user_ids.append(user_id)
                        completeness_scores.append(completeness)
                    
            except Exception as e:
                logger.warning(f"Could not compute embedding for user {user_id}: {e}")
                continue
        
        if len(other_embeddings) == 0:
            logger.info(f"No users with sufficient data (min_completeness={min_completeness})")
            return []
        
        # Compute cosine similarities
        target_matrix = normalize(target_embedding.reshape(1, -1))
        other_matrix = normalize(np.vstack(other_embeddings))
        
        similarities = cosine_similarity(target_matrix, other_matrix)[0]
        
        # Get top K
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        similar_users = [
            {
                'user_id': valid_user_ids[i],
                'similarity': float(similarities[i]),
                'completeness': float(completeness_scores[i])
            }
            for i in top_indices
        ]
        
        logger.info(f"Found {len(similar_users)} similar users for user {target_user_id}")
        return similar_users
    
    def predict_for_exercise(self, user_id: int, exercise_id: int, 
                            similar_users: Optional[List[Dict]] = None,
                            sessions_ahead: List[int] = [1, 2, 4, 10]) -> Dict:
        """
        Generate predictions for a specific exercise based on similar users.
        
        Args:
            user_id: Target user ID
            exercise_id: Exercise ID to predict
            similar_users: Pre-computed similar users (optional)
            sessions_ahead: List of horizons to predict for
            
        Returns:
            Dict with predictions for each horizon:
            {
                1: {'weight_kg': float, 'reps': int, 'rir': int, 'confidence': float},
                2: {...},
                ...
            }
        """
        # Find similar users if not provided
        if similar_users is None:
            similar_users = self.find_similar_users(user_id, top_k=10)
        
        if len(similar_users) == 0:
            # No similar users - return generic fallback
            return self._generic_fallback_prediction(exercise_id, sessions_ahead)
        
        # Collect performance data from similar users for this exercise
        weighted_data = []
        
        for similar_user in similar_users:
            similar_user_id = similar_user['user_id']
            similarity = similar_user['similarity']
            
            # Get their recent performance on this exercise
            performances = self.db.get_exercise_performance_history(
                similar_user_id, exercise_id, days=90
            )
            
            if len(performances) > 0:
                # Take most recent performance
                recent = performances[0]
                
                weighted_data.append({
                    'weight_kg': recent['best_set_weight_kg'],
                    'reps': recent['best_set_reps'],
                    'rir': recent['best_set_rir'] if recent['best_set_rir'] else 3,
                    'similarity': similarity
                })
        
        if len(weighted_data) == 0:
            # Similar users haven't done this exercise
            return self._generic_fallback_prediction(exercise_id, sessions_ahead)
        
        # Generate predictions for each horizon
        predictions = {}
        
        for horizon in sessions_ahead:
            pred = self._weighted_prediction(weighted_data, horizon)
            predictions[horizon] = pred
        
        return predictions
    
    def _weighted_prediction(self, weighted_data: List[Dict], sessions_ahead: int) -> Dict:
        """
        Generate weighted prediction from similar users' data.
        
        Args:
            weighted_data: List of dicts with performance + similarity
            sessions_ahead: Number of sessions in future to predict
            
        Returns:
            Dict: {'weight_kg': float, 'reps': int, 'rir': int, 'confidence': float}
        """
        # Calculate weighted averages
        total_weight = sum([d['similarity'] for d in weighted_data])
        
        weighted_kg = sum([d['weight_kg'] * d['similarity'] for d in weighted_data]) / total_weight
        weighted_reps = sum([d['reps'] * d['similarity'] for d in weighted_data]) / total_weight
        weighted_rir = sum([d['rir'] * d['similarity'] for d in weighted_data]) / total_weight
        
        # Apply progression factor (assume 1.5% strength gain per session)
        progression_factor = 1 + (sessions_ahead * 0.015)
        predicted_weight = weighted_kg * progression_factor
        
        # Adjust reps slightly (harder = fewer reps)
        if progression_factor > 1.05:
            predicted_reps = max(1, int(weighted_reps - (sessions_ahead * 0.3)))
        else:
            predicted_reps = int(round(weighted_reps))
        
        # RIR typically stays similar or increases slightly as weight increases
        predicted_rir = int(round(weighted_rir))
        
        # Calculate confidence (higher confidence for more data and closer predictions)
        base_confidence = 0.7
        data_bonus = min(len(weighted_data) / 10, 0.15)  # More similar users = higher confidence
        horizon_penalty = sessions_ahead * 0.03  # Further ahead = lower confidence
        
        confidence = base_confidence + data_bonus - horizon_penalty
        confidence = max(0.3, min(0.95, confidence))  # Clamp between 30-95%
        
        return {
            'weight_kg': round(predicted_weight, 1),
            'reps': predicted_reps,
            'rir': predicted_rir,
            'confidence': round(confidence, 2)
        }
    
    def _generic_fallback_prediction(self, exercise_id: int, 
                                    sessions_ahead: List[int]) -> Dict:
        """
        Generate generic fallback prediction when no similar users exist.
        
        Args:
            exercise_id: Exercise ID
            sessions_ahead: List of horizons
            
        Returns:
            Dict with generic predictions
        """
        # Get exercise info to determine typical starting weights
        exercise = self.db.get_exercise_by_id(exercise_id)
        
        if not exercise:
            base_weight = 50.0
        else:
            # Estimate based on exercise type
            muscle_group = exercise.get('muscle_group_primary', 'general').lower()
            is_compound = exercise.get('is_compound', False)
            
            # Rough starting weights for untrained individuals
            if is_compound:
                base_weights = {
                    'chest': 60,
                    'back': 70,
                    'legs': 80,
                    'quads': 80,
                    'hamstrings': 60,
                    'shoulders': 40,
                    'general': 50
                }
            else:
                base_weights = {
                    'chest': 30,
                    'back': 40,
                    'biceps': 25,
                    'triceps': 30,
                    'shoulders': 20,
                    'legs': 50,
                    'general': 30
                }
            
            base_weight = base_weights.get(muscle_group, 50)
        
        # Generate predictions with low confidence
        predictions = {}
        for horizon in sessions_ahead:
            progression = 1 + (horizon * 0.015)
            
            predictions[horizon] = {
                'weight_kg': round(base_weight * progression, 1),
                'reps': max(8, 10 - horizon),  # Fewer reps as weight increases
                'rir': 3,
                'confidence': max(0.25, 0.35 - (horizon * 0.02))  # Very low confidence
            }
        
        logger.info(f"Using generic fallback for exercise {exercise_id}")
        return predictions
    
    def predict_all_exercises(self, user_id: int, 
                             exercise_ids: Optional[List[int]] = None) -> Dict:
        """
        Generate predictions for all exercises user trains (or specified exercises).
        
        Args:
            user_id: User ID
            exercise_ids: Optional list of exercise IDs to predict (default: all trained)
            
        Returns:
            Dict: {exercise_id: {horizon: prediction_dict}}
        """
        # Find similar users once
        similar_users = self.find_similar_users(user_id, top_k=10)
        
        # Get exercises to predict
        if exercise_ids is None:
            # Get exercises user has trained
            performances = self.db.get_exercise_performance_history(user_id, days=90)
            if len(performances) > 0:
                exercise_ids = list(set([p['exercise_id'] for p in performances]))
            else:
                # New user - predict for common exercises
                all_exercises = self.db.get_all_exercises()
                exercise_ids = [e['id'] for e in all_exercises[:5]]  # Top 5 exercises
        
        # Generate predictions for each exercise
        all_predictions = {}
        
        for exercise_id in exercise_ids:
            try:
                predictions = self.predict_for_exercise(
                    user_id, exercise_id, similar_users=similar_users
                )
                all_predictions[exercise_id] = predictions
            except Exception as e:
                logger.error(f"Failed to predict for exercise {exercise_id}: {e}")
                continue
        
        return all_predictions
    
    def store_predictions(self, user_id: int, predictions: Dict, 
                         target_date: str = None) -> List[int]:
        """
        Store predictions in database for later validation.
        
        Args:
            user_id: User ID
            predictions: Dict from predict_all_exercises()
            target_date: Target date for predictions (default: today)
            
        Returns:
            List of prediction IDs
        """
        from datetime import datetime, timedelta
        
        if target_date is None:
            target_date = datetime.now().strftime('%Y-%m-%d')
        
        prediction_ids = []
        
        for exercise_id, horizons in predictions.items():
            for sessions_ahead, pred in horizons.items():
                try:
                    # Calculate target session date
                    target_session_date = (
                        datetime.strptime(target_date, '%Y-%m-%d') + 
                        timedelta(days=sessions_ahead * 3)  # Assume 3 days between sessions
                    ).strftime('%Y-%m-%d')
                    
                    # Store prediction
                    pred_id = self.db.store_ml_prediction(
                        user_id=user_id,
                        exercise_id=exercise_id,
                        target_date=target_session_date,
                        sessions_ahead=sessions_ahead,
                        predicted_weight=pred['weight_kg'],
                        predicted_reps=pred['reps'],
                        predicted_rir=pred['rir'],
                        confidence=pred['confidence'],
                        method='collaborative_filtering'
                    )
                    
                    prediction_ids.append(pred_id)
                    
                except Exception as e:
                    logger.error(f"Failed to store prediction: {e}")
                    continue
        
        logger.info(f"Stored {len(prediction_ids)} predictions for user {user_id}")
        return prediction_ids


def generate_predictions_for_new_user(db_manager, user_id: int, 
                                     exercise_ids: Optional[List[int]] = None) -> Dict:
    """
    Convenience function to generate predictions for a new user.
    
    Args:
        db_manager: DatabaseManager instance
        user_id: User ID
        exercise_ids: Optional list of exercises to predict
        
    Returns:
        Dict: {exercise_id: {horizon: prediction_dict}}
    """
    from ml.models.user_embedding import UserEmbeddingCalculator
    
    # Initialize components
    embedding_calc = UserEmbeddingCalculator(db_manager)
    cf_engine = CollaborativeFilteringEngine(db_manager, embedding_calc)
    
    # Generate predictions
    predictions = cf_engine.predict_all_exercises(user_id, exercise_ids)
    
    return predictions
