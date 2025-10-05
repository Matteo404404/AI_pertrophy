"""
Scientific Hypertrophy Trainer - User Manager
High-level user management and statistics
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta


class UserManager:
    """
    Manages user business logic on top of database layer
    - User statistics calculation
    - Tier progression logic
    - User recommendations
    - Activity tracking
    """
    
    def __init__(self, db_manager):
        """
        Initialize user manager
        Args:
            db_manager: DatabaseManager instance
        """
        self.db = db_manager
        self.current_user = None
        
    def get_all_users(self) -> List[Dict]:
        """Get all users with enriched statistics"""
        users = self.db.get_all_users()
        
        # Enrich with statistics
        for user in users:
            user['stats'] = self.calculate_user_stats(user['id'])
            user['tier_progress'] = self.db.get_user_tier_progress(user['id'])
        
        return users
    
    def set_current_user(self, user_id: int) -> bool:
        """
        Set current active user
        Args:
            user_id: User ID to set as current
        Returns: True if successful
        """
        user = self.db.get_user_by_id(user_id)
        if user:
            self.current_user = user
            self.db.update_user_last_active(user_id)
            print(f"👤 Current user set: {user['username']} (Tier {self.get_current_tier() + 1})")
            return True
        return False
    
    def get_current_user(self) -> Optional[Dict]:
        """Get current active user"""
        return self.current_user
    
    def create_user(self, username: str, experience_level: str, primary_goal: str, weight_kg: Optional[float] = None) -> Dict:
        """
        Create new user profile
        Args:
            username: Unique username
            experience_level: beginner, intermediate, advanced
            primary_goal: muscle_growth, strength, endurance, etc.
            weight_kg: Optional body weight in kg
        Returns: Created user dictionary
        """
        # Validate inputs
        valid_experience = ['beginner', 'intermediate', 'advanced']
        if experience_level not in valid_experience:
            raise ValueError(f"Experience level must be one of: {valid_experience}")
        
        # Create in database
        user_id = self.db.create_user(username, experience_level, primary_goal, weight_kg)
        user = self.db.get_user_by_id(user_id)
        
        print(f"✅ Created user: {username} ({experience_level}, {primary_goal})")
        
        return user
    
    def calculate_user_stats(self, user_id: int) -> Dict:
        """
        Calculate comprehensive user statistics
        Args:
            user_id: User ID
        Returns: Dictionary with various statistics
        """
        assessments = self.db.get_user_assessments(user_id)
        
        # Basic assessment stats
        total_assessments = len(assessments)
        passed_assessments = sum(1 for a in assessments if a['passed'])
        
        # Calculate total questions and correct answers
        total_questions = 0
        total_correct = 0
        for assessment in assessments:
            total_questions += assessment['total_questions']
            total_correct += assessment['score']
        
        accuracy = round((total_correct / total_questions * 100) if total_questions > 0 else 0, 1)
        
        # Tier progression
        tier_progress = self.db.get_user_tier_progress(user_id)
        
        # Recent activity
        recent_assessments = assessments[:3] if assessments else []
        
        return {
            'total_assessments': total_assessments,
            'passed_assessments': passed_assessments,
            'total_questions_answered': total_questions,
            'total_correct_answers': total_correct,
            'accuracy_percentage': accuracy,
            'current_tier': tier_progress['current_tier'],
            'tiers_unlocked': sum([
                tier_progress['tier_2_unlocked'],
                tier_progress['tier_3_unlocked']
            ]),
            'recent_assessments': recent_assessments
        }
    
    def get_current_tier(self) -> int:
        """Get current user's tier level (0, 1, or 2)"""
        if not self.current_user:
            return 0
        
        progress = self.db.get_user_tier_progress(self.current_user['id'])
        return progress['current_tier']
    
    def can_access_tier(self, tier_level: int) -> bool:
        """
        Check if current user can access specified tier
        Args:
            tier_level: 0 (Tier 1), 1 (Tier 2), 2 (Tier 3)
        Returns: True if accessible
        """
        if not self.current_user:
            return False
        
        progress = self.db.get_user_tier_progress(self.current_user['id'])
        
        # Tier 1 always accessible
        if tier_level == 0:
            return True
        # Tier 2 requires Tier 1 passed
        elif tier_level == 1:
            return progress['tier_2_unlocked']
        # Tier 3 requires Tier 2 passed
        elif tier_level == 2:
            return progress['tier_3_unlocked']
        
        return False
    
    def get_tier_status(self) -> Dict:
        """
        Get detailed tier status for current user
        Returns: Dictionary with tier access and completion status
        """
        if not self.current_user:
            return {}
        
        progress = self.db.get_user_tier_progress(self.current_user['id'])
        
        return {
            'current_tier': progress['current_tier'] + 1,  # Display as 1, 2, 3
            'tiers': [
                {
                    'level': 1,
                    'name': 'Fundamentals',
                    'accessible': True,
                    'completed': progress['tier_1_passed'],
                    'unlocked': True
                },
                {
                    'level': 2,
                    'name': 'Intermediate',
                    'accessible': progress['tier_2_unlocked'],
                    'completed': progress['tier_2_passed'],
                    'unlocked': progress['tier_2_unlocked']
                },
                {
                    'level': 3,
                    'name': 'Advanced',
                    'accessible': progress['tier_3_unlocked'],
                    'completed': progress['tier_3_passed'],
                    'unlocked': progress['tier_3_unlocked']
                }
            ]
        }
    
    def save_assessment_result(self, assessment_results: Dict) -> int:
        """
        Save assessment results and update user progression
        Args:
            assessment_results: Results from AssessmentEngine
        Returns: Assessment ID
        """
        if not self.current_user:
            raise ValueError("No current user to save assessment for")
        
        assessment_id = self.db.save_assessment_result(
            user_id=self.current_user['id'],
            tier_level=assessment_results['tier_level'],
            score=assessment_results['score'],
            total_questions=assessment_results['total_questions'],
            percentage=assessment_results['percentage'],
            passed=assessment_results['passed'],
            answers=assessment_results['answers']
        )
        
        # Update tier progress is handled automatically by database
        
        print(f"💾 Saved assessment result: {'✅ PASSED' if assessment_results['passed'] else '❌ FAILED'}")
        
        return assessment_id
    
    def get_user_learning_data(self) -> Dict:
        """
        Get learning data for current user (wrong answers grouped by concept)
        Returns: Dictionary with grouped mistakes for learning system
        """
        if not self.current_user:
            return {'mistakes': []}
        
        assessments = self.db.get_user_assessments(self.current_user['id'])
        
        all_wrong_answers = []
        for assessment in assessments:
            answers = self.db.get_assessment_answers(assessment['id'])
            wrong_answers = [a for a in answers if not a['is_correct']]
            all_wrong_answers.extend(wrong_answers)
        
        # Group by concept (using simple categorization)
        grouped_mistakes = {}
        for answer in all_wrong_answers:
            # Categorize based on question text
            concept = self._categorize_mistake(answer['question_text'])
            if concept not in grouped_mistakes:
                grouped_mistakes[concept] = []
            grouped_mistakes[concept].append(answer)
        
        return {
            'total_mistakes': len(all_wrong_answers),
            'grouped_mistakes': grouped_mistakes,
            'priority_areas': sorted(
                grouped_mistakes.items(),
                key=lambda x: len(x[1]),
                reverse=True
            )[:3]  # Top 3 problem areas
        }
    
    def _categorize_mistake(self, question_text: str) -> str:
        """Categorize question into concept area"""
        text_lower = question_text.lower()
        
        if any(kw in text_lower for kw in ['volume', 'sets', 'mrv']):
            return 'Training Volume'
        elif any(kw in text_lower for kw in ['intensity', 'failure', 'rir']):
            return 'Training Intensity'
        elif any(kw in text_lower for kw in ['protein', 'nutrition', 'diet']):
            return 'Nutrition'
        elif any(kw in text_lower for kw in ['sleep', 'recovery']):
            return 'Recovery & Sleep'
        elif any(kw in text_lower for kw in ['frequency', 'times per week']):
            return 'Training Frequency'
        else:
            return 'General Hypertrophy'
    
    def get_personalized_recommendations(self) -> List[Dict]:
        """
        Generate personalized recommendations based on user data
        Returns: List of recommendation dictionaries
        """
        if not self.current_user:
            return []
        
        recommendations = []
        user_id = self.current_user['id']
        
        # Check assessment progress
        assessments = self.db.get_user_assessments(user_id)
        if len(assessments) == 0:
            recommendations.append({
                'category': 'Assessment',
                'priority': 'high',
                'title': 'Take Your First Assessment',
                'message': 'Start with Tier 1 Fundamentals to establish your knowledge baseline',
                'action': 'Start Tier 1 Assessment'
            })
        
        # Check if failed recent assessment
        if assessments and not assessments[0]['passed']:
            recommendations.append({
                'category': 'Learning',
                'priority': 'high',
                'title': 'Review Missed Concepts',
                'message': 'Study the concepts you missed in your last assessment',
                'action': 'Go to Learning Center'
            })
        
        # Check tracking data
        diet_entries = self.db.get_diet_entries(user_id, days=7)
        if len(diet_entries) == 0:
            recommendations.append({
                'category': 'Tracking',
                'priority': 'medium',
                'title': 'Start Tracking Nutrition',
                'message': 'Log your daily protein and calorie intake to optimize muscle growth',
                'action': 'Track Diet'
            })
        
        sleep_entries = self.db.get_sleep_entries(user_id, days=7)
        if len(sleep_entries) == 0:
            recommendations.append({
                'category': 'Tracking',
                'priority': 'medium',
                'title': 'Monitor Sleep Quality',
                'message': 'Track your sleep to ensure optimal recovery and muscle protein synthesis',
                'action': 'Track Sleep'
            })
        
        return recommendations
    
    def get_dashboard_data(self) -> Dict:
        """
        Get comprehensive dashboard data for current user
        Returns: Dictionary with all dashboard metrics
        """
        if not self.current_user:
            return {}
        
        user_id = self.current_user['id']
        stats = self.calculate_user_stats(user_id)
        tier_status = self.get_tier_status()
        recommendations = self.get_personalized_recommendations()
        
        # Get recent tracking data
        recent_diet = self.db.get_diet_entries(user_id, days=7)
        recent_sleep = self.db.get_sleep_entries(user_id, days=7)
        recent_training = self.db.get_training_entries(user_id, days=7)
        
        return {
            'user': self.current_user,
            'stats': stats,
            'tier_status': tier_status,
            'recommendations': recommendations,
            'recent_activity': {
                'diet_entries': len(recent_diet),
                'sleep_entries': len(recent_sleep),
                'training_entries': len(recent_training)
            }
        }


# Test function
if __name__ == "__main__":
    print("🧪 Testing UserManager...")
    
    from database.db_manager import DatabaseManager
    
    db = DatabaseManager("data/test_users.db")
    manager = UserManager(db)
    
    # Test getting users
    users = manager.get_all_users()
    print(f"\n✅ Found {len(users)} users")
    
    # Test setting current user
    if users:
        manager.set_current_user(users[0]['id'])
        stats = manager.calculate_user_stats(users[0]['id'])
        print(f"\n📊 User stats: {stats}")
        
        tier_status = manager.get_tier_status()
        print(f"\n🎯 Tier status: {tier_status}")
    
    print("\n✅ UserManager test complete!")
