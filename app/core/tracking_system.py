"""
Scientific Hypertrophy Trainer - Tracking System
Analytics and statistics for diet, sleep, and training data
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
from statistics import mean, stdev


class TrackingSystem:
    """
    Handles tracking analytics:
    - Calculate statistics (averages, trends)
    - Correlation analysis
    - Generate recommendations
    - Progress visualization data
    """
    
    def __init__(self, db_manager):
        """
        Initialize tracking system
        Args:
            db_manager: DatabaseManager instance
        """
        self.db = db_manager
    
    # ===== DIET TRACKING =====
    
    def calculate_diet_stats(self, user_id: int, days: int = 30) -> Dict:
        """
        Calculate diet statistics for specified period
        Args:
            user_id: User ID
            days: Number of days to analyze
        Returns: Dictionary with diet statistics
        """
        entries = self.db.get_diet_entries(user_id, days)
        
        if not entries:
            return {
                'entries_count': 0,
                'avg_protein_g': 0,
                'avg_calories': 0,
                'avg_protein_per_kg': 0,
                'consistency_percentage': 0,
                'trend': 'no_data'
            }
        
        # Calculate averages
        avg_protein = round(mean([e['protein_g'] for e in entries]), 1)
        avg_calories = round(mean([e['calories'] for e in entries]))
        
        protein_per_kg_values = [e['protein_per_kg'] for e in entries if e['protein_per_kg']]
        avg_protein_per_kg = round(mean(protein_per_kg_values), 2) if protein_per_kg_values else 0
        
        # Calculate consistency (% of days tracked)
        consistency = round((len(entries) / days) * 100, 1)
        
        # Calculate trend (last 7 days vs previous 7 days)
        trend = self._calculate_trend(entries, 'protein_g', days)
        
        return {
            'entries_count': len(entries),
            'avg_protein_g': avg_protein,
            'avg_calories': avg_calories,
            'avg_protein_per_kg': avg_protein_per_kg,
            'consistency_percentage': consistency,
            'trend': trend,
            'last_entry': entries[0] if entries else None
        }
    
    def get_diet_recommendations(self, user_id: int, user_weight_kg: Optional[float] = None) -> List[Dict]:
        """
        Generate diet recommendations based on tracking data
        Args:
            user_id: User ID
            user_weight_kg: User's body weight in kg
        Returns: List of recommendation dictionaries
        """
        recommendations = []
        stats = self.calculate_diet_stats(user_id, 7)
        
        # Check protein intake
        if user_weight_kg and stats['avg_protein_per_kg'] > 0:
            if stats['avg_protein_per_kg'] < 1.6:
                recommendations.append({
                    'category': 'Protein',
                    'priority': 'high',
                    'message': f"Current: {stats['avg_protein_per_kg']}g/kg. Target: 1.6-2.2g/kg for optimal hypertrophy",
                    'suggestion': f"Increase protein by {round((1.6 * user_weight_kg) - stats['avg_protein_g'])}g daily"
                })
        
        # Check consistency
        if stats['consistency_percentage'] < 70:
            recommendations.append({
                'category': 'Consistency',
                'priority': 'medium',
                'message': f"Only tracking {stats['consistency_percentage']}% of days",
                'suggestion': "Set daily reminders to log nutrition for better insights"
            })
        
        # Check calorie sufficiency (rough estimate)
        if stats['avg_calories'] < 1800:
            recommendations.append({
                'category': 'Energy',
                'priority': 'medium',
                'message': "Calorie intake may be too low for optimal muscle growth",
                'suggestion': "Consider increasing calories if not in a cutting phase"
            })
        
        return recommendations
    
    # ===== SLEEP TRACKING =====
    
    def calculate_sleep_stats(self, user_id: int, days: int = 30) -> Dict:
        """
        Calculate sleep statistics for specified period
        Args:
            user_id: User ID
            days: Number of days to analyze
        Returns: Dictionary with sleep statistics
        """
        entries = self.db.get_sleep_entries(user_id, days)
        
        if not entries:
            return {
                'entries_count': 0,
                'avg_sleep_hours': 0,
                'avg_sleep_quality': 0,
                'consistency_percentage': 0,
                'sleep_debt_hours': 0
            }
        
        # Calculate averages
        avg_hours = round(mean([e['sleep_duration_hours'] for e in entries]), 1)
        avg_quality = round(mean([e['sleep_quality'] for e in entries]), 1)
        
        # Calculate consistency
        consistency = round((len(entries) / days) * 100, 1)
        
        # Calculate sleep debt (assuming 8 hours target)
        target_sleep = 8.0
        sleep_debt = max(0, (target_sleep - avg_hours) * len(entries))
        
        return {
            'entries_count': len(entries),
            'avg_sleep_hours': avg_hours,
            'avg_sleep_quality': avg_quality,
            'consistency_percentage': consistency,
            'sleep_debt_hours': round(sleep_debt, 1),
            'last_entry': entries[0] if entries else None
        }
    
    def get_sleep_recommendations(self, user_id: int) -> List[Dict]:
        """
        Generate sleep recommendations based on tracking data
        Args:
            user_id: User ID
        Returns: List of recommendation dictionaries
        """
        recommendations = []
        stats = self.calculate_sleep_stats(user_id, 7)
        
        # Check sleep duration
        if stats['avg_sleep_hours'] < 7.0:
            recommendations.append({
                'category': 'Sleep Duration',
                'priority': 'high',
                'message': f"Averaging {stats['avg_sleep_hours']}h. Target: 7-9h for optimal recovery",
                'suggestion': "Set a consistent bedtime to increase sleep duration"
            })
        
        # Check sleep quality
        if stats['avg_sleep_quality'] < 6:
            recommendations.append({
                'category': 'Sleep Quality',
                'priority': 'high',
                'message': f"Sleep quality score: {stats['avg_sleep_quality']}/10",
                'suggestion': "Optimize sleep environment (dark, cool, quiet)"
            })
        
        # Check sleep debt
        if stats['sleep_debt_hours'] > 7:
            recommendations.append({
                'category': 'Recovery',
                'priority': 'high',
                'message': f"Accumulated sleep debt: {stats['sleep_debt_hours']}h",
                'suggestion': "Consider prioritizing recovery with extra sleep this week"
            })
        
        return recommendations
    
    # ===== TRAINING TRACKING =====
    
    def calculate_training_stats(self, user_id: int, days: int = 30) -> Dict:
        """
        Calculate training statistics for specified period
        Args:
            user_id: User ID
            days: Number of days to analyze
        Returns: Dictionary with training statistics
        """
        entries = self.db.get_training_entries(user_id, days)
        
        if not entries:
            return {
                'entries_count': 0,
                'total_training_minutes': 0,
                'avg_session_duration': 0,
                'sessions_per_week': 0,
                'avg_session_quality': 0
            }
        
        # Calculate statistics
        total_minutes = sum([e['duration_minutes'] for e in entries])
        avg_duration = round(mean([e['duration_minutes'] for e in entries]))
        
        quality_scores = [e['session_quality'] for e in entries if e['session_quality']]
        avg_quality = round(mean(quality_scores), 1) if quality_scores else 0
        
        # Sessions per week
        weeks = days / 7
        sessions_per_week = round(len(entries) / weeks, 1)
        
        return {
            'entries_count': len(entries),
            'total_training_minutes': total_minutes,
            'avg_session_duration': avg_duration,
            'sessions_per_week': sessions_per_week,
            'avg_session_quality': avg_quality,
            'last_entry': entries[0] if entries else None
        }
    
    # ===== CORRELATION ANALYSIS =====
    
    def analyze_sleep_training_correlation(self, user_id: int, days: int = 30) -> Dict:
        """
        Analyze correlation between sleep quality and training performance
        Args:
            user_id: User ID
            days: Days to analyze
        Returns: Correlation analysis results
        """
        sleep_entries = self.db.get_sleep_entries(user_id, days)
        training_entries = self.db.get_training_entries(user_id, days)
        
        if len(sleep_entries) < 3 or len(training_entries) < 3:
            return {
                'correlation': 0,
                'insight': 'Not enough data for correlation analysis',
                'sample_size': 0
            }
        
        # Match sleep and training by date
        sleep_by_date = {e['entry_date']: e for e in sleep_entries}
        
        matched_pairs = []
        for training in training_entries:
            if training['entry_date'] in sleep_by_date:
                sleep = sleep_by_date[training['entry_date']]
                if sleep['sleep_quality'] and training['session_quality']:
                    matched_pairs.append({
                        'sleep_quality': sleep['sleep_quality'],
                        'training_quality': training['session_quality']
                    })
        
        if len(matched_pairs) < 3:
            return {
                'correlation': 0,
                'insight': 'Not enough matched sleep/training data',
                'sample_size': len(matched_pairs)
            }
        
        # Simple correlation calculation
        sleep_scores = [p['sleep_quality'] for p in matched_pairs]
        training_scores = [p['training_quality'] for p in matched_pairs]
        
        correlation = self._calculate_correlation(sleep_scores, training_scores)
        
        # Generate insight
        if correlation > 0.5:
            insight = "Strong positive correlation: Better sleep leads to better training"
        elif correlation > 0.3:
            insight = "Moderate positive correlation between sleep and training quality"
        elif correlation < -0.3:
            insight = "Negative correlation detected - investigate further"
        else:
            insight = "Weak or no clear correlation between sleep and training"
        
        return {
            'correlation': round(correlation, 2),
            'insight': insight,
            'sample_size': len(matched_pairs)
        }
    
    # ===== HELPER METHODS =====
    
    def _calculate_trend(self, entries: List[Dict], field: str, days: int) -> str:
        """Calculate trend direction for a metric"""
        if len(entries) < 7:
            return 'insufficient_data'
        
        # Sort by date
        sorted_entries = sorted(entries, key=lambda x: x['entry_date'])
        
        # Split into two halves
        mid_point = len(sorted_entries) // 2
        first_half = sorted_entries[:mid_point]
        second_half = sorted_entries[mid_point:]
        
        try:
            avg_first = mean([e[field] for e in first_half if e[field]])
            avg_second = mean([e[field] for e in second_half if e[field]])
            
            diff_percent = ((avg_second - avg_first) / avg_first) * 100
            
            if diff_percent > 5:
                return 'increasing'
            elif diff_percent < -5:
                return 'decreasing'
            else:
                return 'stable'
        except:
            return 'insufficient_data'
    
    def _calculate_correlation(self, x: List[float], y: List[float]) -> float:
        """Calculate Pearson correlation coefficient"""
        if len(x) != len(y) or len(x) < 2:
            return 0.0
        
        try:
            n = len(x)
            sum_x = sum(x)
            sum_y = sum(y)
            sum_xy = sum(xi * yi for xi, yi in zip(x, y))
            sum_x2 = sum(xi * xi for xi in x)
            sum_y2 = sum(yi * yi for yi in y)
            
            numerator = n * sum_xy - sum_x * sum_y
            denominator = ((n * sum_x2 - sum_x ** 2) * (n * sum_y2 - sum_y ** 2)) ** 0.5
            
            if denominator == 0:
                return 0.0
            
            return numerator / denominator
        except:
            return 0.0


# Test function
if __name__ == "__main__":
    print("🧪 Testing TrackingSystem...")
    
    from database.db_manager import DatabaseManager
    
    db = DatabaseManager("data/test_users.db")
    tracking = TrackingSystem(db)
    
    # Test with first user
    users = db.get_all_users()
    if users:
        user_id = users[0]['id']
        
        diet_stats = tracking.calculate_diet_stats(user_id, 7)
        print(f"\n📊 Diet stats: {diet_stats}")
        
        sleep_stats = tracking.calculate_sleep_stats(user_id, 7)
        print(f"\n😴 Sleep stats: {sleep_stats}")
        
        training_stats = tracking.calculate_training_stats(user_id, 7)
        print(f"\n💪 Training stats: {training_stats}")
    
    print("\n✅ TrackingSystem test complete!")
