"""
Simple Lifting Gains Predictor for Scientific Hypertrophy Trainer

What users ACTUALLY want:
1. "Will I get stronger this week?" (lifting gains forecast)
2. "Should I use more or less RIR?" (RIR recommendation)
3. "Should I rest longer between sets?" (rest time optimization)

That's it. No overcomplicated ML nonsense.

Author: Scientific Hypertrophy Trainer ML Team
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class LiftingForecast:
    """Simple lifting gains forecast."""
    # Main prediction
    strength_gain_next_week: str        # "likely", "possible", "unlikely"
    strength_confidence: float          # 0-100%
    
    # Simple recommendations
    recommended_rir: str                # "0-1 RIR", "2-3 RIR", "3-4 RIR"  
    recommended_rest_seconds: int       # 120, 150, 180, 240
    
    # Simple reasoning
    main_reason: str                    # Why this recommendation
    warning: Optional[str]              # Any warnings


class SimpleLiftingPredictor:
    """
    Ultra-simple lifting gains predictor.
    
    Based on just the essentials:
    - Recent performance trend
    - Recovery indicators (sleep, stress)
    - Volume progression
    """
    
    def __init__(self):
        """Initialize predictor."""
        pass
    
    def predict_lifting_gains(self, user_data: pd.DataFrame) -> LiftingForecast:
        """
        Predict lifting gains for next week.
        
        Args:
            user_data: Last 2-4 weeks of training data
            
        Returns:
            Simple lifting forecast with RIR/rest recommendations
        """
        
        # Get recent data (last 14 days)
        recent_data = user_data.tail(14).copy()
        
        if len(recent_data) < 7:
            return self._fallback_prediction()
        
        # Calculate key indicators
        volume_trend = self._get_volume_trend(recent_data)
        recovery_quality = self._get_recovery_quality(recent_data) 
        performance_trend = self._get_performance_trend(recent_data)
        fatigue_level = self._get_fatigue_level(recent_data)
        
        # Make simple prediction
        strength_likelihood = self._predict_strength_gains(
            volume_trend, recovery_quality, performance_trend, fatigue_level
        )
        
        # Get RIR recommendation
        rir_rec = self._recommend_rir(fatigue_level, recovery_quality, performance_trend)
        
        # Get rest time recommendation - FIXED: Consider recovery quality
        if fatigue_level == "high" or recovery_quality == "poor":
            rest_rec = 240  # 4 minutes for high fatigue OR poor recovery
        elif strength_likelihood == "likely":
            rest_rec = 180  # 3 minutes when gains likely
        else:
            rest_rec = 150  # 2.5 minutes standard
        
        # Generate reasoning
        reason = self._generate_reasoning(
            volume_trend, recovery_quality, performance_trend, fatigue_level
        )
        
        # Check for warnings
        warning = self._check_warnings(recent_data, fatigue_level, recovery_quality)
        
        return LiftingForecast(
            strength_gain_next_week=strength_likelihood,
            strength_confidence=self._calculate_confidence(recent_data),
            recommended_rir=rir_rec,
            recommended_rest_seconds=rest_rec,
            main_reason=reason,
            warning=warning
        )
    
    def _get_volume_trend(self, data: pd.DataFrame) -> str:
        """Get volume progression trend."""
        if 'total_sets' not in data.columns:
            return "stable"
        
        volumes = data['total_sets'].rolling(3).mean().dropna()
        
        if len(volumes) < 3:
            return "stable"
        
        # Simple trend calculation
        recent_avg = volumes.tail(3).mean()
        earlier_avg = volumes.head(3).mean()
        
        if earlier_avg == 0:
            return "stable"
        
        change_pct = (recent_avg - earlier_avg) / earlier_avg * 100
        
        if change_pct > 10:
            return "increasing"
        elif change_pct < -10:
            return "decreasing"
        else:
            return "stable"
    
    def _get_recovery_quality(self, data: pd.DataFrame) -> str:
        """Get overall recovery quality."""
        recovery_score = 0
        factors_checked = 0
        
        # Sleep quality
        if 'sleep_quality_1_10' in data.columns:
            avg_sleep = data['sleep_quality_1_10'].tail(7).mean()
            if avg_sleep >= 7:
                recovery_score += 1
            elif avg_sleep < 5:
                recovery_score -= 1
            factors_checked += 1
        
        # HRV
        if 'hrv_rmssd' in data.columns:
            avg_hrv = data['hrv_rmssd'].tail(7).mean()
            if avg_hrv >= 45:
                recovery_score += 1
            elif avg_hrv < 35:
                recovery_score -= 1
            factors_checked += 1
        
        # Stress levels
        if 'perceived_stress_1_10' in data.columns:
            avg_stress = data['perceived_stress_1_10'].tail(7).mean()
            if avg_stress <= 4:
                recovery_score += 1
            elif avg_stress >= 7:
                recovery_score -= 1
            factors_checked += 1
        
        if factors_checked == 0:
            return "unknown"
        
        # Convert to simple categories
        if recovery_score >= 1:
            return "good"
        elif recovery_score <= -1:
            return "poor"
        else:
            return "average"
    
    def _get_performance_trend(self, data: pd.DataFrame) -> str:
        """Get performance trend from RPE efficiency."""
        if 'average_rpe' not in data.columns or 'total_sets' not in data.columns:
            return "stable"
        
        # Calculate RPE per set (efficiency indicator)
        data_copy = data.copy()
        data_copy['rpe_per_set'] = data_copy['average_rpe'] / data_copy['total_sets'].replace(0, 1)
        rpe_efficiency = data_copy['rpe_per_set'].rolling(3).mean().dropna()
        
        if len(rpe_efficiency) < 4:
            return "stable"
        
        # Trend in RPE efficiency (lower = better performance)
        recent_eff = rpe_efficiency.tail(2).mean()
        earlier_eff = rpe_efficiency.head(2).mean()
        
        if earlier_eff == 0:
            return "stable"
        
        change_pct = (recent_eff - earlier_eff) / earlier_eff * 100
        
        if change_pct < -5:  # RPE per set decreasing = performance improving
            return "improving"
        elif change_pct > 5:   # RPE per set increasing = performance declining
            return "declining"
        else:
            return "stable"
    
    def _get_fatigue_level(self, data: pd.DataFrame) -> str:
        """Estimate current fatigue level."""
        fatigue_score = 0
        
        # Recent training volume
        if 'total_sets' in data.columns:
            recent_volume = data['total_sets'].tail(7).sum()
            typical_volume = data['total_sets'].mean() * 7
            
            if typical_volume > 0:
                if recent_volume > typical_volume * 1.2:
                    fatigue_score += 1
                elif recent_volume < typical_volume * 0.8:
                    fatigue_score -= 1
        
        # RPE trend
        if 'average_rpe' in data.columns:
            recent_rpe = data['average_rpe'].tail(5).mean()
            if recent_rpe >= 8:
                fatigue_score += 1
            elif recent_rpe <= 7:
                fatigue_score -= 1
        
        # Sleep debt
        if 'sleep_duration_hours' in data.columns:
            avg_sleep = data['sleep_duration_hours'].tail(7).mean()
            if avg_sleep < 7:
                fatigue_score += 1
        
        # Convert to categories
        if fatigue_score >= 2:
            return "high"
        elif fatigue_score <= -1:
            return "low"
        else:
            return "moderate"
    
    def _predict_strength_gains(self, volume_trend: str, recovery: str, 
                               performance: str, fatigue: str) -> str:
        """Predict likelihood of strength gains."""
        
        # Good conditions for strength gains
        positive_factors = 0
        negative_factors = 0
        
        # Volume trend
        if volume_trend == "increasing":
            positive_factors += 1
        elif volume_trend == "decreasing":
            negative_factors += 1
        
        # Recovery quality
        if recovery == "good":
            positive_factors += 2  # Recovery is very important
        elif recovery == "poor":
            negative_factors += 2
        
        # Performance trend
        if performance == "improving":
            positive_factors += 1
        elif performance == "declining":
            negative_factors += 1
        
        # Fatigue level
        if fatigue == "low":
            positive_factors += 1
        elif fatigue == "high":
            negative_factors += 2  # High fatigue kills gains
        
        # Make prediction
        net_score = positive_factors - negative_factors
        
        if net_score >= 2:
            return "likely"
        elif net_score <= -2:
            return "unlikely" 
        else:
            return "possible"
    
    def _recommend_rir(self, fatigue: str, recovery: str, performance: str) -> str:
        """Recommend RIR (Reps in Reserve)."""
        
        # High fatigue or poor recovery = more RIR (back off)
        if fatigue == "high" or recovery == "poor":
            return "3-4 RIR"
        
        # Low fatigue + good recovery + improving = push harder
        elif fatigue == "low" and recovery == "good" and performance == "improving":
            return "0-1 RIR"
        
        # Most people, most of the time
        else:
            return "2-3 RIR"
    
    def _generate_reasoning(self, volume_trend: str, recovery: str, 
                           performance: str, fatigue: str) -> str:
        """Generate simple reasoning for the recommendation."""
        
        reasons = []
        
        if recovery == "good":
            reasons.append("good recovery")
        elif recovery == "poor":
            reasons.append("poor recovery")
        
        if performance == "improving":
            reasons.append("performance improving")
        elif performance == "declining":
            reasons.append("performance declining")
        
        if fatigue == "high":
            reasons.append("high fatigue detected")
        elif fatigue == "low":
            reasons.append("low fatigue")
        
        if volume_trend == "increasing":
            reasons.append("volume trending up")
        elif volume_trend == "decreasing":
            reasons.append("volume declining")
        
        if not reasons:
            return "Based on recent training patterns"
        
        return "Based on: " + ", ".join(reasons)
    
    def _check_warnings(self, data: pd.DataFrame, fatigue: str, recovery: str) -> Optional[str]:
        """Check for any warnings."""
        
        warnings = []
        
        # Poor recovery warning
        if recovery == "poor":
            warnings.append("Poor recovery detected")
        
        # High fatigue warning  
        if fatigue == "high":
            warnings.append("High fatigue - consider backing off")
        
        # Sleep warning
        if 'sleep_duration_hours' in data.columns:
            avg_sleep = data['sleep_duration_hours'].tail(7).mean()
            if avg_sleep < 6:
                warnings.append("Insufficient sleep (<6h average)")
        
        # Stress warning
        if 'perceived_stress_1_10' in data.columns:
            avg_stress = data['perceived_stress_1_10'].tail(7).mean()
            if avg_stress >= 8:
                warnings.append("High stress levels")
        
        return "; ".join(warnings) if warnings else None
    
    def _calculate_confidence(self, data: pd.DataFrame) -> float:
        """Calculate prediction confidence."""
        
        confidence = 60.0  # Base confidence
        
        # More data = higher confidence
        if len(data) >= 14:
            confidence += 10
        elif len(data) < 7:
            confidence -= 20
        
        # Complete data = higher confidence
        key_cols = ['total_sets', 'average_rpe', 'sleep_quality_1_10']
        available_cols = [col for col in key_cols if col in data.columns]
        confidence += len(available_cols) * 5
        
        # Consistent patterns = higher confidence
        if 'total_sets' in data.columns:
            volume_mean = data['total_sets'].mean()
            if volume_mean > 0:
                volume_cv = data['total_sets'].std() / volume_mean
                if volume_cv < 0.3:  # Low variation = consistent
                    confidence += 10
        
        return min(90.0, max(30.0, confidence))
    
    def _fallback_prediction(self) -> LiftingForecast:
        """Fallback when insufficient data."""
        return LiftingForecast(
            strength_gain_next_week="possible",
            strength_confidence=50.0,
            recommended_rir="2-3 RIR",
            recommended_rest_seconds=150,
            main_reason="Insufficient data for detailed analysis",
            warning="Need more training history"
        )


def format_simple_recommendation(forecast: LiftingForecast) -> str:
    """Format forecast as simple user message."""
    
    # Strength prediction
    if forecast.strength_gain_next_week == "likely":
        strength_msg = "💪 STRENGTH GAINS LIKELY this week!"
    elif forecast.strength_gain_next_week == "possible":
        strength_msg = "🤔 STRENGTH GAINS POSSIBLE this week"
    else:
        strength_msg = "😐 STRENGTH GAINS UNLIKELY this week"
    
    # Simple recommendations
    output = []
    output.append(f"{strength_msg} ({forecast.strength_confidence:.0f}% confident)")
    output.append("")
    output.append("📋 SIMPLE RECOMMENDATIONS:")
    output.append(f"   RIR: {forecast.recommended_rir}")
    output.append(f"   Rest: {forecast.recommended_rest_seconds}s between sets")
    output.append("")
    output.append(f"🧠 Why: {forecast.main_reason}")
    
    if forecast.warning:
        output.append(f"⚠️  Warning: {forecast.warning}")
    
    return "\n".join(output)


# Example usage
def example_prediction():
    """Example of how to use the simple predictor."""
    
    # Create sample data
    dates = pd.date_range('2024-01-01', periods=21, freq='D')
    sample_data = pd.DataFrame({
        'date': dates,
        'total_sets': [12, 14, 13, 15, 11, 16, 0, 13, 15, 14, 16, 12, 17, 0, 
                      14, 16, 15, 18, 13, 17, 16],
        'average_rpe': [7.5, 7.8, 7.2, 8.0, 7.1, 8.2, 0, 7.3, 7.9, 7.4, 8.1, 
                       7.2, 8.3, 0, 7.4, 8.0, 7.6, 8.4, 7.3, 8.1, 7.8],
        'sleep_quality_1_10': [7, 8, 6, 7, 8, 5, 9, 7, 6, 8, 7, 8, 6, 9, 
                               7, 6, 8, 7, 8, 6, 7],
        'sleep_duration_hours': [7.5, 8.0, 6.5, 7.0, 8.2, 5.5, 9.0, 7.2, 6.8, 
                               8.1, 7.3, 8.0, 6.2, 9.2, 7.4, 6.9, 8.1, 7.1, 8.0, 6.8, 7.2],
        'hrv_rmssd': [45, 48, 42, 44, 49, 38, 52, 46, 43, 48, 44, 47, 40, 51, 
                     45, 42, 47, 43, 48, 41, 46],
        'perceived_stress_1_10': [4, 3, 6, 5, 3, 7, 2, 4, 5, 3, 5, 3, 6, 2, 
                                 4, 5, 3, 6, 4, 5, 4]
    })
    
    # Make prediction
    predictor = SimpleLiftingPredictor()
    forecast = predictor.predict_lifting_gains(sample_data)
    
    # Format for user
    recommendation = format_simple_recommendation(forecast)
    
    print("=" * 50)
    print("EXAMPLE LIFTING GAINS PREDICTION")
    print("=" * 50)
    print(recommendation)
    
    return forecast


if __name__ == "__main__":
    print("🚀 Simple Lifting Gains Predictor")
    print("Focused on what users actually want:")
    print("1. Will I get stronger?")
    print("2. What RIR should I use?") 
    print("3. How long should I rest?")
    print("")
    
    # Run example
    example_prediction()
    
    print("\n✅ Simple predictor ready!")
