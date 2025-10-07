"""
TDEE Calculator Module for Scientific Hypertrophy Trainer

This module implements evidence-based Total Daily Energy Expenditure (TDEE) calculations
using multiple validated equations and adaptive metabolic adjustments based on longitudinal
weight change data.

References:
- Mifflin et al. 1990: Most accurate BMR equation for general population (82% accuracy within 10%)
- Frankenfield et al. 2005: Systematic review confirming Mifflin-St Jeor superiority
- Katch-McArdle: Preferred when accurate body fat % available
- Cunningham: Best for lean athletic populations (BF% < 15%M/25%F)
- Adaptive Thermogenesis: 15-20% metabolic reduction in calorie restriction (Leibel et al. 1995)
- Activity multipliers: Validated ranges from DLW studies (Westerterp 2013)
- CALERIE studies: Redman et al. 2018 - Metabolic adaptation 5-13% during CR

Scientific Evidence:
- Mifflin-St Jeor: 82% accuracy within 10% of measured RMR in non-obese adults
- 70% accuracy in obese individuals (vs 60-65% for other equations)
- Metabolic adaptation: 240-430 kcal/day reduction beyond body composition changes
- TDEE components: BMR (60-70%), NEAT (15-30%), EAT (15-30%), TEF (8-15%)

Author: Scientific Hypertrophy Trainer ML Team
Version: 1.0 - Evidence-based implementation
"""

from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np
from datetime import datetime, timedelta
import logging

# Import from ML configuration
from ..base.config import ACTIVITY_MULTIPLIERS, NUTRITION_RANGES, NOISE_LEVELS
from ..utils.logger import ModelLogger


class ActivityLevel(Enum):
    """Activity level classifications with evidence-based multipliers."""
    SEDENTARY = 1.2          # Desk job, no exercise
    LIGHTLY_ACTIVE = 1.375   # Light exercise 1-3 days/week  
    MODERATELY_ACTIVE = 1.55 # Moderate exercise 3-5 days/week
    VERY_ACTIVE = 1.725      # Heavy exercise 6-7 days/week
    EXTREMELY_ACTIVE = 1.9   # Physical job + exercise, 2x/day training


class EquationType(Enum):
    """BMR calculation methods with specific use cases."""
    MIFFLIN_ST_JEOR = "mifflin"    # Most accurate for general population (82% within 10%)
    KATCH_MCARDLE = "katch"        # Best when BF% known accurately (<5% error)
    CUNNINGHAM = "cunningham"      # Optimal for lean athletes (BF% < 15%M/25%F)


@dataclass
class TDEEResult:
    """TDEE calculation result with breakdown components."""
    bmr: float                    # Basal Metabolic Rate (kcal/day)
    neat: float                   # Non-Exercise Activity Thermogenesis
    eat: float                    # Exercise Activity Thermogenesis  
    tef: float                    # Thermic Effect of Food (8-15% of intake)
    tdee: float                   # Total Daily Energy Expenditure
    activity_multiplier: float    # Applied activity factor
    equation_used: str           # BMR equation identifier
    adaptive_factor: float       # Metabolic adaptation adjustment (0.7-1.3)
    confidence_score: float      # Calculation reliability (0-1)
    metabolic_adaptation_kcal: float  # Estimated metabolic adaptation (kcal/day)


@dataclass
class MetabolicData:
    """Historical data for adaptive thermogenesis calculations."""
    date: datetime
    weight_kg: float
    calories_intake: float
    tdee_estimated: float
    weight_change_rate: float    # kg/week


class TDEECalculator:
    """
    Advanced TDEE calculator with adaptive metabolic adjustments.
    
    Based on CALERIE study findings (Redman et al. 2018):
    - Metabolic adaptation: 5-13% reduction beyond body composition
    - Occurs within 3 months of calorie restriction
    - Persists during weight maintenance phase
    - Larger in free-living vs laboratory conditions
    
    Features:
    - Multiple validated BMR equations with accuracy metrics
    - Activity-specific multipliers from DLW validation studies
    - Adaptive thermogenesis modeling (Leibel et al. 1995)
    - Longitudinal weight trend analysis for metabolic adaptation
    - Body composition considerations for accurate predictions
    """
    
    # Metabolic adaptation constants from CALERIE studies
    ADAPTATION_RATES = {
        'severe_deficit': -0.15,   # >25% restriction: 15% reduction (Redman et al.)
        'moderate_deficit': -0.08, # 15-25% restriction: 8% reduction  
        'mild_deficit': -0.03,     # 5-15% restriction: 3% reduction
        'maintenance': 0.0,        # Energy balance: no adaptation
        'mild_surplus': 0.03,      # 5-15% surplus: 3% increase
        'moderate_surplus': 0.08,  # 15-25% surplus: 8% increase (Johannsen et al.)
        'severe_surplus': 0.12     # >25% surplus: 12% increase
    }
    
    # TEF percentages by macronutrient (Westerterp 2004)
    TEF_MACROS = {
        'protein': 0.25,       # 20-30% thermic effect
        'carbs': 0.08,         # 5-10% thermic effect  
        'fats': 0.03          # 0-5% thermic effect
    }
    
    # BMR equation accuracy rates (Frankenfield et al. 2005)
    EQUATION_ACCURACY = {
        'mifflin': {'non_obese': 0.82, 'obese': 0.70, 'athletes': 0.75},
        'katch': {'non_obese': 0.78, 'obese': 0.72, 'athletes': 0.85},
        'cunningham': {'non_obese': 0.75, 'obese': 0.65, 'athletes': 0.88}
    }
    
    def __init__(self):
        """Initialize TDEE calculator with physiological constants."""
        self.metabolic_history: List[MetabolicData] = []
        self.logger = logging.getLogger(__name__)
        
        # Adaptive thermogenesis tracking
        self.baseline_tdee = None
        self.adaptation_onset_date = None
        self.current_adaptation_rate = 0.0
        

    def calculate_bmr(self, 
                     age: int, 
                     weight_kg: float, 
                     height_cm: float, 
                     gender: str, 
                     body_fat_percentage: Optional[float] = None,
                     equation: EquationType = EquationType.MIFFLIN_ST_JEOR) -> Dict:
        """
        Calculate Basal Metabolic Rate using validated equations.
        
        Args:
            age: Age in years
            weight_kg: Body weight in kilograms
            height_cm: Height in centimeters  
            gender: 'male' or 'female'
            body_fat_percentage: Optional body fat % (0-100)
            equation: BMR calculation method
            
        Returns:
            Dictionary with BMR, equation used, and confidence metrics
            
        References:
            - Mifflin-St Jeor: 82% accuracy within 10% (Frankenfield 2005)
            - Katch-McArdle: Best when BF% available (<5% error)
            - Cunningham: Optimal for athletes (BF% < 15%M/25%F)
        """
        
        # Calculate Fat-Free Mass if body fat available
        ffm_kg = None
        if body_fat_percentage is not None:
            ffm_kg = weight_kg * (1 - body_fat_percentage / 100)
        
        # Select optimal equation based on available data
        if equation == EquationType.MIFFLIN_ST_JEOR or ffm_kg is None:
            # Mifflin-St Jeor: Most validated equation (Mifflin et al. 1990)
            if gender.lower() == 'male':
                bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age + 5
            else:
                bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age - 161
            equation_used = 'mifflin_st_jeor'
            
        elif equation == EquationType.KATCH_MCARDLE and ffm_kg is not None:
            # Katch-McArdle: Uses FFM for higher accuracy
            bmr = 370 + (21.6 * ffm_kg)
            equation_used = 'katch_mcardle'
            
        elif equation == EquationType.CUNNINGHAM and ffm_kg is not None:
            # Cunningham: Best for lean athletes
            bmr = 500 + (22 * ffm_kg)
            equation_used = 'cunningham'
            
        else:
            # Fallback to Mifflin-St Jeor
            if gender.lower() == 'male':
                bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age + 5
            else:
                bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age - 161
            equation_used = 'mifflin_st_jeor_fallback'
        
        # Calculate confidence score based on equation accuracy
        confidence = self._calculate_confidence_score(
            equation_used, weight_kg, height_cm, body_fat_percentage, age
        )
        
        return {
            'bmr': round(bmr, 1),
            'equation_used': equation_used,
            'confidence_score': confidence,
            'ffm_kg': ffm_kg
        }
    

    def _calculate_confidence_score(self, 
                                  equation: str, 
                                  weight_kg: float, 
                                  height_cm: float,
                                  body_fat_percentage: Optional[float],
                                  age: int) -> float:
        """
        Calculate prediction confidence based on equation accuracy and user characteristics.
        
        Based on Frankenfield et al. 2005 validation data:
        - Mifflin-St Jeor: 82% within 10% (non-obese), 70% (obese)
        - Lower confidence for elderly (>65) due to limited validation
        - Higher confidence when body composition known
        """
        
        # Base accuracy from literature
        base_equation = equation.split('_')[0] if '_' in equation else equation
        
        # Determine population category
        bmi = weight_kg / ((height_cm / 100) ** 2)
        if bmi < 30:
            population = 'non_obese' 
        else:
            population = 'obese'
            
        # Check if likely athlete (low BF% if available)
        if (body_fat_percentage is not None and 
            ((body_fat_percentage < 15 and 'male' in equation) or 
             (body_fat_percentage < 25 and 'female' in equation))):
            population = 'athletes'
        
        # Get base accuracy
        if base_equation in self.EQUATION_ACCURACY:
            base_confidence = self.EQUATION_ACCURACY[base_equation].get(population, 0.75)
        else:
            base_confidence = 0.75
        
        # Adjust for age (limited elderly validation)
        if age > 65:
            base_confidence *= 0.9
        elif age < 18:
            base_confidence *= 0.85
            
        # Bonus for body composition data
        if body_fat_percentage is not None and 'katch' in equation:
            base_confidence *= 1.1
            
        return min(base_confidence, 1.0)
    

    def calculate_tdee(self, 
                      age: int, 
                      weight_kg: float, 
                      height_cm: float, 
                      gender: str,
                      activity_level: ActivityLevel,
                      body_fat_percentage: Optional[float] = None,
                      training_frequency: int = 0,
                      cardio_minutes_weekly: int = 0,
                      occupation_multiplier: float = 1.0,
                      apply_metabolic_adaptation: bool = True) -> TDEEResult:
        """
        Calculate Total Daily Energy Expenditure with comprehensive adjustments.
        
        Args:
            age: Age in years
            weight_kg: Body weight in kg
            height_cm: Height in cm
            gender: 'male' or 'female'
            activity_level: ActivityLevel enum
            body_fat_percentage: Optional BF% (0-100)
            training_frequency: Resistance training sessions per week
            cardio_minutes_weekly: Cardio training minutes per week
            occupation_multiplier: Job activity adjustment (0.8-1.3)
            apply_metabolic_adaptation: Apply adaptive thermogenesis
            
        Returns:
            TDEEResult with complete breakdown and confidence metrics
        """
        
        with ModelLogger("TDEE Calculation"):
            # Calculate BMR using optimal equation
            bmr_result = self.calculate_bmr(
                age, weight_kg, height_cm, gender, body_fat_percentage
            )
            bmr = bmr_result['bmr']
            
            # Calculate activity multiplier with training adjustments
            base_multiplier = activity_level.value
            
            # Adjust for resistance training frequency (Schoenfeld et al.)
            training_adjustment = min(training_frequency * 0.05, 0.2)  # Max 20% increase
            
            # Adjust for cardio volume (Wilson et al. 2012)
            cardio_adjustment = min(cardio_minutes_weekly / 300, 0.3)  # Max 30% increase
            
            # Occupation adjustment
            total_multiplier = (base_multiplier + training_adjustment + cardio_adjustment) * occupation_multiplier
            total_multiplier = max(1.2, min(total_multiplier, 2.0))  # Physiological bounds
            
            # Calculate TDEE components
            tdee_baseline = bmr * total_multiplier
            
            # Break down components (approximations based on literature)
            neat = bmr * 0.15  # ~15% of BMR (varies 15-30%)
            eat = bmr * (training_adjustment + cardio_adjustment)  # Exercise component
            tef = tdee_baseline * 0.10  # ~10% of total intake (8-15% range)
            
            # Apply metabolic adaptation if historical data available
            metabolic_adaptation_kcal = 0.0
            adaptive_factor = 1.0
            
            if apply_metabolic_adaptation and len(self.metabolic_history) > 14:  # 2+ weeks data
                adaptation_result = self._calculate_metabolic_adaptation()
                adaptive_factor = adaptation_result['factor']
                metabolic_adaptation_kcal = adaptation_result['kcal_adjustment']
            
            # Final TDEE with adaptations
            tdee_final = tdee_baseline * adaptive_factor
            
            return TDEEResult(
                bmr=bmr,
                neat=neat,
                eat=eat,
                tef=tef,
                tdee=tdee_final,
                activity_multiplier=total_multiplier,
                equation_used=bmr_result['equation_used'],
                adaptive_factor=adaptive_factor,
                confidence_score=bmr_result['confidence_score'],
                metabolic_adaptation_kcal=metabolic_adaptation_kcal
            )
    

    def add_metabolic_data_point(self, 
                               date: datetime,
                               weight_kg: float,
                               calories_intake: float,
                               tdee_estimated: float) -> None:
        """
        Add metabolic data point for adaptive thermogenesis tracking.
        
        Args:
            date: Date of measurement
            weight_kg: Body weight in kg
            calories_intake: Calorie intake for the day
            tdee_estimated: Estimated TDEE for the day
        """
        
        # Calculate weight change rate if sufficient data
        weight_change_rate = 0.0
        if len(self.metabolic_history) >= 7:  # 1+ weeks of data
            week_ago_weight = next(
                (data.weight_kg for data in reversed(self.metabolic_history[-7:])
                 if (date - data.date).days >= 6), weight_kg
            )
            weight_change_rate = (weight_kg - week_ago_weight) / 7  # kg per day -> kg per week
            weight_change_rate *= 7
        
        # Add data point
        data_point = MetabolicData(
            date=date,
            weight_kg=weight_kg,
            calories_intake=calories_intake,
            tdee_estimated=tdee_estimated,
            weight_change_rate=weight_change_rate
        )
        
        self.metabolic_history.append(data_point)
        
        # Keep rolling window (12 weeks max)
        if len(self.metabolic_history) > 84:
            self.metabolic_history = self.metabolic_history[-84:]
        
        # Set baseline TDEE if first measurement
        if self.baseline_tdee is None:
            self.baseline_tdee = tdee_estimated
    

    def _calculate_metabolic_adaptation(self) -> Dict:
        """
        Calculate metabolic adaptation based on weight change trends.
        
        Uses CALERIE study findings:
        - 5-13% reduction in TDEE beyond body composition changes
        - Adaptation occurs within 3 months of energy deficit
        - Larger adaptation with greater calorie restriction
        
        Returns:
            Dictionary with adaptation factor and kcal adjustment
        """
        
        if len(self.metabolic_history) < 14:
            return {'factor': 1.0, 'kcal_adjustment': 0.0}
        
        # Analyze last 2 weeks of data
        recent_data = self.metabolic_history[-14:]
        
        # Calculate average energy balance
        avg_calories = np.mean([d.calories_intake for d in recent_data])
        avg_tdee = np.mean([d.tdee_estimated for d in recent_data])
        avg_balance = avg_calories - avg_tdee
        
        # Calculate expected vs actual weight change
        expected_weight_change = avg_balance * 7 / 7700  # kcal to kg per week (7700 kcal/kg fat)
        actual_weight_change = np.mean([d.weight_change_rate for d in recent_data])
        
        # Determine adaptation magnitude based on energy balance
        balance_percentage = avg_balance / avg_tdee
        
        if balance_percentage <= -0.25:  # Severe deficit (>25%)
            base_adaptation = self.ADAPTATION_RATES['severe_deficit']
        elif balance_percentage <= -0.15:  # Moderate deficit (15-25%) 
            base_adaptation = self.ADAPTATION_RATES['moderate_deficit']
        elif balance_percentage <= -0.05:  # Mild deficit (5-15%)
            base_adaptation = self.ADAPTATION_RATES['mild_deficit']
        elif balance_percentage >= 0.25:   # Severe surplus (>25%)
            base_adaptation = self.ADAPTATION_RATES['severe_surplus']
        elif balance_percentage >= 0.15:   # Moderate surplus (15-25%)
            base_adaptation = self.ADAPTATION_RATES['moderate_surplus']
        elif balance_percentage >= 0.05:   # Mild surplus (5-15%)
            base_adaptation = self.ADAPTATION_RATES['mild_surplus']
        else:
            base_adaptation = self.ADAPTATION_RATES['maintenance']
        
        # Adjust based on duration (adaptation increases over time)
        weeks_in_deficit = len([d for d in recent_data if d.calories_intake < d.tdee_estimated])
        duration_multiplier = min(weeks_in_deficit / 4, 1.0)  # Max at 4 weeks
        
        # Final adaptation factor
        adaptation_factor = 1.0 + (base_adaptation * duration_multiplier)
        adaptation_factor = max(0.7, min(adaptation_factor, 1.3))  # Physiological bounds
        
        # Calculate kcal adjustment
        kcal_adjustment = avg_tdee * (adaptation_factor - 1.0)
        
        self.current_adaptation_rate = base_adaptation
        
        self.logger.info(f"Metabolic adaptation: {base_adaptation:.1%}, "
                        f"Factor: {adaptation_factor:.3f}, "
                        f"Adjustment: {kcal_adjustment:.0f} kcal/day")
        
        return {
            'factor': adaptation_factor,
            'kcal_adjustment': kcal_adjustment,
            'weeks_in_balance': weeks_in_deficit,
            'energy_balance_percentage': balance_percentage
        }
    

    def estimate_calorie_needs(self, 
                             current_weight: float,
                             target_weight: float,
                             timeframe_weeks: int,
                             tdee: float) -> Dict:
        """
        Estimate calorie needs for weight change goals.
        
        Args:
            current_weight: Current weight in kg
            target_weight: Target weight in kg
            timeframe_weeks: Timeframe in weeks
            tdee: Current TDEE estimate
            
        Returns:
            Dictionary with calorie recommendations and timeline
        """
        
        weight_change = target_weight - current_weight
        weekly_change = weight_change / timeframe_weeks
        
        # Validate healthy rate (0.25-1.0 kg/week loss, 0.25-0.5 kg/week gain)
        if weight_change < 0:  # Weight loss
            max_healthy_rate = -1.0
            min_healthy_rate = -0.25
        else:  # Weight gain  
            max_healthy_rate = 0.5
            min_healthy_rate = 0.25
        
        is_healthy_rate = min_healthy_rate <= abs(weekly_change) <= max_healthy_rate
        
        # Calculate calorie adjustment (3500 kcal ≈ 0.45 kg, 7700 kcal ≈ 1 kg fat)
        kcal_per_kg = 7700  # More accurate for fat tissue
        weekly_kcal_change = weekly_change * kcal_per_kg
        daily_kcal_change = weekly_kcal_change / 7
        
        target_calories = tdee + daily_kcal_change
        
        # Apply metabolic adaptation prediction for deficits
        if daily_kcal_change < -200:  # Significant deficit
            adaptation_estimate = 0.08  # Expect ~8% reduction
            adjusted_deficit = daily_kcal_change / (1 - adaptation_estimate)
            target_calories = tdee + adjusted_deficit
        
        return {
            'target_calories': round(target_calories),
            'daily_change': round(daily_kcal_change),
            'weekly_weight_change': round(weekly_change, 2),
            'is_healthy_rate': is_healthy_rate,
            'recommended_range': {
                'min': round(target_calories - 100),
                'max': round(target_calories + 100)
            },
            'timeline_weeks': timeframe_weeks,
            'predicted_adaptation': adaptation_estimate if daily_kcal_change < -200 else 0.0
        }
    

    def get_adaptation_status(self) -> Dict:
        """
        Get current metabolic adaptation status and recommendations.
        
        Returns:
            Dictionary with adaptation metrics and recommendations
        """
        
        if len(self.metabolic_history) < 7:
            return {
                'status': 'insufficient_data',
                'message': 'Need at least 1 week of data for adaptation analysis',
                'recommendations': ['Track weight and calorie intake daily']
            }
        
        recent_data = self.metabolic_history[-7:]
        avg_balance = np.mean([d.calories_intake - d.tdee_estimated for d in recent_data])
        
        recommendations = []
        
        if self.current_adaptation_rate < -0.05:  # Significant adaptation
            recommendations.extend([
                'Consider a diet break (1-2 weeks at maintenance)',
                'Increase daily steps for NEAT',
                'Monitor sleep quality (affects hormones)',
                'Consider refeeds every 5-7 days'
            ])
        elif avg_balance < -500:  # Large deficit
            recommendations.extend([
                'Current deficit may be too aggressive',
                'Consider smaller deficit (300-500 kcal)',
                'Monitor energy levels and performance'
            ])
        elif abs(avg_balance) < 100:  # Maintenance
            recommendations.extend([
                'Good energy balance for maintenance',
                'Focus on body recomposition',
                'Maintain current calorie intake'
            ])
        
        return {
            'status': 'active' if abs(self.current_adaptation_rate) > 0.02 else 'minimal',
            'adaptation_percentage': round(self.current_adaptation_rate * 100, 1),
            'avg_energy_balance': round(avg_balance),
            'recommendations': recommendations,
            'data_quality': 'good' if len(self.metabolic_history) >= 14 else 'limited'
        }


    def simulate_metabolic_response(self, 
                                  initial_weight: float,
                                  calorie_intake: float,
                                  initial_tdee: float,
                                  duration_weeks: int) -> List[Dict]:
        """
        Simulate metabolic response over time for ML training data generation.
        
        Args:
            initial_weight: Starting weight in kg
            calorie_intake: Average daily calorie intake
            initial_tdee: Initial TDEE estimate
            duration_weeks: Simulation duration in weeks
            
        Returns:
            List of daily metabolic data points for ML training
        """
        
        simulation_data = []
        current_weight = initial_weight
        current_tdee = initial_tdee
        adaptation_factor = 1.0
        
        for week in range(duration_weeks):
            for day in range(7):
                # Calculate energy balance
                energy_balance = calorie_intake - current_tdee
                
                # Update weight (with realistic noise)
                daily_weight_change = energy_balance / 7700  # kg per day
                weight_noise = np.random.normal(0, NOISE_LEVELS['weight_std'])
                current_weight += daily_weight_change + weight_noise
                
                # Calculate metabolic adaptation (progressive)
                if energy_balance < -200:  # Deficit
                    adaptation_rate = min(week * 0.01, 0.15)  # Progressive adaptation
                    adaptation_factor = 1.0 - adaptation_rate
                elif energy_balance > 300:  # Surplus
                    adaptation_rate = min(week * 0.005, 0.08)  # Smaller surplus adaptation
                    adaptation_factor = 1.0 + adaptation_rate
                
                # Update TDEE with adaptation
                current_tdee = initial_tdee * adaptation_factor
                
                # Add realistic calorie intake noise
                daily_calories = calorie_intake + np.random.normal(0, NOISE_LEVELS['calories_std'])
                
                simulation_data.append({
                    'day': week * 7 + day + 1,
                    'weight_kg': round(current_weight, 2),
                    'calorie_intake': round(daily_calories),
                    'tdee_estimated': round(current_tdee),
                    'adaptation_factor': round(adaptation_factor, 4),
                    'energy_balance': round(daily_calories - current_tdee)
                })
        
        return simulation_data


# Utility functions for quick calculations
def quick_tdee(weight_kg: float, height_cm: float, age: int, gender: str, 
               activity_level: str = 'moderately_active') -> float:
    """
    Quick TDEE calculation using Mifflin-St Jeor and standard activity multipliers.
    
    Args:
        weight_kg: Weight in kg
        height_cm: Height in cm  
        age: Age in years
        gender: 'male' or 'female'
        activity_level: Activity level string
        
    Returns:
        Estimated TDEE in kcal/day
    """
    
    # Calculate BMR (Mifflin-St Jeor)
    if gender.lower() == 'male':
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age + 5
    else:
        bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age - 161
    
    # Apply activity multiplier
    multipliers = {
        'sedentary': 1.2,
        'lightly_active': 1.375,
        'moderately_active': 1.55,
        'very_active': 1.725,
        'extremely_active': 1.9
    }
    
    multiplier = multipliers.get(activity_level.lower(), 1.55)
    return round(bmr * multiplier)


def estimate_body_fat_navy(neck_cm: float, waist_cm: float, height_cm: float, 
                          hip_cm: Optional[float] = None, gender: str = 'male') -> float:
    """
    Estimate body fat percentage using US Navy method.
    
    Args:
        neck_cm: Neck circumference in cm
        waist_cm: Waist circumference in cm
        height_cm: Height in cm
        hip_cm: Hip circumference in cm (required for females)
        gender: 'male' or 'female'
        
    Returns:
        Estimated body fat percentage (0-100)
    """
    
    import math
    
    if gender.lower() == 'male':
        # Male formula
        body_fat = (495 / (1.0324 - 0.19077 * math.log10(waist_cm - neck_cm) + 
                          0.15456 * math.log10(height_cm))) - 450
    else:
        # Female formula (requires hip measurement)
        if hip_cm is None:
            raise ValueError("Hip measurement required for female body fat estimation")
        
        body_fat = (495 / (1.29579 - 0.35004 * math.log10(waist_cm + hip_cm - neck_cm) + 
                          0.22100 * math.log10(height_cm))) - 450
    
    return max(5, min(body_fat, 50))  # Physiological bounds


if __name__ == "__main__":
    """Example usage and testing."""
    
    # Initialize calculator
    calc = TDEECalculator()
    
    # Example calculation
    result = calc.calculate_tdee(
        age=25,
        weight_kg=80,
        height_cm=180,
        gender='male',
        activity_level=ActivityLevel.MODERATELY_ACTIVE,
        body_fat_percentage=15,
        training_frequency=4,
        cardio_minutes_weekly=150
    )
    
    print(f"TDEE Results:")
    print(f"BMR: {result.bmr:.0f} kcal/day")
    print(f"TDEE: {result.tdee:.0f} kcal/day")
    print(f"Activity Multiplier: {result.activity_multiplier:.2f}")
    print(f"Equation: {result.equation_used}")
    print(f"Confidence: {result.confidence_score:.1%}")
    print(f"Adaptive Factor: {result.adaptive_factor:.3f}")