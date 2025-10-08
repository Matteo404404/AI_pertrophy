"""
TDEE Calculator Module for Scientific Hypertrophy Trainer

This module implements evidence-based Total Daily Energy Expenditure (TDEE) calculations
using multiple validated equations and adaptive metabolic adjustments based on longitudinal
weight change data.

References:
- Mifflin et al. 1990: Most accurate BMR equation for general population
- Katch-McArdle: Preferred when accurate body fat % available (FFMI validation)  
- Cunningham: Best for lean athletic populations
- Adaptive Thermogenesis: Metabolic adaptation ~20-30% in calorie restriction (Leibel et al. 1995)
- Activity multipliers: Validated ranges from DLW studies (Westerterp 2013)
- Redman et al. 2009: TDEE reduction with caloric restriction (-431±51 kcal/d at 3 months)

Author: Scientific Hypertrophy Trainer ML Team
"""

from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np
from datetime import datetime, timedelta
import logging

# Set up logging
logger = logging.getLogger(__name__)


class ActivityLevel(Enum):
    """Activity level classifications with evidence-based multipliers."""
    SEDENTARY = 1.2          # Desk job, no exercise  
    LIGHTLY_ACTIVE = 1.375   # Light exercise 1-3 days/week
    MODERATELY_ACTIVE = 1.55 # Moderate exercise 3-5 days/week
    VERY_ACTIVE = 1.725      # Heavy exercise 6-7 days/week
    EXTREMELY_ACTIVE = 1.9   # Physical job + exercise, 2x/day training


class EquationType(Enum):
    """BMR calculation methods with specific use cases."""
    MIFFLIN_ST_JEOR = "mifflin"    # Most accurate for general population
    KATCH_MCARDLE = "katch"        # Best when BF% known accurately  
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
    estimated_surplus: float     # Calories above maintenance
    weight_change_trend: float   # Weekly weight change trend (kg/week)


class TDEECalculator:
    """
    Advanced TDEE calculator with adaptive metabolic adjustments.
    
    Features:
    - Multiple validated BMR equations with automatic selection
    - Activity-specific multipliers based on training volume
    - Adaptive thermogenesis modeling (Leibel et al. 1995)
    - Longitudinal weight trend analysis for metabolic adaptation
    - Body composition considerations (FFMI validation)
    - Training volume-based EAT estimation
    
    Scientific Backing:
    - Mifflin-St Jeor: Most accurate BMR equation (±10% for 90% of population)
    - Adaptive thermogenesis: 15-20% TDEE reduction in energy restriction
    - NEAT variation: 100-800 kcal/day individual differences
    - TEF: 8-15% of total caloric intake, higher with protein
    """
    
    # Metabolic adaptation constants (Leibel et al. 1995, Redman et al. 2009)
    ADAPTATION_RATES = {
        'severe_deficit': -0.20,   # 20% reduction in severe restriction (>25% deficit)
        'moderate_deficit': -0.15, # 15% reduction in moderate restriction (15-25% deficit)
        'mild_deficit': -0.08,     # 8% reduction in mild restriction (<15% deficit)
        'maintenance': 0.02,       # 2% natural variation
        'mild_surplus': 0.05,      # 5% increase in mild surplus
        'surplus': 0.08,           # 8% increase in overfeeding
    }
    
    # TEF percentages by macronutrient (Westerterp 2004)
    TEF_MACROS = {
        'protein': 0.25,       # 20-30% thermic effect
        'carbs': 0.08,         # 5-10% thermic effect  
        'fats': 0.03,          # 0-5% thermic effect
        'mixed': 0.10          # Typical mixed meal
    }
    
    # Training volume to EAT conversion (kcal per set, based on exercise physiology)
    EAT_PER_SET = {
        'compound': 12,        # Squat, deadlift, bench press
        'isolation': 8,        # Bicep curls, leg extensions
        'cardio_moderate': 300, # 30min moderate cardio
        'cardio_high': 450     # 30min high intensity
    }
    
    def __init__(self):
        """Initialize TDEE calculator with physiological constants."""
        self.weight_history: List[Tuple[datetime, float]] = []
        self.calorie_history: List[Tuple[datetime, float]] = []
        self.training_history: List[Tuple[datetime, Dict]] = []
        
    def calculate_bmr(self, 
                     weight_kg: float, 
                     height_cm: float, 
                     age_years: int,
                     sex: str,
                     body_fat_percent: Optional[float] = None,
                     equation: EquationType = EquationType.MIFFLIN_ST_JEOR) -> Tuple[float, str]:
        """
        Calculate Basal Metabolic Rate using multiple validated equations.
        
        Args:
            weight_kg: Body weight in kilograms
            height_cm: Height in centimeters  
            age_years: Age in years
            sex: 'male' or 'female'
            body_fat_percent: Body fat percentage (if available)
            equation: BMR equation to use
            
        Returns:
            Tuple of (BMR in kcal/day, equation_name)
            
        Scientific References:
        - Mifflin-St Jeor: Most accurate for general population (Mifflin et al. 1990)
        - Katch-McArdle: Best when accurate BF% available
        - Cunningham: Optimal for lean athletes (BF% < 15%M/25%F)
        """
        sex_lower = sex.lower()
        
        # Auto-select best equation based on available data
        if equation == EquationType.MIFFLIN_ST_JEOR or body_fat_percent is None:
            # Mifflin-St Jeor equation (most accurate for general population)
            if sex_lower == 'male':
                bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age_years + 5
            else:  # female
                bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age_years - 161
            equation_name = "Mifflin-St Jeor"
            
        elif equation == EquationType.KATCH_MCARDLE and body_fat_percent is not None:
            # Katch-McArdle equation (uses lean body mass)
            lean_mass_kg = weight_kg * (1 - body_fat_percent / 100)
            bmr = 370 + (21.6 * lean_mass_kg)
            equation_name = "Katch-McArdle"
            
        elif equation == EquationType.CUNNINGHAM and body_fat_percent is not None:
            # Cunningham equation (best for lean, trained individuals)
            lean_mass_kg = weight_kg * (1 - body_fat_percent / 100)
            bmr = 500 + (22 * lean_mass_kg)
            equation_name = "Cunningham"
            
        else:
            # Fallback to Mifflin-St Jeor
            if sex_lower == 'male':
                bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age_years + 5
            else:
                bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age_years - 161
            equation_name = "Mifflin-St Jeor (fallback)"
            
        return bmr, equation_name
    
    def estimate_neat(self, 
                     activity_level: ActivityLevel,
                     occupation_type: str = "sedentary",
                     daily_steps: Optional[int] = None) -> float:
        """
        Estimate Non-Exercise Activity Thermogenesis.
        
        NEAT accounts for 15-30% of total daily energy expenditure in healthy individuals
        and shows the highest inter-individual variability (100-800 kcal/day).
        
        Args:
            activity_level: General activity classification
            occupation_type: "sedentary", "standing", "physical"
            daily_steps: Average daily step count if available
            
        Returns:
            NEAT in kcal/day
            
        References:
        - Levine 2004: NEAT variation 100-800 kcal/day
        - Westerterp 2013: Activity multipliers from DLW studies
        """
        # Base NEAT estimate from activity multiplier
        base_multiplier = activity_level.value
        neat_factor = base_multiplier - 1.0  # Remove BMR component
        
        # Adjust for occupation
        occupation_multipliers = {
            "sedentary": 1.0,
            "standing": 1.2,
            "physical": 1.5,
            "manual_labor": 1.8
        }
        
        occupation_mult = occupation_multipliers.get(occupation_type, 1.0)
        neat_factor *= occupation_mult
        
        # Adjust for step count if available
        if daily_steps is not None:
            # Rough conversion: 2000 steps = ~100 kcal for average person
            step_kcal = max(0, (daily_steps - 2000) * 0.05)
            neat_factor += step_kcal / 2000  # Normalize to multiplier
        
        # Convert to absolute value (will be multiplied by BMR later)
        return neat_factor
    
    def estimate_eat(self, 
                    training_sessions_per_week: float,
                    average_session_duration_min: float = 60,
                    training_intensity: str = "moderate",
                    cardio_minutes_per_week: float = 0) -> float:
        """
        Estimate Exercise Activity Thermogenesis.
        
        Args:
            training_sessions_per_week: Resistance training frequency
            average_session_duration_min: Session duration
            training_intensity: "low", "moderate", "high"
            cardio_minutes_per_week: Weekly cardio volume
            
        Returns:
            EAT in kcal/day (averaged)
        """
        # Resistance training EAT
        intensity_multipliers = {
            "low": 0.8,      # RPE 5-6, long rest periods
            "moderate": 1.0,  # RPE 7-8, moderate rest
            "high": 1.3      # RPE 9+, short rest, high volume
        }
        
        intensity_mult = intensity_multipliers.get(training_intensity, 1.0)
        
        # Estimate kcal per session (rough approximation)
        kcal_per_session = (average_session_duration_min / 60) * 250 * intensity_mult
        weekly_resistance_kcal = training_sessions_per_week * kcal_per_session
        
        # Cardio EAT
        weekly_cardio_kcal = cardio_minutes_per_week * 8  # ~8 kcal/min moderate cardio
        
        # Daily average
        total_weekly_eat = weekly_resistance_kcal + weekly_cardio_kcal
        daily_eat = total_weekly_eat / 7
        
        return daily_eat
    
    def calculate_tef(self, 
                     total_calories: float,
                     protein_percent: float = 15,
                     carb_percent: float = 50,
                     fat_percent: float = 35) -> float:
        """
        Calculate Thermic Effect of Food based on macronutrient composition.
        
        Args:
            total_calories: Total daily caloric intake
            protein_percent: Percentage of calories from protein
            carb_percent: Percentage of calories from carbohydrates  
            fat_percent: Percentage of calories from fats
            
        Returns:
            TEF in kcal/day
            
        References:
        - Westerterp 2004: Macronutrient-specific TEF values
        - Protein: 20-30% TEF, Carbs: 5-10%, Fats: 0-5%
        """
        # Normalize percentages
        total_percent = protein_percent + carb_percent + fat_percent
        if total_percent != 100:
            protein_percent = protein_percent / total_percent * 100
            carb_percent = carb_percent / total_percent * 100
            fat_percent = fat_percent / total_percent * 100
        
        # Calculate TEF for each macronutrient
        protein_tef = (total_calories * protein_percent / 100) * self.TEF_MACROS['protein']
        carb_tef = (total_calories * carb_percent / 100) * self.TEF_MACROS['carbs']
        fat_tef = (total_calories * fat_percent / 100) * self.TEF_MACROS['fats']
        
        total_tef = protein_tef + carb_tef + fat_tef
        
        return total_tef
    
    def calculate_adaptive_factor(self, 
                                 current_weight: float,
                                 baseline_weight: float,
                                 average_calorie_intake: float,
                                 estimated_maintenance: float,
                                 weeks_in_phase: int) -> float:
        """
        Calculate metabolic adaptation factor based on weight change and energy balance.
        
        Based on Leibel et al. 1995 and Redman et al. 2009 research showing:
        - 10% weight loss → 250-300 kcal/day TDEE reduction
        - 20% weight loss → 400-500 kcal/day TDEE reduction
        - Adaptation occurs within 3-6 weeks and persists long-term
        
        Args:
            current_weight: Current body weight (kg)
            baseline_weight: Starting/reference weight (kg)
            average_calorie_intake: Recent average calorie intake
            estimated_maintenance: Estimated maintenance calories
            weeks_in_phase: Duration of current diet/surplus phase
            
        Returns:
            Adaptive factor (0.8-1.2, where 1.0 = no adaptation)
        """
        # Calculate weight change percentage
        weight_change_percent = (current_weight - baseline_weight) / baseline_weight * 100
        
        # Calculate energy balance
        energy_balance = average_calorie_intake - estimated_maintenance
        deficit_percent = abs(energy_balance) / estimated_maintenance * 100
        
        # Determine adaptation based on energy balance severity
        if energy_balance < -500:  # Severe deficit
            base_adaptation = self.ADAPTATION_RATES['severe_deficit']
        elif energy_balance < -300:  # Moderate deficit
            base_adaptation = self.ADAPTATION_RATES['moderate_deficit'] 
        elif energy_balance < -100:  # Mild deficit
            base_adaptation = self.ADAPTATION_RATES['mild_deficit']
        elif energy_balance > 300:   # Surplus
            base_adaptation = self.ADAPTATION_RATES['surplus']
        elif energy_balance > 100:   # Mild surplus
            base_adaptation = self.ADAPTATION_RATES['mild_surplus']
        else:                        # Maintenance
            base_adaptation = self.ADAPTATION_RATES['maintenance']
        
        # Scale by weight change magnitude (more weight loss = more adaptation)
        if weight_change_percent < 0:  # Weight loss
            weight_factor = min(abs(weight_change_percent) / 10, 2.0)  # Cap at 20% loss effect
        else:  # Weight gain
            weight_factor = min(weight_change_percent / 15, 1.0)  # Less adaptation in surplus
        
        # Time factor (adaptation develops over 3-6 weeks)
        time_factor = min(weeks_in_phase / 4, 1.0)  # Full adaptation by week 4
        
        # Calculate final adaptive factor
        adaptation_magnitude = base_adaptation * weight_factor * time_factor
        adaptive_factor = 1.0 + adaptation_magnitude
        
        # Constrain to physiologically realistic range
        adaptive_factor = max(0.75, min(1.25, adaptive_factor))
        
        logger.info(f"Adaptive factor: {adaptive_factor:.3f} "
                   f"(weight change: {weight_change_percent:.1f}%, "
                   f"energy balance: {energy_balance:.0f} kcal)")
        
        return adaptive_factor
    
    def calculate_tdee(self,
                      weight_kg: float,
                      height_cm: float, 
                      age_years: int,
                      sex: str,
                      activity_level: ActivityLevel,
                      body_fat_percent: Optional[float] = None,
                      training_sessions_per_week: float = 0,
                      cardio_minutes_per_week: float = 0,
                      daily_steps: Optional[int] = None,
                      current_calories: Optional[float] = None,
                      baseline_weight: Optional[float] = None,
                      weeks_in_phase: int = 0,
                      protein_percent: float = 20,
                      equation: EquationType = EquationType.MIFFLIN_ST_JEOR) -> TDEEResult:
        """
        Calculate comprehensive TDEE with all components and adaptive adjustments.
        
        Args:
            weight_kg: Current body weight
            height_cm: Height in centimeters
            age_years: Age in years  
            sex: 'male' or 'female'
            activity_level: General activity classification
            body_fat_percent: Body fat percentage (optional)
            training_sessions_per_week: Resistance training frequency
            cardio_minutes_per_week: Weekly cardio volume
            daily_steps: Average daily steps (optional)
            current_calories: Current calorie intake for adaptation calc
            baseline_weight: Reference weight for adaptation calc
            weeks_in_phase: Duration of current diet phase
            protein_percent: Protein percentage of diet (affects TEF)
            equation: BMR equation to use
            
        Returns:
            TDEEResult with complete breakdown
        """
        # Calculate BMR
        bmr, equation_name = self.calculate_bmr(
            weight_kg, height_cm, age_years, sex, body_fat_percent, equation
        )
        
        # Calculate NEAT (as multiplier of BMR)
        neat_multiplier = self.estimate_neat(activity_level, daily_steps=daily_steps)
        neat = bmr * neat_multiplier
        
        # Calculate EAT
        eat = self.estimate_eat(
            training_sessions_per_week, 
            cardio_minutes_per_week=cardio_minutes_per_week
        )
        
        # Calculate TEF (estimate based on typical intake or provided calories)
        estimated_calories = current_calories if current_calories else bmr * activity_level.value
        tef = self.calculate_tef(estimated_calories, protein_percent)
        
        # Calculate base TDEE
        base_tdee = bmr + neat + eat + tef
        
        # Apply metabolic adaptation if enough data available
        adaptive_factor = 1.0
        if current_calories and baseline_weight and weeks_in_phase > 0:
            adaptive_factor = self.calculate_adaptive_factor(
                weight_kg, baseline_weight, current_calories, base_tdee, weeks_in_phase
            )
        
        final_tdee = base_tdee * adaptive_factor
        
        # Calculate confidence score based on data availability
        confidence_score = self._calculate_confidence_score(
            body_fat_percent, daily_steps, weeks_in_phase, baseline_weight
        )
        
        # Estimate surplus/deficit
        estimated_surplus = (current_calories - final_tdee) if current_calories else 0
        
        # Calculate weight change trend
        weight_change_trend = self._calculate_weight_trend()
        
        return TDEEResult(
            bmr=bmr,
            neat=neat,
            eat=eat,
            tef=tef,
            tdee=final_tdee,
            activity_multiplier=activity_level.value,
            equation_used=equation_name,
            adaptive_factor=adaptive_factor,
            confidence_score=confidence_score,
            estimated_surplus=estimated_surplus,
            weight_change_trend=weight_change_trend
        )
    
    def _calculate_confidence_score(self, 
                                   body_fat_percent: Optional[float],
                                   daily_steps: Optional[int],
                                   weeks_in_phase: int,
                                   baseline_weight: Optional[float]) -> float:
        """Calculate confidence score for TDEE estimate (0-1)."""
        score = 0.6  # Base score for basic calculation
        
        # Bonus for body fat percentage (allows better BMR equation)
        if body_fat_percent is not None:
            score += 0.15
            
        # Bonus for step count data
        if daily_steps is not None:
            score += 0.1
            
        # Bonus for adaptation data
        if baseline_weight is not None and weeks_in_phase > 2:
            score += 0.15
            
        return min(1.0, score)
    
    def _calculate_weight_trend(self) -> float:
        """Calculate recent weight change trend from history."""
        if len(self.weight_history) < 2:
            return 0.0
            
        # Use last 4 weeks of data if available
        recent_weights = self.weight_history[-28:] if len(self.weight_history) >= 28 else self.weight_history
        
        if len(recent_weights) < 2:
            return 0.0
            
        # Simple linear trend
        dates = [(w[0] - recent_weights[0][0]).days for w in recent_weights]
        weights = [w[1] for w in recent_weights]
        
        # Calculate slope (kg per day)
        n = len(dates)
        sum_x = sum(dates)
        sum_y = sum(weights)
        sum_xy = sum(x * y for x, y in zip(dates, weights))
        sum_x2 = sum(x * x for x in dates)
        
        if n * sum_x2 - sum_x * sum_x == 0:
            return 0.0
            
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        
        # Convert to kg per week
        return slope * 7
    
    def add_weight_entry(self, date: datetime, weight_kg: float):
        """Add weight entry to history for trend analysis."""
        self.weight_history.append((date, weight_kg))
        # Keep only last 12 weeks
        if len(self.weight_history) > 84:
            self.weight_history = self.weight_history[-84:]
    
    def add_calorie_entry(self, date: datetime, calories: float):
        """Add calorie entry to history.""" 
        self.calorie_history.append((date, calories))
        # Keep only last 12 weeks
        if len(self.calorie_history) > 84:
            self.calorie_history = self.calorie_history[-84:]
    
    def get_tdee_summary(self, tdee_result: TDEEResult) -> Dict[str, Union[float, str]]:
        """Get formatted summary of TDEE calculation."""
        return {
            'total_tdee': round(tdee_result.tdee),
            'bmr': round(tdee_result.bmr),
            'neat': round(tdee_result.neat),
            'eat': round(tdee_result.eat),
            'tef': round(tdee_result.tef),
            'equation': tdee_result.equation_used,
            'adaptive_factor': round(tdee_result.adaptive_factor, 3),
            'confidence': f"{tdee_result.confidence_score:.1%}",
            'estimated_surplus': round(tdee_result.estimated_surplus),
            'weight_trend_kg_per_week': round(tdee_result.weight_change_trend, 3)
        }


# Convenience functions for quick calculations
def calculate_maintenance_calories(weight_kg: float,
                                 height_cm: float,
                                 age_years: int,
                                 sex: str,
                                 activity_level: str = "moderately_active") -> float:
    """
    Quick maintenance calorie calculation.
    
    Args:
        weight_kg: Body weight in kg
        height_cm: Height in cm
        age_years: Age in years
        sex: 'male' or 'female'
        activity_level: Activity level string
        
    Returns:
        Maintenance calories per day
    """
    calculator = TDEECalculator()
    
    # Map string to enum
    activity_map = {
        "sedentary": ActivityLevel.SEDENTARY,
        "lightly_active": ActivityLevel.LIGHTLY_ACTIVE,
        "moderately_active": ActivityLevel.MODERATELY_ACTIVE,
        "very_active": ActivityLevel.VERY_ACTIVE,
        "extremely_active": ActivityLevel.EXTREMELY_ACTIVE
    }
    
    activity_enum = activity_map.get(activity_level, ActivityLevel.MODERATELY_ACTIVE)
    
    result = calculator.calculate_tdee(
        weight_kg=weight_kg,
        height_cm=height_cm,
        age_years=age_years,
        sex=sex,
        activity_level=activity_enum
    )
    
    return result.tdee


def estimate_surplus_for_goal(current_tdee: float, 
                            goal_rate_kg_per_week: float,
                            body_fat_percent: Optional[float] = None) -> float:
    """
    Estimate caloric surplus needed for specific muscle gain rate.
    
    Based on research:
    - 1 lb muscle ≈ 2500-3500 kcal (varies by individual)
    - Higher body fat = more efficient muscle building initially
    
    Args:
        current_tdee: Current maintenance calories
        goal_rate_kg_per_week: Target muscle gain rate
        body_fat_percent: Current body fat percentage
        
    Returns:
        Recommended daily caloric surplus
    """
    # Convert kg/week to lb/week
    goal_rate_lb_per_week = goal_rate_kg_per_week * 2.20462
    
    # Calories per pound of muscle (conservative estimate)
    kcal_per_lb_muscle = 3000
    
    # Calculate weekly surplus needed
    weekly_surplus = goal_rate_lb_per_week * kcal_per_lb_muscle
    daily_surplus = weekly_surplus / 7
    
    # Adjust for body fat (leaner individuals need larger surplus)
    if body_fat_percent:
        if body_fat_percent < 10:  # Very lean
            daily_surplus *= 1.3
        elif body_fat_percent < 15:  # Lean
            daily_surplus *= 1.15
        elif body_fat_percent > 25:  # Higher body fat
            daily_surplus *= 0.85
    
    # Reasonable bounds
    daily_surplus = max(100, min(800, daily_surplus))
    
    return daily_surplus


if __name__ == "__main__":
    # Example usage
    calc = TDEECalculator()
    
    # Example calculation
    result = calc.calculate_tdee(
        weight_kg=80,
        height_cm=180,
        age_years=25,
        sex="male",
        activity_level=ActivityLevel.MODERATELY_ACTIVE,
        training_sessions_per_week=4,
        cardio_minutes_per_week=90,
        protein_percent=25
    )
    
    print("TDEE Calculation Results:")
    print("=" * 40)
    summary = calc.get_tdee_summary(result)
    for key, value in summary.items():
        print(f"{key.replace('_', ' ').title()}: {value}")
