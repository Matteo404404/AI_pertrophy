"""
Advanced Synthetic Data Generator for Scientific Hypertrophy Trainer

Generates scientifically realistic training data for 3 demo users over 90 days,
incorporating evidence-based muscle gain rates, minimal real-world interruptions,
and proper newbie gains for beginners.

Scientific References:
- Schoenfeld et al. 2017: Volume-hypertrophy dose-response
- Morton et al. 2018: Protein intake 1.6-2.2g/kg optimal  
- McDonald/Aragon: Beginners 20-25 lbs (9-11kg) first year for males
- Minimum effective volume: 1 set 3x/week creates stimulus

Author: Scientific Hypertrophy Trainer ML Team
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from enum import Enum

# Import TDEE calculator
try:
    from .tdee_calculator import TDEECalculator, ActivityLevel, calculate_maintenance_calories
except ImportError:
    try:
        from tdee_calculator import TDEECalculator, ActivityLevel, calculate_maintenance_calories
    except ImportError:
        print("Warning: TDEE calculator not found - using simplified version")
        
        def calculate_maintenance_calories(weight_kg, height_cm, age_years, sex, activity_level="moderately_active"):
            """Simplified TDEE calculation."""
            if sex.lower() == 'male':
                bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age_years + 5
            else:
                bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age_years - 161
            return bmr * 1.55  # Moderate activity

# Set up logging
logger = logging.getLogger(__name__)


class ExperienceLevel(Enum):
    """Training experience classifications with gain rates."""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate" 
    ADVANCED = "advanced"


class Phase(Enum):
    """Training phases - only building and cutting."""
    BUILDING = "building"      # Muscle building phase
    CUTTING = "cutting"        # Fat loss phase


@dataclass
class UserProfile:
    """Complete user profile for data generation."""
    name: str
    age: int
    weight_start_kg: float
    height_cm: float
    sex: str
    experience_level: ExperienceLevel
    body_fat_start: float
    training_age_years: float
    baseline_gain_rate_kg_week: float  # Under optimal conditions
    consistency_factor: float  # 0.7-0.95 (adherence to plan)
    recovery_capacity: float   # 0.8-1.2 (individual recovery ability)
    genetic_response: float    # 0.85-1.15 (genetic response to training)


class AdvancedDataGenerator:
    """
    Generates scientifically realistic hypertrophy training data.
    
    Features:
    - Evidence-based volume prescriptions (minimum effective volume)
    - Realistic newbie gains for beginners (9-11kg first year for males)
    - Building vs cutting phases only
    - Minimal realistic interruptions (illness, vacation)
    - Proper fatigue detection through progress stalling
    - Individual response variation
    - Surplus not required for muscle building (maintenance works)
    """
    
    # Muscle gain rates from literature (kg/week under optimal conditions)
    GAIN_RATES = {
        ExperienceLevel.BEGINNER: {
            'male': (0.15, 0.25),    # 0.33-0.55 lb/week (realistic for newbie gains)
            'female': (0.08, 0.12)   # 0.18-0.26 lb/week (female newbie gains)
        },
        ExperienceLevel.INTERMEDIATE: {
            'male': (0.05, 0.10),    # 0.11-0.22 lb/week  
            'female': (0.025, 0.05)  # 0.055-0.11 lb/week
        },
        ExperienceLevel.ADVANCED: {
            'male': (0.02, 0.04),    # 0.044-0.088 lb/week
            'female': (0.01, 0.02)   # 0.022-0.044 lb/week
        }
    }
    
    # Minimum effective volumes (sets per week)
    MIN_EFFECTIVE_VOLUME = {
        ExperienceLevel.BEGINNER: 8,     # 1 set 3x/week per muscle
        ExperienceLevel.INTERMEDIATE: 12, # Need more volume
        ExperienceLevel.ADVANCED: 16     # Even more volume needed
    }
    
    # Maximum recoverable volumes (sets per week)
    MAX_RECOVERABLE_VOLUME = {
        ExperienceLevel.BEGINNER: 20,    # Lower recovery capacity
        ExperienceLevel.INTERMEDIATE: 28, # Better recovery
        ExperienceLevel.ADVANCED: 35     # Best recovery
    }
    
    # Noise parameters for realistic variation
    NOISE_PATTERNS = {
        'weight_daily_std': 0.4,      # kg (glycogen, hydration, gut content)
        'calories_daily_std': 180,     # kcal (meal timing, measurement error)
        'protein_daily_std': 15,       # g (meal composition variation)
        'sleep_duration_std': 0.6,     # hours
        'sleep_quality_std': 1.2,      # 1-10 scale
        'training_volume_std': 1.8,    # sets (fatigue, time constraints)
        'hrv_daily_std': 8,            # RMSSD units
        'stress_daily_std': 1.0        # 1-10 scale
    }
    
    def __init__(self, start_date: datetime = None):
        """Initialize data generator."""
        self.start_date = start_date or datetime(2024, 1, 1)
        
        # Initialize user profiles
        self.users = self._create_user_profiles()
        
        # Realistic interruptions only
        self.vacation_weeks = [6, 11]   # 2 vacation weeks over 90 days
        
    def _create_user_profiles(self) -> List[UserProfile]:
        """Create the three demo user profiles with realistic characteristics."""
        users = [
            UserProfile(
                name="Alex_Beginner",
                age=22,
                weight_start_kg=70.0,
                height_cm=175,
                sex="male", 
                experience_level=ExperienceLevel.BEGINNER,
                body_fat_start=16.0,
                training_age_years=0.5,
                baseline_gain_rate_kg_week=0.18,  # Realistic newbie gains ~9kg/year
                consistency_factor=0.78,
                recovery_capacity=1.05,  # Young advantage
                genetic_response=0.95    # Slightly below average initially
            ),
            UserProfile(
                name="Sarah_Intermediate", 
                age=28,
                weight_start_kg=62.0,
                height_cm=165,
                sex="female",
                experience_level=ExperienceLevel.INTERMEDIATE,
                body_fat_start=22.0,
                training_age_years=3.2,
                baseline_gain_rate_kg_week=0.04,  # Realistic intermediate female
                consistency_factor=0.88,
                recovery_capacity=0.92,
                genetic_response=1.08    # Good responder
            ),
            UserProfile(
                name="Mike_Advanced",
                age=35,
                weight_start_kg=85.0,
                height_cm=182,
                sex="male",
                experience_level=ExperienceLevel.ADVANCED,
                body_fat_start=12.0,
                training_age_years=8.5,
                baseline_gain_rate_kg_week=0.03,  # Slow advanced gains
                consistency_factor=0.94,
                recovery_capacity=0.88,  # Age-related decline
                genetic_response=1.02    # Average response
            )
        ]
        
        logger.info(f"Created {len(users)} user profiles for data generation")
        return users
    
    def generate_user_data(self, user: UserProfile, days: int = 90) -> pd.DataFrame:
        """
        Generate complete dataset for one user over specified period.
        
        Args:
            user: User profile with characteristics
            days: Number of days to generate
            
        Returns:
            DataFrame with daily data points and engineered features
        """
        logger.info(f"Generating {days} days of data for {user.name}")
        
        # Initialize data storage
        data = []
        current_weight = user.weight_start_kg
        current_muscle_mass = self._estimate_initial_muscle_mass(user)
        
        # Calculate initial TDEE
        baseline_tdee = calculate_maintenance_calories(
            weight_kg=user.weight_start_kg,
            height_cm=user.height_cm,
            age_years=user.age,
            sex=user.sex,
            activity_level="moderately_active"
        )
        
        # Determine phase (mostly building, some cutting)
        phase_schedule = self._create_phase_schedule(days)
        
        for day in range(days):
            date = self.start_date + timedelta(days=day)
            week = day // 7 + 1
            day_of_week = day % 7
            
            # Get current phase
            current_phase = phase_schedule[day]
            
            # Generate training data
            training_data = self._generate_training_day(
                user, day, week, day_of_week, current_phase
            )
            
            # Generate nutrition data
            nutrition_data = self._generate_nutrition_day(
                user, day, week, baseline_tdee, current_weight, current_phase
            )
            
            # Generate sleep data
            sleep_data = self._generate_sleep_day(user, day, week)
            
            # Generate body metrics
            body_data = self._generate_body_metrics(
                user, day, week, current_weight, current_muscle_mass, training_data, nutrition_data, sleep_data, current_phase
            )
            
            # Update progressive values
            current_weight = body_data['weight_kg']
            current_muscle_mass = body_data['estimated_muscle_mass_kg']
            
            # Combine all data
            day_data = {
                'date': date,
                'user_name': user.name,
                'day': day + 1,
                'week': week,
                'day_of_week': day_of_week,
                'phase': current_phase,
                **training_data,
                **nutrition_data,
                **sleep_data, 
                **body_data
            }
            
            data.append(day_data)
        
        # Convert to DataFrame and add engineered features
        df = pd.DataFrame(data)
        df = self._add_engineered_features(df, user)
        
        logger.info(f"Generated {len(df)} data points for {user.name}")
        return df
    
    def _create_phase_schedule(self, days: int) -> List[str]:
        """Create realistic phase schedule - mostly building."""
        schedule = []
        
        for day in range(days):
            week = day // 7 + 1
            
            # Vacation weeks (minimal training, maintenance calories)
            if week in self.vacation_weeks:
                schedule.append('vacation')
            # Illness (random, 3% chance - reduced from 5%)
            elif np.random.random() < 0.03:
                schedule.append('illness')
            # Small cutting phase (weeks 9-10)
            elif 9 <= week <= 10:
                schedule.append('cutting')
            # Everything else is building
            else:
                schedule.append('building')
        
        return schedule
    
    def _estimate_initial_muscle_mass(self, user: UserProfile) -> float:
        """Estimate initial muscle mass using FFMI calculations."""
        # Fat-free mass
        ffm = user.weight_start_kg * (1 - user.body_fat_start / 100)
        
        # Estimate muscle mass as ~40-45% of FFM for trained individuals
        muscle_percentage = 0.42 if user.sex == 'male' else 0.36
        estimated_muscle_mass = ffm * muscle_percentage
        
        return estimated_muscle_mass
    
    def _generate_training_day(self, user: UserProfile, day: int, week: int, 
                              day_of_week: int, phase: str) -> Dict:
        """Generate realistic training data for one day."""
        # Training frequency based on experience level
        training_days_per_week = {
            ExperienceLevel.BEGINNER: 3,    # Full body 3x/week
            ExperienceLevel.INTERMEDIATE: 4, # Upper/lower 4x/week
            ExperienceLevel.ADVANCED: 5     # Push/pull/legs 5x/week
        }
        
        weekly_frequency = training_days_per_week[user.experience_level]
        
        # Determine if training day
        training_days = self._get_training_schedule(weekly_frequency, user.consistency_factor)
        is_training_day = day_of_week in training_days
        
        # Phase modifications
        if phase == 'vacation':
            is_training_day = is_training_day and np.random.random() < 0.3  # 70% chance to skip
        elif phase == 'illness':
            is_training_day = False  # No training when sick
        
        if not is_training_day:
            return {
                'is_training_day': False,
                'total_sets': 0,
                'training_volume_load': 0,
                'average_rpe': 0,
                'session_duration_min': 0,
                'compound_sets': 0,
                'isolation_sets': 0,
                'effective_reps': 0,
                'time_under_tension_min': 0
            }
        
        # Base training volume (evidence-based) - FIXED: Higher volumes
        min_volume = self.MIN_EFFECTIVE_VOLUME[user.experience_level] + 4  # +4 sets minimum
        max_volume = self.MAX_RECOVERABLE_VOLUME[user.experience_level]
        
        # Distribute weekly volume across training days
        weekly_sets = np.random.uniform(min_volume, max_volume)
        daily_sets = weekly_sets / weekly_frequency
        
        # Add variation and consistency effects
        daily_variation = np.random.normal(1.0, 0.15)
        consistency_effect = np.random.normal(user.consistency_factor, 0.1)
        
        total_sets = daily_sets * daily_variation * consistency_effect
        total_sets = max(6, min(25, total_sets))  # FIXED: Higher minimum (6 instead of 3)
        
        # Phase modifications
        if phase == 'cutting':
            total_sets *= 0.9  # Slight volume reduction in deficit
        elif phase == 'vacation':
            total_sets *= 0.6  # Reduced volume on vacation
        
        # Distribute between compound and isolation
        compound_ratio = 0.65 if user.experience_level == ExperienceLevel.BEGINNER else 0.55
        compound_sets = int(total_sets * compound_ratio)
        isolation_sets = int(total_sets - compound_sets)
        
        # RPE - aim for effective stimulus (RPE 7-9)
        base_rpe = np.random.normal(7.5, 0.8)
        if phase == 'cutting':
            base_rpe += 0.5  # Harder when in deficit
        avg_rpe = max(6, min(9, base_rpe))
        
        # Effective reps (reps within 0-3 RIR)
        reps_per_set = np.random.normal(9, 2)
        effective_rep_percentage = min(0.9, avg_rpe / 10 * 0.8)
        effective_reps = total_sets * reps_per_set * effective_rep_percentage
        
        # Session duration
        base_duration = 45 + total_sets * 2.5
        session_duration = max(30, base_duration + np.random.normal(0, 10))
        
        # Time under tension
        time_under_tension = total_sets * np.random.normal(35, 8)
        
        # Volume load approximation
        bodyweight_multiplier = user.weight_start_kg / 80
        volume_load = total_sets * reps_per_set * bodyweight_multiplier * 100
        
        return {
            'is_training_day': True,
            'total_sets': round(total_sets, 1),
            'training_volume_load': round(volume_load),
            'average_rpe': round(avg_rpe, 1),
            'session_duration_min': round(session_duration),
            'compound_sets': compound_sets,
            'isolation_sets': isolation_sets,
            'effective_reps': round(effective_reps),
            'time_under_tension_min': round(time_under_tension / 60, 1)
        }
    
    def _get_training_schedule(self, frequency: int, consistency: float) -> List[int]:
        """Generate realistic training schedule for the week."""
        # Base schedules
        schedules = {
            3: [1, 3, 5],  # Mon, Wed, Fri
            4: [1, 2, 4, 5],  # Mon, Tue, Thu, Fri  
            5: [0, 1, 2, 4, 5],  # Mon-Wed, Fri-Sat
        }
        
        base_schedule = schedules.get(frequency, [1, 3, 5])
        
        # Apply consistency
        actual_schedule = []
        for day in base_schedule:
            if np.random.random() < consistency:
                actual_schedule.append(day)
        
        return actual_schedule
    
    def _generate_nutrition_day(self, user: UserProfile, day: int, week: int,
                               baseline_tdee: float, current_weight: float, phase: str) -> Dict:
        """Generate realistic nutrition data for one day."""
        # Calculate current TDEE
        current_tdee = calculate_maintenance_calories(
            weight_kg=current_weight,
            height_cm=user.height_cm,
            age_years=user.age,
            sex=user.sex,
            activity_level="moderately_active"
        )
        
        # Target calories based on phase
        if phase == 'cutting':
            # Moderate deficit
            target_calories = current_tdee - 400
        elif phase == 'building':
            # Small surplus or maintenance (surplus not required for muscle building)
            surplus = np.random.choice([0, 100, 200, 300], p=[0.3, 0.3, 0.3, 0.1])
            target_calories = current_tdee + surplus
        else:  # vacation, illness
            # Maintenance or slightly under
            target_calories = current_tdee + np.random.normal(-100, 150)
        
        # Apply consistency and weekly patterns
        weekend_effect = 1.15 if day % 7 in [5, 6] else 1.0
        consistency_variation = np.random.normal(user.consistency_factor, 0.08)
        
        actual_calories = target_calories * weekend_effect * consistency_variation
        actual_calories += np.random.normal(0, self.NOISE_PATTERNS['calories_daily_std'])
        
        # Protein intake (Morton et al. 2018: 1.6-2.2g/kg optimal)
        if phase == 'cutting':
            # Higher protein in deficit
            target_protein_g_per_kg = np.random.normal(2.2, 0.2)
        else:
            target_protein_g_per_kg = np.random.normal(2.0, 0.2)
        
        target_protein = current_weight * target_protein_g_per_kg
        actual_protein = target_protein * consistency_variation
        actual_protein += np.random.normal(0, self.NOISE_PATTERNS['protein_daily_std'])
        actual_protein = max(current_weight * 1.4, actual_protein)  # Minimum 1.4g/kg
        
        # Carbohydrate intake
        carb_g_per_kg = np.random.normal(4.0, 0.8)
        if phase == 'cutting':
            carb_g_per_kg *= 0.7  # Lower carbs in deficit
        carb_grams = current_weight * carb_g_per_kg
        
        # Fat intake (remaining calories)
        protein_kcal = actual_protein * 4
        carb_kcal = carb_grams * 4
        fat_kcal = max(300, actual_calories - protein_kcal - carb_kcal)
        fat_grams = fat_kcal / 9
        
        # Meal timing and frequency
        meals_per_day = np.random.choice([3, 4, 5], p=[0.3, 0.5, 0.2])
        
        # Hydration
        base_hydration = 35 * current_weight / 1000  # 35ml per kg
        hydration_liters = base_hydration + np.random.normal(0, 0.4)
        hydration_liters = max(1.5, min(5.0, hydration_liters))
        
        # Micronutrients
        vitamin_d_iu = np.random.normal(2000, 500)
        creatine_g = 5 if np.random.random() < 0.7 else 0
        
        return {
            'calories': round(actual_calories),
            'protein_g': round(actual_protein, 1),
            'carbs_g': round(carb_grams, 1),
            'fats_g': round(fat_grams, 1),
            'protein_g_per_kg': round(actual_protein / current_weight, 2),
            'fiber_g': round(carb_grams * 0.035, 1),
            'hydration_liters': round(hydration_liters, 1),
            'meals_per_day': meals_per_day,
            'vitamin_d_iu': round(vitamin_d_iu),
            'creatine_g': creatine_g,
            'calorie_surplus': round(actual_calories - current_tdee),
            'tdee_estimate': round(current_tdee)
        }
    
    def _generate_sleep_day(self, user: UserProfile, day: int, week: int) -> Dict:
        """Generate realistic sleep data for one day."""
        # Base sleep duration by age
        base_sleep_hours = 8.0 if user.age < 26 else 7.5
        
        # Weekly patterns
        day_of_week = day % 7
        if day_of_week in [4, 5]:  # Friday/Saturday - later bedtime
            sleep_duration = base_sleep_hours - 0.5
        elif day_of_week == 6:  # Sunday - catch up sleep
            sleep_duration = base_sleep_hours + 0.3
        else:
            sleep_duration = base_sleep_hours
        
        # Individual variation
        sleep_duration += np.random.normal(0, self.NOISE_PATTERNS['sleep_duration_std'])
        sleep_duration = max(4.0, min(11.0, sleep_duration))
        
        # Sleep quality (1-10 scale)
        base_quality = 7.5
        weekend_effect = 0.4 if day_of_week in [5, 6] else 0
        
        sleep_quality = base_quality + weekend_effect
        sleep_quality += np.random.normal(0, self.NOISE_PATTERNS['sleep_quality_std'])
        sleep_quality = max(1, min(10, sleep_quality))
        
        # Sleep architecture
        deep_sleep_percent = np.random.normal(22, 4)
        rem_sleep_percent = np.random.normal(23, 3)
        sleep_efficiency = min(98, np.random.normal(88, 6))
        sleep_latency = max(5, np.random.exponential(15))
        
        return {
            'sleep_duration_hours': round(sleep_duration, 1),
            'sleep_quality_1_10': round(sleep_quality, 1),
            'deep_sleep_percent': round(deep_sleep_percent, 1),
            'rem_sleep_percent': round(rem_sleep_percent, 1),
            'sleep_efficiency_percent': round(sleep_efficiency, 1),
            'sleep_latency_min': round(sleep_latency),
            'awakenings_count': np.random.poisson(1.5)
        }
    
    def _generate_body_metrics(self, user: UserProfile, day: int, week: int,
                              current_weight: float, current_muscle_mass: float,
                              training_data: Dict, nutrition_data: Dict, 
                              sleep_data: Dict, phase: str) -> Dict:
        """Generate body composition and physiological metrics."""
        # Muscle gain calculation
        daily_gain_rate = self._calculate_daily_muscle_gain(
            user, day, week, training_data, nutrition_data, sleep_data, phase
        )
        
        # Update muscle mass
        new_muscle_mass = current_muscle_mass + daily_gain_rate
        
        # Weight change
        muscle_weight_change = daily_gain_rate
        
        # Water retention effects
        water_change = 0
        if nutrition_data['creatine_g'] > 0:
            water_change += 0.02
        
        if training_data['is_training_day']:
            water_change += np.random.normal(0.1, 0.15)
        
        # Fat loss/gain based on phase
        if phase == 'cutting' and nutrition_data['calorie_surplus'] < -200:
            # Fat loss in deficit
            fat_change = -0.01  # ~70g fat loss per day in good deficit
        elif phase == 'building' and nutrition_data['calorie_surplus'] > 300:
            # Some fat gain in large surplus
            fat_change = 0.005  # ~35g fat gain per day
        else:
            fat_change = 0
        
        # Daily weight change
        weight_change = muscle_weight_change + water_change + fat_change
        new_weight = current_weight + weight_change + np.random.normal(0, self.NOISE_PATTERNS['weight_daily_std'])
        
        # Body fat percentage
        fat_mass = current_weight * (user.body_fat_start / 100) + (fat_change * 7 * (day / 7))
        new_body_fat_percent = (fat_mass / new_weight) * 100
        
        # HRV and recovery metrics
        base_hrv = 45
        training_effect = -training_data['total_sets'] * 0.3 if training_data['total_sets'] > 0 else 0
        sleep_effect = (sleep_data['sleep_quality_1_10'] - 7) * 2
        recovery_effect = (user.recovery_capacity - 1.0) * 10
        
        hrv = base_hrv + training_effect + sleep_effect + recovery_effect
        hrv += np.random.normal(0, self.NOISE_PATTERNS['hrv_daily_std'])
        hrv = max(15, min(80, hrv))
        
        # Resting heart rate
        resting_hr = 60 + (50 - hrv) * 0.3
        resting_hr = max(45, min(85, resting_hr))
        
        # Stress level
        base_stress = 4.0
        if phase == 'cutting':
            base_stress += 1.0  # More stress in deficit
        
        work_stress = np.random.normal(0, 1.0)
        training_stress = training_data['total_sets'] * 0.05 if training_data['total_sets'] > 0 else 0
        sleep_stress = max(0, (6 - sleep_data['sleep_quality_1_10']) * 0.5)
        
        perceived_stress = base_stress + work_stress + training_stress + sleep_stress
        perceived_stress = max(1, min(10, perceived_stress))
        
        return {
            'weight_kg': round(new_weight, 2),
            'estimated_muscle_mass_kg': round(new_muscle_mass, 2),
            'body_fat_percent': round(new_body_fat_percent, 1),
            'daily_muscle_gain_g': round(daily_gain_rate * 1000, 1),
            'hrv_rmssd': round(hrv, 1),
            'resting_heart_rate': round(resting_hr),
            'perceived_stress_1_10': round(perceived_stress, 1),
            'weekly_weight_trend_kg': round(daily_gain_rate * 7, 3)
        }
    
    def _calculate_daily_muscle_gain(self, user: UserProfile, day: int, week: int,
                                   training_data: Dict, nutrition_data: Dict,
                                   sleep_data: Dict, phase: str) -> float:
        """
        Calculate realistic daily muscle gain based on evidence.
        
        FIXED VERSION: Less harsh penalties, more generous for beginners
        """
        # Base gain rate
        baseline_daily = user.baseline_gain_rate_kg_week / 7
        
        # BEGINNER BONUS - newbie gains are real and significant!
        if user.experience_level == ExperienceLevel.BEGINNER:
            # First 6 months get substantial bonus, gradual decline
            beginner_bonus = max(1.5, 3.0 - (week / 20))  # 3x bonus first week, declining to 1.5x at 5 months
        else:
            beginner_bonus = 1.0
        
        # Training stimulus - more generous
        if training_data['total_sets'] == 0:
            training_multiplier = 0.3  # Still some gains without training (newbie bonus)
        else:
            # Check if above minimum effective volume
            weekly_sets = training_data['total_sets'] * 7 / 4  # Estimate weekly
            min_volume = self.MIN_EFFECTIVE_VOLUME[user.experience_level]
            
            if weekly_sets >= min_volume:
                # Above MEV = excellent stimulus
                volume_efficiency = min(1.5, weekly_sets / min_volume * 1.0)
            else:
                # Below MEV = still decent gains for beginners
                volume_efficiency = max(0.8, weekly_sets / min_volume)
            
            # RPE effect - more forgiving
            rpe_efficiency = max(0.8, min(1.3, (training_data['average_rpe'] - 4) / 4))
            
            training_multiplier = volume_efficiency * rpe_efficiency
        
        # Protein adequacy - more generous
        protein_per_kg = nutrition_data['protein_g_per_kg']
        if protein_per_kg >= 1.4:  # Lowered threshold
            protein_multiplier = min(1.3, 0.9 + (protein_per_kg - 1.4) * 0.2)
        else:
            protein_multiplier = max(0.7, protein_per_kg / 1.8)  # Less harsh penalty
        
        # Sleep effect - more forgiving
        sleep_quality = sleep_data['sleep_quality_1_10']
        sleep_duration = sleep_data['sleep_duration_hours']
        
        duration_efficiency = max(0.85, sleep_duration / 8)  # Less penalty
        quality_efficiency = max(0.85, sleep_quality / 10)   # Less penalty
        sleep_multiplier = (duration_efficiency + quality_efficiency) / 2
        
        # Caloric surplus effect - surplus helpful but not required
        surplus = nutrition_data['calorie_surplus']
        if surplus > 200:
            surplus_multiplier = min(1.2, 1.0 + surplus / 800)
        elif surplus < -300:  # Only big deficits hurt
            surplus_multiplier = max(0.7, 1.0 + surplus / 400)
        else:  # Maintenance to small surplus works great
            surplus_multiplier = 1.0
        
        # Phase effect - less harsh
        if phase == 'cutting':
            phase_multiplier = 0.8  # Only modest reduction
        elif phase in ['vacation', 'illness']:
            phase_multiplier = 0.4  # Some gains still possible
        else:
            phase_multiplier = 1.0
        
        # Individual factors
        genetic_multiplier = user.genetic_response
        recovery_multiplier = user.recovery_capacity
        
        # Progressive adaptation - much more forgiving for beginners
        if user.experience_level == ExperienceLevel.BEGINNER:
            adaptation_decay = max(0.95, 1 - week * 0.002)  # Very slow decline
        elif user.experience_level == ExperienceLevel.INTERMEDIATE:
            adaptation_decay = max(0.9, 1 - week * 0.005)
        else:  # Advanced
            adaptation_decay = max(0.85, 1 - week * 0.008)
        
        # Combine all factors - MUCH more generous
        total_multiplier = (training_multiplier * protein_multiplier * surplus_multiplier * 
                          sleep_multiplier * genetic_multiplier * recovery_multiplier * 
                          adaptation_decay * phase_multiplier * beginner_bonus)
        
        # Apply consistency - more forgiving
        consistency_effect = max(0.9, user.consistency_factor)  # High minimum
        total_multiplier *= consistency_effect
        
        daily_gain = baseline_daily * total_multiplier
        
        # Add variation
        daily_gain *= np.random.normal(1.0, 0.15)
        
        # Higher minimum baseline - especially for beginners
        if training_data['total_sets'] > 0:
            if user.experience_level == ExperienceLevel.BEGINNER:
                daily_gain = max(baseline_daily * 0.6, daily_gain)  # Very generous minimum
            else:
                daily_gain = max(baseline_daily * 0.4, daily_gain)
        else:
            daily_gain = max(0, daily_gain)
        
        return daily_gain
    
    def _add_engineered_features(self, df: pd.DataFrame, user: UserProfile) -> pd.DataFrame:
        """Add derived features for ML model training."""
        # Rolling averages
        df['calories_7d_avg'] = df['calories'].rolling(7, min_periods=1).mean()
        df['protein_7d_avg'] = df['protein_g'].rolling(7, min_periods=1).mean()
        df['sleep_7d_avg'] = df['sleep_duration_hours'].rolling(7, min_periods=1).mean()
        df['training_volume_7d_avg'] = df['total_sets'].rolling(7, min_periods=1).mean()
        df['weight_7d_avg'] = df['weight_kg'].rolling(7, min_periods=1).mean()
        
        # Trends
        df['weight_trend_14d'] = df['weight_kg'].rolling(14, min_periods=7).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] * 7 if len(x) >= 2 else 0
        )
        
        # Cumulative metrics
        df['cumulative_training_volume'] = df['total_sets'].cumsum()
        df['training_consistency_7d'] = df['is_training_day'].rolling(7, min_periods=1).mean()
        
        # Recovery metrics
        df['recovery_score'] = (df['sleep_quality_1_10'] * df['hrv_rmssd'] / 100) / df['perceived_stress_1_10']
        df['training_stress_balance'] = df['total_sets'] / (df['sleep_duration_hours'] + 1)
        
        # Nutrition ratios
        df['protein_adequacy_ratio'] = df['protein_g_per_kg'] / 1.8
        df['calorie_surplus_per_kg'] = df['calorie_surplus'] / df['weight_kg']
        
        # Phase encoding
        phase_map = {'building': 1, 'cutting': 2, 'vacation': 3, 'illness': 4}
        df['phase_encoded'] = df['phase'].map(phase_map)
        
        # User characteristics
        df['age'] = user.age
        df['sex_encoded'] = 1 if user.sex == 'male' else 0
        df['training_age_years'] = user.training_age_years
        df['experience_level_encoded'] = {
            ExperienceLevel.BEGINNER: 1,
            ExperienceLevel.INTERMEDIATE: 2, 
            ExperienceLevel.ADVANCED: 3
        }[user.experience_level]
        
        # Target variables
        df['muscle_gain_kg_per_week'] = df['daily_muscle_gain_g'] / 1000 * 7
        df['weight_change_kg_per_week'] = df['weight_kg'].diff() * 7
        
        return df
    
    def generate_all_users_data(self, days: int = 90) -> pd.DataFrame:
        """Generate data for all users and combine into single dataset."""
        logger.info(f"Generating {days} days of data for {len(self.users)} users")
        
        all_data = []
        for user in self.users:
            user_data = self.generate_user_data(user, days)
            all_data.append(user_data)
        
        # Combine all users
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Add cross-user features
        combined_df['user_id'] = combined_df['user_name'].astype('category').cat.codes
        
        logger.info(f"Generated complete dataset: {len(combined_df)} rows, {len(combined_df.columns)} columns")
        
        return combined_df
    
    def save_data(self, df: pd.DataFrame, filepath: str):
        """Save generated data to CSV file."""
        df.to_csv(filepath, index=False)
        logger.info(f"Saved dataset to {filepath}")
        
        # Log data quality summary
        logger.info("Data Quality Summary:")
        logger.info(f"  Shape: {df.shape}")
        logger.info(f"  Missing values: {df.isnull().sum().sum()}")
        logger.info(f"  Date range: {df['date'].min()} to {df['date'].max()}")
        logger.info(f"  Users: {df['user_name'].unique().tolist()}")
        logger.info(f"  Average muscle gain per week: {df['muscle_gain_kg_per_week'].mean():.4f} kg")


if __name__ == "__main__":
    # Example usage
    generator = AdvancedDataGenerator()
    
    # Generate data for all users
    dataset = generator.generate_all_users_data(days=90)
    
    # Save to file
    generator.save_data(dataset, "ml/data/synthetic_hypertrophy_data.csv")
    
    # Display summary statistics
    print("\nDataset Summary:")
    print("=" * 50)
    print(f"Total samples: {len(dataset)}")
    print(f"Features: {len(dataset.columns)}")
    print(f"Users: {dataset['user_name'].nunique()}")
    print(f"Date range: {dataset['date'].min()} to {dataset['date'].max()}")
    
    print("\nMuscle Gain Statistics:")
    print(dataset.groupby('user_name')['muscle_gain_kg_per_week'].agg(['mean', 'std', 'min', 'max']))
    
    print("\nTraining Volume Statistics:")
    print(dataset.groupby('user_name')['total_sets'].agg(['mean', 'std']))
