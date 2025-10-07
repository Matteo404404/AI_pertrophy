"""
ML Configuration - Hyperparameters and Constants

Based on exercise science literature and model optimization
"""

from datetime import timedelta
import numpy as np

# ================================================================================
# MODEL HYPERPARAMETERS
# ================================================================================

# XGBoost Hyperparameters (optimized for small datasets)
XGBOOST_PARAMS = {
    'n_estimators': 100,        # Number of trees
    'max_depth': 6,             # Tree depth (prevent overfitting)
    'learning_rate': 0.1,       # Step size shrinkage
    'subsample': 0.8,           # Row sampling
    'colsample_bytree': 0.8,    # Feature sampling
    'random_state': 42,         # Reproducibility
    'n_jobs': -1,               # Use all CPU cores
}

# ARIMA Hyperparameters
ARIMA_PARAMS = {
    'seasonal': False,          # No seasonal patterns (yet)
    'max_p': 3,                 # Max autoregressive terms
    'max_d': 2,                 # Max differencing
    'max_q': 3,                 # Max moving average terms
    'information_criterion': 'aic',  # Model selection
}

# Ensemble Weights
ENSEMBLE_WEIGHTS = {
    'xgboost': 0.7,             # Primary model
    'arima': 0.3,               # Smoothing baseline
}

# ================================================================================
# FEATURE ENGINEERING CONSTANTS
# ================================================================================

# Rolling window sizes (days)
ROLLING_WINDOWS = {
    'short': 7,                 # Weekly averages
    'medium': 14,               # Bi-weekly trends  
    'long': 30,                 # Monthly patterns
}

# Feature scaling ranges
FEATURE_RANGES = {
    'weight_kg': (50, 150),
    'calories': (1200, 4000),
    'protein_g': (50, 300),
    'sleep_hours': (4, 12),
    'training_volume': (0, 50),  # Sets per week
}

# Missing data thresholds
MISSING_DATA_THRESHOLD = 0.3    # 30% missing = exclude feature

# ================================================================================
# PHYSIOLOGICAL CONSTANTS (Literature-Based)
# ================================================================================

# Muscle gain rates (kg per week) - realistic ranges
MUSCLE_GAIN_RATES = {
    'novice': {
        'male': (0.25, 0.5),       # 1-2 lbs/month
        'female': (0.125, 0.25),   # 0.5-1 lb/month
    },
    'intermediate': {
        'male': (0.125, 0.25),     # 0.5-1 lb/month
        'female': (0.06, 0.125),   # 0.25-0.5 lb/month
    },
    'advanced': {
        'male': (0.06, 0.125),     # 0.25-0.5 lb/month
        'female': (0.03, 0.06),    # 0.125-0.25 lb/month
    }
}

# Atrophy rates (McMaster et al. 2013)
ATROPHY_RATES = {
    'week_1': 0.0,              # No loss first week
    'week_2': 0.02,             # 2% loss
    'week_3': 0.04,             # 4% cumulative
    'week_4': 0.06,             # 6% cumulative
    'chronic': 0.01,            # 1% per week after month
}

# TDEE Multipliers (activity levels)
ACTIVITY_MULTIPLIERS = {
    'sedentary': 1.2,           # Desk job, no exercise
    'lightly_active': 1.375,    # Light exercise 1-3 days
    'moderately_active': 1.55,  # Moderate exercise 3-5 days
    'very_active': 1.725,       # Hard exercise 6-7 days
    'extra_active': 1.9,        # Very hard + physical job
}

# Optimal nutrition ranges (per kg bodyweight)
NUTRITION_RANGES = {
    'protein_g_per_kg': (1.6, 2.2),    # Morton et al. 2018
    'carbs_g_per_kg': (3, 7),          # Burke et al.
    'fats_g_per_kg': (0.5, 1.5),       # Hormone support
    'calories_surplus': (200, 500),     # Lean bulk range
}

# ================================================================================
# PREDICTION SETTINGS
# ================================================================================

# Prediction horizons
PREDICTION_WEEKS = [4, 8, 12]          # Standard timeframes

# Confidence intervals
CONFIDENCE_LEVELS = [0.8, 0.95]        # 80% and 95% CI

# Minimum training data (days)
MIN_TRAINING_DAYS = 30                 # Need 1 month minimum

# Update frequency  
MODEL_RETRAIN_DAYS = 14                # Retrain every 2 weeks

# ================================================================================
# SYNTHETIC DATA GENERATION
# ================================================================================

# Demo user profiles
DEMO_USERS = {
    'beginner': {
        'age': 22,
        'weight_start': 70,
        'training_age': 0.5,        # 6 months
        'gain_rate': 0.3,           # kg/week
        'consistency': 0.8,         # 80% adherence
    },
    'intermediate': {
        'age': 28,
        'weight_start': 80,
        'training_age': 3,          # 3 years
        'gain_rate': 0.15,          # kg/week
        'consistency': 0.9,         # 90% adherence
    },
    'advanced': {
        'age': 35,
        'weight_start': 85,
        'training_age': 8,          # 8 years
        'gain_rate': 0.08,          # kg/week
        'consistency': 0.95,        # 95% adherence
    }
}

# Noise parameters (realistic variation)
NOISE_LEVELS = {
    'weight_std': 0.3,              # kg daily variation
    'calories_std': 200,            # kcal variation
    'sleep_std': 0.5,               # hours variation
    'volume_std': 2,                # sets variation
}

# ================================================================================
# MODEL VALIDATION
# ================================================================================

# Cross-validation settings
CV_FOLDS = 5                        # Time series CV
TEST_SIZE = 0.2                     # 20% holdout

# Performance thresholds
MIN_R2_SCORE = 0.6                  # Minimum acceptable R²
MAX_RMSE_KG = 0.5                   # Max error (kg)

# ================================================================================
# LOGGING SETTINGS  
# ================================================================================

LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_FILE = 'ml/logs/ml_training.log'

# ================================================================================
# FILE PATHS
# ================================================================================

# Model persistence
MODEL_DIR = 'ml/trained_models/'
SCALER_DIR = 'ml/scalers/'

# Data directories
SYNTHETIC_DATA_DIR = 'ml/data/synthetic/'
REAL_DATA_CACHE_DIR = 'ml/data/cache/'

# Visualization output
CHART_OUTPUT_DIR = 'ml/charts/'

# ================================================================================
# FEATURE LISTS (for reference)
# ================================================================================

# Core features (always included)
CORE_FEATURES = [
    'weight_kg', 'calories', 'protein_g', 'sleep_hours',
    'training_volume', 'training_frequency', 'age', 'training_age'
]

# Optional features (if available)
OPTIONAL_FEATURES = [
    'carbs_g', 'fats_g', 'fiber_g', 'hydration_l',
    'sleep_quality', 'stress_level', 'hrv_score',
    'cardio_minutes', 'steps_daily'
]

# Derived features (calculated)
DERIVED_FEATURES = [
    'calorie_surplus', 'protein_per_kg', 'volume_per_week',
    'recovery_score', 'anabolic_index', 'fatigue_ratio'
]
