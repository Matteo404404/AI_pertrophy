"""Extract normalization statistics from training data."""

import pandas as pd
import json

# Load training data
df = pd.read_csv('ml/data/training_data.csv')

print("Extracting normalization statistics from training data...")
print(f"Dataset: {len(df)} records")

# Calculate mean and std for each feature
feature_columns = [
    'weight_kg', 'reps', 'rir', 'total_sets', 'total_volume',
    'age', 'weight_kg_user', 'height_cm', 'body_fat_pct',
    'assessment_score', 'training_literacy_index', 'load_management_score',
    'technique_score', 'recovery_knowledge',
    'calories', 'protein_g', 'carbs_g', 'fats_g', 'fiber_g', 'water_ml',
    'sleep_hours', 'sleep_quality', 'stress_level', 'days_since_last_session',
    'creatine', 'protein_powder', 'pre_workout', 'caffeine_mg',
    'soreness_level', 'fatigue_level', 'readiness_score', 'hrv',
    'resting_heart_rate', 'session_rpe', 'recovery_quality'
]


stats = {}

for col in feature_columns:
    if col in df.columns:
        stats[col] = {
            'mean': float(df[col].mean()),
            'std': float(df[col].std()),
            'min': float(df[col].min()),
            'max': float(df[col].max())
        }
        print(f"✅ {col:<30} mean={stats[col]['mean']:.2f}, std={stats[col]['std']:.2f}")
    else:
        print(f"⚠️  {col} not found in dataset")

# Save to JSON
output_path = 'ml/data/normalization_stats.json'
with open(output_path, 'w') as f:
    json.dump(stats, f, indent=2)

print(f"\n✅ Saved normalization stats to: {output_path}")
print(f"   Total features: {len(stats)}")
