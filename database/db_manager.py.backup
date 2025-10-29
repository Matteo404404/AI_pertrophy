"""
Scientific Hypertrophy Trainer - Enhanced Database Manager
SQLite database with workout system, body measurements, comprehensive tracking
"""

import sqlite3
import json
import os
from datetime import datetime, date
from pathlib import Path


class DatabaseManager:
    """Enhanced database manager with workout system and comprehensive tracking"""
    
    def __init__(self, db_path="data/users.db"):
        """Initialize database connection and create all tables"""
        self.db_path = db_path
        self.ensure_database_directory()
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.create_tables()
        self.create_demo_data()
        
    def ensure_database_directory(self):
        """Ensure the database directory exists"""
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir)
    
    def create_tables(self):
        """Create all database tables"""
        cursor = self.conn.cursor()
        
        # Users table - ENHANCED with height and body fat
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                experience_level TEXT NOT NULL,
                primary_goal TEXT DEFAULT 'hypertrophy',
                weight_kg REAL NOT NULL,
                height_cm REAL NOT NULL,
                body_fat_percentage REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Assessment system tables (unchanged)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS assessment_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                tier_level INTEGER NOT NULL,
                score INTEGER NOT NULL,
                total_questions INTEGER NOT NULL,
                percentage REAL NOT NULL,
                passed BOOLEAN NOT NULL,
                completion_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS assessment_answers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                assessment_id INTEGER NOT NULL,
                question_id TEXT NOT NULL,
                question_text TEXT NOT NULL,
                selected_answer TEXT NOT NULL,
                correct_answer TEXT NOT NULL,
                is_correct BOOLEAN NOT NULL,
                FOREIGN KEY (assessment_id) REFERENCES assessment_history(id)
            )
        """)
        
        # Enhanced diet tracking - COMPREHENSIVE
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS diet_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                entry_date DATE NOT NULL,
                total_calories INTEGER NOT NULL,
                protein_g REAL NOT NULL,
                carbs_g REAL NOT NULL,
                fat_g REAL NOT NULL,
                fiber_g REAL,
                sugar_g REAL,
                sodium_mg REAL,
                potassium_mg REAL,
                calcium_mg REAL,
                iron_mg REAL,
                vitamin_d_ug REAL,
                vitamin_c_mg REAL,
                b12_ug REAL,
                hydration_liters REAL,
                meals_count INTEGER,
                protein_per_kg REAL,
                meal_timing TEXT,
                pre_workout_carbs_g REAL,
                post_workout_carbs_g REAL,
                creatine_g REAL,
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        
        # Enhanced sleep tracking
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sleep_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                entry_date DATE NOT NULL,
                bedtime TIME,
                wake_time TIME,
                sleep_duration_hours REAL NOT NULL,
                sleep_quality INTEGER NOT NULL,
                time_to_fall_asleep_minutes INTEGER,
                awakenings_count INTEGER,
                deep_sleep_hours REAL,
                rem_sleep_hours REAL,
                caffeine_cutoff_time TIME,
                screen_time_before_bed_minutes INTEGER,
                room_temperature_celsius REAL,
                sleep_environment_rating INTEGER,
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        
        # Exercise database
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS exercises (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                muscle_group_primary TEXT NOT NULL,
                muscle_group_secondary TEXT,
                equipment TEXT,
                movement_type TEXT NOT NULL,
                rep_range_min INTEGER,
                rep_range_max INTEGER,
                suggested_rir INTEGER,
                rest_time_seconds INTEGER,
                is_unilateral BOOLEAN DEFAULT FALSE,
                limb_priority TEXT,
                resistance_profile TEXT,
                difficulty_level INTEGER DEFAULT 1,
                description TEXT,
                notes TEXT,
                created_by_user INTEGER,
                is_system_exercise BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (created_by_user) REFERENCES users(id)
            )
        """)
        
        # Workout templates
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS workout_templates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                description TEXT,
                workout_type TEXT NOT NULL,
                estimated_duration_minutes INTEGER,
                difficulty_level INTEGER DEFAULT 1,
                muscle_groups_targeted TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        
        # Exercises in workout templates
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS workout_template_exercises (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                workout_template_id INTEGER NOT NULL,
                exercise_id INTEGER NOT NULL,
                order_in_workout INTEGER NOT NULL,
                target_sets INTEGER,
                target_reps_min INTEGER,
                target_reps_max INTEGER,
                target_rir INTEGER,
                rest_seconds INTEGER,
                notes TEXT,
                FOREIGN KEY (workout_template_id) REFERENCES workout_templates(id),
                FOREIGN KEY (exercise_id) REFERENCES exercises(id)
            )
        """)
        
        # Workout sessions
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS workout_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                workout_template_id INTEGER,
                session_name TEXT NOT NULL,
                session_date DATE NOT NULL,
                start_time TIMESTAMP,
                end_time TIMESTAMP,
                duration_minutes INTEGER,
                overall_rating INTEGER,
                perceived_exertion INTEGER,
                session_quality INTEGER,
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id),
                FOREIGN KEY (workout_template_id) REFERENCES workout_templates(id)
            )
        """)
        
        # Exercise performances
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS exercise_performances (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                workout_session_id INTEGER NOT NULL,
                exercise_id INTEGER NOT NULL,
                set_number INTEGER NOT NULL,
                weight_kg REAL,
                reps_completed INTEGER,
                rir_actual INTEGER,
                rest_seconds INTEGER,
                tempo TEXT,
                notes TEXT,
                FOREIGN KEY (workout_session_id) REFERENCES workout_sessions(id),
                FOREIGN KEY (exercise_id) REFERENCES exercises(id)
            )
        """)
        
        # Body measurements
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS body_measurements (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                measurement_date DATE NOT NULL,
                weight_kg REAL,
                body_fat_percentage REAL,
                muscle_mass_kg REAL,
                waist_cm REAL,
                chest_cm REAL,
                arm_cm REAL,
                thigh_cm REAL,
                neck_cm REAL,
                forearm_cm REAL,
                calf_cm REAL,
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        
        self.conn.commit()
    
    def create_demo_data(self):
        """Create demo users and system exercises"""
        self.create_demo_users()
        self.create_system_exercises()
    
    def create_system_exercises(self):
        """Create scientifically-based exercise database with CORRECT biomechanics"""
        system_exercises = [
            # ==================== TRICEPS ====================
            # Long Head Focus (shoulder extended/neutral - arm at/behind body)
            {'name': 'Cable Tricep Extension (Arm At Side)', 'primary': 'Triceps Long Head', 'secondary': 'Triceps Lateral Head,Triceps Medial Head', 'equipment': 'Cable', 'position': 'Lengthened', 'unilateral': True, 'profile': 'Shoulder extended arm at side - true lengthened'},
            {'name': 'Cable Tricep Extension (Arm Behind)', 'primary': 'Triceps Long Head', 'secondary': 'Triceps Lateral Head,Triceps Medial Head', 'equipment': 'Cable', 'position': 'Lengthened', 'unilateral': True, 'profile': 'Shoulder extended arm behind body'},
            {'name': 'Lying Dumbbell Extension', 'primary': 'Triceps Long Head', 'secondary': 'Triceps Lateral Head,Triceps Medial Head', 'equipment': 'Dumbbells', 'position': 'Lengthened', 'unilateral': True, 'profile': 'Lying shoulder extended position'},
            {'name': 'Lying Barbell Extension (Skull Crusher)', 'primary': 'Triceps Long Head', 'secondary': 'Triceps Lateral Head,Triceps Medial Head', 'equipment': 'Barbell', 'position': 'Lengthened', 'unilateral': False, 'profile': 'Lying shoulder extended skull crusher'},
            
            # Overhead variations (PASSIVE INSUFFICIENCY - lateral/medial heads more active)
            {'name': 'Cable Overhead Extension', 'primary': 'Triceps Lateral Head,Triceps Medial Head', 'secondary': None, 'equipment': 'Cable', 'position': 'Full Range', 'unilateral': False, 'profile': 'Overhead passive insufficiency - long head limited'},
            {'name': 'Cable Overhead Extension', 'primary': 'Triceps Lateral Head,Triceps Medial Head', 'secondary': None, 'equipment': 'Cable', 'position': 'Full Range', 'unilateral': True, 'profile': 'Single arm overhead passive insufficiency'},
            {'name': 'Dumbbell Overhead Extension', 'primary': 'Triceps Lateral Head,Triceps Medial Head', 'secondary': None, 'equipment': 'Dumbbells', 'position': 'Full Range', 'unilateral': True, 'profile': 'Overhead passive insufficiency'},
            {'name': 'Dumbbell Overhead Extension (Both Arms)', 'primary': 'Triceps Lateral Head,Triceps Medial Head', 'secondary': None, 'equipment': 'Dumbbells', 'position': 'Full Range', 'unilateral': False, 'profile': 'Bilateral overhead passive insufficiency'},
            {'name': 'Barbell Overhead Extension', 'primary': 'Triceps Lateral Head,Triceps Medial Head', 'secondary': None, 'equipment': 'Barbell', 'position': 'Full Range', 'unilateral': False, 'profile': 'Barbell overhead passive insufficiency'},
            
            # Lateral & Medial Head Focus
            {'name': 'Cable Pushdown (Straight Bar)', 'primary': 'Triceps Lateral Head,Triceps Medial Head', 'secondary': None, 'equipment': 'Cable', 'position': 'Full Range', 'unilateral': False, 'profile': 'Standard pushdown'},
            {'name': 'Cable Pushdown (Single Arm)', 'primary': 'Triceps Lateral Head,Triceps Medial Head', 'secondary': None, 'equipment': 'Cable', 'position': 'Full Range', 'unilateral': True, 'profile': 'Unilateral pushdown'},
            {'name': 'Cable Pushdown (Rope)', 'primary': 'Triceps Lateral Head,Triceps Medial Head', 'secondary': None, 'equipment': 'Cable', 'position': 'Full Range', 'unilateral': False, 'profile': 'Rope pushdown'},
            {'name': 'JM Press', 'primary': 'Triceps Lateral Head,Triceps Medial Head', 'secondary': None, 'equipment': 'Barbell', 'position': 'Full Range', 'unilateral': False, 'profile': 'Compound elbow extension'},
            {'name': 'Close Grip Bench Press', 'primary': 'Triceps Lateral Head,Triceps Medial Head', 'secondary': 'Mid Chest', 'equipment': 'Barbell', 'position': 'Full Range', 'unilateral': False, 'profile': 'Compound press tricep emphasis'},
            {'name': 'Quinton Press', 'primary': 'Triceps Lateral Head,Triceps Medial Head', 'secondary': None, 'equipment': 'Machine', 'position': 'Shortened', 'unilateral': True, 'profile': 'Shortened peak machine'},
            {'name': 'Dips', 'primary': 'Triceps Lateral Head,Triceps Medial Head', 'secondary': 'Lower Chest', 'equipment': 'Bodyweight', 'position': 'Shortened', 'unilateral': False, 'profile': 'Bodyweight compound shortened'},
            {'name': 'Weighted Dips', 'primary': 'Triceps Lateral Head,Triceps Medial Head', 'secondary': 'Lower Chest', 'equipment': 'Bodyweight', 'position': 'Shortened', 'unilateral': False, 'profile': 'Loaded compound shortened'},
            {'name': 'Dumbbell Kickback', 'primary': 'Triceps Lateral Head,Triceps Medial Head', 'secondary': None, 'equipment': 'Dumbbells', 'position': 'Shortened', 'unilateral': True, 'profile': 'Shortened peak contraction'},
            {'name': 'Cable Kickback', 'primary': 'Triceps Lateral Head,Triceps Medial Head', 'secondary': None, 'equipment': 'Cable', 'position': 'Shortened', 'unilateral': True, 'profile': 'Shortened peak constant tension'},

            # ==================== SHOULDERS ====================
            # Front Delts
            {'name': 'Barbell Overhead Press', 'primary': 'Front Delts', 'secondary': 'Triceps Lateral Head,Triceps Medial Head,Mid Delts', 'equipment': 'Barbell', 'position': 'Full Range', 'unilateral': False, 'profile': 'Overhead press'},
            {'name': 'Dumbbell Shoulder Press', 'primary': 'Front Delts', 'secondary': 'Triceps Lateral Head,Triceps Medial Head,Mid Delts', 'equipment': 'Dumbbells', 'position': 'Full Range', 'unilateral': True, 'profile': 'Dumbbell overhead press'},
            {'name': 'Machine Shoulder Press', 'primary': 'Front Delts', 'secondary': 'Triceps Lateral Head,Triceps Medial Head', 'equipment': 'Machine', 'position': 'Full Range', 'unilateral': True, 'profile': 'Machine press'},
            {'name': 'Cable Top Half Press', 'primary': 'Front Delts', 'secondary': 'Triceps Lateral Head,Triceps Medial Head', 'equipment': 'Cable', 'position': 'Shortened', 'unilateral': True, 'profile': 'Top range emphasis'},
            {'name': 'Cable Front Raises', 'primary': 'Front Delts', 'secondary': None, 'equipment': 'Cable', 'position': 'Lengthened', 'unilateral': True, 'profile': 'Front raise constant tension'},
            {'name': 'Dumbbell Front Raises', 'primary': 'Front Delts', 'secondary': None, 'equipment': 'Dumbbells', 'position': 'Full Range', 'unilateral': True, 'profile': 'Dumbbell front raise'},

            # Mid Delts
            {'name': 'Cable Y-Raise', 'primary': 'Mid Delts', 'secondary': None, 'equipment': 'Cable', 'position': 'Lengthened', 'unilateral': True, 'profile': 'Y-position lengthened'},
            {'name': 'Cable Lateral Raises', 'primary': 'Mid Delts', 'secondary': None, 'equipment': 'Cable', 'position': 'Lengthened', 'unilateral': True, 'profile': 'Lateral constant tension'},
            {'name': 'Dumbbell Lateral Raises', 'primary': 'Mid Delts', 'secondary': None, 'equipment': 'Dumbbells', 'position': 'Lengthened', 'unilateral': True, 'profile': 'Classic lateral raise'},
            {'name': 'Machine Lateral Raises', 'primary': 'Mid Delts', 'secondary': None, 'equipment': 'Machine', 'position': 'Full Range', 'unilateral': False, 'profile': 'Machine guided'},

            # Rear Delts
            {'name': 'Machine Reverse Fly', 'primary': 'Rear Delts', 'secondary': None, 'equipment': 'Machine', 'position': 'Full Range', 'unilateral': False, 'profile': 'Machine rear delt'},
            {'name': 'Cable Reverse Fly', 'primary': 'Rear Delts', 'secondary': None, 'equipment': 'Cable', 'position': 'Full Range', 'unilateral': True, 'profile': 'Cable rear delt'},
            {'name': 'Dumbbell Reverse Fly', 'primary': 'Rear Delts', 'secondary': None, 'equipment': 'Dumbbells', 'position': 'Full Range', 'unilateral': True, 'profile': 'Bent over fly'},
            {'name': 'Cable Face Pull', 'primary': 'Rear Delts', 'secondary': 'Mid Traps', 'equipment': 'Cable', 'position': 'Full Range', 'unilateral': False, 'profile': 'Face pull'},

            # ==================== CHEST ====================
            # Upper Chest
            {'name': 'Incline Barbell Press', 'primary': 'Upper Chest', 'secondary': 'Front Delts,Triceps Lateral Head,Triceps Medial Head', 'equipment': 'Barbell', 'position': 'Full Range', 'unilateral': False, 'profile': 'Incline press'},
            {'name': 'Incline Dumbbell Press', 'primary': 'Upper Chest', 'secondary': 'Front Delts,Triceps Lateral Head,Triceps Medial Head', 'equipment': 'Dumbbells', 'position': 'Full Range', 'unilateral': True, 'profile': 'Incline dumbbell'},
            {'name': 'Machine Incline Press', 'primary': 'Upper Chest', 'secondary': 'Front Delts,Triceps Lateral Head,Triceps Medial Head', 'equipment': 'Machine', 'position': 'Full Range', 'unilateral': True, 'profile': 'Machine incline'},
            {'name': 'Incline Dumbbell Fly', 'primary': 'Upper Chest', 'secondary': None, 'equipment': 'Dumbbells', 'position': 'Lengthened', 'unilateral': True, 'profile': 'Incline fly stretch'},
            {'name': 'Incline Cable Fly', 'primary': 'Upper Chest', 'secondary': None, 'equipment': 'Cable', 'position': 'Lengthened', 'unilateral': True, 'profile': 'Incline cable fly'},

            # Mid Chest
            {'name': 'Barbell Bench Press', 'primary': 'Mid Chest', 'secondary': 'Triceps Lateral Head,Triceps Medial Head,Front Delts', 'equipment': 'Barbell', 'position': 'Full Range', 'unilateral': False, 'profile': 'Flat bench press'},
            {'name': 'Dumbbell Bench Press', 'primary': 'Mid Chest', 'secondary': 'Triceps Lateral Head,Triceps Medial Head,Front Delts', 'equipment': 'Dumbbells', 'position': 'Full Range', 'unilateral': True, 'profile': 'Flat dumbbell'},
            {'name': 'Machine Chest Press', 'primary': 'Mid Chest', 'secondary': 'Triceps Lateral Head,Triceps Medial Head', 'equipment': 'Machine', 'position': 'Full Range', 'unilateral': True, 'profile': 'Machine press'},
            {'name': 'Dumbbell Fly', 'primary': 'Mid Chest', 'secondary': None, 'equipment': 'Dumbbells', 'position': 'Lengthened', 'unilateral': True, 'profile': 'Flat fly'},
            {'name': 'Cable Fly', 'primary': 'Mid Chest', 'secondary': None, 'equipment': 'Cable', 'position': 'Lengthened', 'unilateral': True, 'profile': 'Cable mid fly'},
            {'name': 'Push-Ups', 'primary': 'Mid Chest', 'secondary': 'Triceps Lateral Head,Triceps Medial Head', 'equipment': 'Bodyweight', 'position': 'Full Range', 'unilateral': False, 'profile': 'Bodyweight push-up'},

            # Lower Chest
            {'name': 'Decline Barbell Press', 'primary': 'Lower Chest', 'secondary': 'Triceps Lateral Head,Triceps Medial Head', 'equipment': 'Barbell', 'position': 'Full Range', 'unilateral': False, 'profile': 'Decline press'},
            {'name': 'Decline Dumbbell Press', 'primary': 'Lower Chest', 'secondary': 'Triceps Lateral Head,Triceps Medial Head', 'equipment': 'Dumbbells', 'position': 'Full Range', 'unilateral': True, 'profile': 'Decline dumbbell'},
            {'name': 'Cable Low Fly', 'primary': 'Lower Chest', 'secondary': None, 'equipment': 'Cable', 'position': 'Lengthened', 'unilateral': True, 'profile': 'Low cable fly'},
            {'name': 'Dips (Chest Focus)', 'primary': 'Lower Chest', 'secondary': 'Triceps Lateral Head,Triceps Medial Head', 'equipment': 'Bodyweight', 'position': 'Lengthened', 'unilateral': False, 'profile': 'Forward lean chest dips'},

            # ==================== BICEPS ====================
            # Long Head (shoulder extended/behind body)
            {'name': 'Incline Dumbbell Curl', 'primary': 'Biceps Long Head', 'secondary': 'Biceps Short Head', 'equipment': 'Dumbbells', 'position': 'Lengthened', 'unilateral': True, 'profile': 'Shoulder extended incline - true lengthened'},
            {'name': 'Cable Curl (Behind Body)', 'primary': 'Biceps Long Head', 'secondary': 'Biceps Short Head', 'equipment': 'Cable', 'position': 'Lengthened', 'unilateral': True, 'profile': 'Cable behind body shoulder extended'},
            
            # Short Head (shoulder flexed/in front)
            {'name': 'Preacher Curl (Barbell)', 'primary': 'Biceps Short Head', 'secondary': 'Biceps Long Head', 'equipment': 'Barbell', 'position': 'Lengthened', 'unilateral': False, 'profile': 'Shoulder flexed preacher'},
            {'name': 'Preacher Curl (Dumbbell)', 'primary': 'Biceps Short Head', 'secondary': 'Biceps Long Head', 'equipment': 'Dumbbells', 'position': 'Lengthened', 'unilateral': True, 'profile': 'Dumbbell preacher'},
            {'name': 'Spider Curl', 'primary': 'Biceps Short Head', 'secondary': 'Biceps Long Head', 'equipment': 'Dumbbells', 'position': 'Lengthened', 'unilateral': True, 'profile': 'Spider curl vertical arm'},
            {'name': 'Concentration Curl', 'primary': 'Biceps Short Head', 'secondary': 'Biceps Long Head', 'equipment': 'Dumbbells', 'position': 'Full Range', 'unilateral': True, 'profile': 'Seated concentration'},
            
            # Neutral/Both Heads
            {'name': 'Standing Barbell Curl', 'primary': 'Biceps Long Head,Biceps Short Head', 'secondary': None, 'equipment': 'Barbell', 'position': 'Full Range', 'unilateral': False, 'profile': 'Standard barbell curl'},
            {'name': 'Standing Dumbbell Curl', 'primary': 'Biceps Long Head,Biceps Short Head', 'secondary': None, 'equipment': 'Dumbbells', 'position': 'Full Range', 'unilateral': True, 'profile': 'Standing dumbbell'},
            {'name': 'Cable Curl', 'primary': 'Biceps Long Head,Biceps Short Head', 'secondary': None, 'equipment': 'Cable', 'position': 'Full Range', 'unilateral': True, 'profile': 'Cable curl'},
            {'name': 'EZ Bar Curl', 'primary': 'Biceps Long Head,Biceps Short Head', 'secondary': None, 'equipment': 'Barbell', 'position': 'Full Range', 'unilateral': False, 'profile': 'EZ bar curl'},
            
            # Brachialis
            {'name': 'Hammer Curl', 'primary': 'Brachialis', 'secondary': 'Biceps Long Head,Biceps Short Head', 'equipment': 'Dumbbells', 'position': 'Full Range', 'unilateral': True, 'profile': 'Neutral grip hammer'},
            {'name': 'Cable Hammer Curl', 'primary': 'Brachialis', 'secondary': 'Biceps Long Head,Biceps Short Head', 'equipment': 'Cable', 'position': 'Full Range', 'unilateral': True, 'profile': 'Cable rope hammer'},
            {'name': 'Cable Top Half Curl', 'primary': 'Brachialis', 'secondary': 'Biceps Long Head,Biceps Short Head', 'equipment': 'Cable', 'position': 'Shortened', 'unilateral': True, 'profile': 'Top ROM brachialis focus'},
            {'name': 'Reverse Curl', 'primary': 'Brachialis', 'secondary': 'Forearm Extensors', 'equipment': 'Barbell', 'position': 'Full Range', 'unilateral': False, 'profile': 'Pronated grip reverse'},

            # ==================== FOREARMS ====================
            {'name': 'Dumbbell Wrist Curl', 'primary': 'Forearm Flexors', 'secondary': None, 'equipment': 'Dumbbells', 'position': 'Full Range', 'unilateral': True, 'profile': 'Wrist flexion'},
            {'name': 'Barbell Wrist Curl', 'primary': 'Forearm Flexors', 'secondary': None, 'equipment': 'Barbell', 'position': 'Full Range', 'unilateral': False, 'profile': 'Barbell wrist flexion'},
            {'name': 'Cable Wrist Curl', 'primary': 'Forearm Flexors', 'secondary': None, 'equipment': 'Cable', 'position': 'Full Range', 'unilateral': True, 'profile': 'Cable wrist flexion'},
            {'name': 'Dumbbell Wrist Extension', 'primary': 'Forearm Extensors', 'secondary': None, 'equipment': 'Dumbbells', 'position': 'Full Range', 'unilateral': True, 'profile': 'Wrist extension'},
            {'name': 'Barbell Wrist Extension', 'primary': 'Forearm Extensors', 'secondary': None, 'equipment': 'Barbell', 'position': 'Full Range', 'unilateral': False, 'profile': 'Barbell extension'},
            {'name': 'Farmers Walk', 'primary': 'Forearm Flexors', 'secondary': 'Upper Traps', 'equipment': 'Dumbbells', 'position': 'Full Range', 'unilateral': False, 'profile': 'Loaded carry grip'},

            # ==================== CORE ====================
            {'name': 'Machine Crunch', 'primary': 'Rectus Abdominis', 'secondary': None, 'equipment': 'Machine', 'position': 'Full Range', 'unilateral': False, 'profile': 'Loaded spinal flexion'},
            {'name': 'Cable Crunch', 'primary': 'Rectus Abdominis', 'secondary': None, 'equipment': 'Cable', 'position': 'Full Range', 'unilateral': False, 'profile': 'Kneeling cable crunch'},
            {'name': 'Hanging Leg Raise', 'primary': 'Rectus Abdominis', 'secondary': 'Hip Flexors', 'equipment': 'Bodyweight', 'position': 'Full Range', 'unilateral': False, 'profile': 'Hanging leg raise'},
            {'name': 'Ab Wheel Rollout', 'primary': 'Rectus Abdominis', 'secondary': None, 'equipment': 'Ab Wheel', 'position': 'Lengthened', 'unilateral': False, 'profile': 'Anti-extension'},
            {'name': 'Cable Oblique Twist', 'primary': 'Internal Obliques,External Obliques', 'secondary': None, 'equipment': 'Cable', 'position': 'Full Range', 'unilateral': True, 'profile': 'Rotational'},
            {'name': 'Machine Oblique Crunch', 'primary': 'Internal Obliques,External Obliques', 'secondary': None, 'equipment': 'Machine', 'position': 'Full Range', 'unilateral': True, 'profile': 'Side flexion'},
            {'name': 'Pallof Press', 'primary': 'Internal Obliques,External Obliques', 'secondary': 'Rectus Abdominis', 'equipment': 'Cable', 'position': 'Full Range', 'unilateral': False, 'profile': 'Anti-rotation'},

            # ==================== BACK ====================
            # Lats
            {'name': 'Pull-Ups', 'primary': 'Lats', 'secondary': 'Biceps Long Head,Biceps Short Head', 'equipment': 'Bodyweight', 'position': 'Lengthened', 'unilateral': False, 'profile': 'Vertical pull bodyweight'},
            {'name': 'Weighted Pull-Ups', 'primary': 'Lats', 'secondary': 'Biceps Long Head,Biceps Short Head', 'equipment': 'Bodyweight', 'position': 'Lengthened', 'unilateral': False, 'profile': 'Weighted vertical pull'},
            {'name': 'Lat Pulldown (Wide)', 'primary': 'Lats', 'secondary': 'Biceps Long Head,Biceps Short Head', 'equipment': 'Cable', 'position': 'Lengthened', 'unilateral': False, 'profile': 'Wide grip pulldown'},
            {'name': 'Lat Pulldown (Close)', 'primary': 'Lats', 'secondary': 'Biceps Long Head,Biceps Short Head', 'equipment': 'Cable', 'position': 'Lengthened', 'unilateral': False, 'profile': 'Close grip pulldown'},
            {'name': 'Lat Pulldown (Unilateral)', 'primary': 'Lats', 'secondary': 'Biceps Long Head,Biceps Short Head', 'equipment': 'Cable', 'position': 'Lengthened', 'unilateral': True, 'profile': 'Single arm pulldown'},
            {'name': 'Cable Pullover', 'primary': 'Lats', 'secondary': None, 'equipment': 'Cable', 'position': 'Lengthened', 'unilateral': False, 'profile': 'Straight arm pullover'},
            {'name': 'Dumbbell Pullover', 'primary': 'Lats', 'secondary': None, 'equipment': 'Dumbbells', 'position': 'Lengthened', 'unilateral': False, 'profile': 'Dumbbell pullover'},
            
            # Mid Back / Rhomboids
            {'name': 'Barbell Row', 'primary': 'Mid Back,Rhomboids', 'secondary': 'Biceps Long Head,Biceps Short Head,Rear Delts', 'equipment': 'Barbell', 'position': 'Full Range', 'unilateral': False, 'profile': 'Bent over row'},
            {'name': 'Dumbbell Row', 'primary': 'Mid Back,Rhomboids', 'secondary': 'Biceps Long Head,Biceps Short Head,Rear Delts', 'equipment': 'Dumbbells', 'position': 'Full Range', 'unilateral': True, 'profile': 'Single arm row'},
            {'name': 'Cable Row', 'primary': 'Mid Back,Rhomboids', 'secondary': 'Biceps Long Head,Biceps Short Head', 'equipment': 'Cable', 'position': 'Full Range', 'unilateral': True, 'profile': 'Cable row'},
            {'name': 'T-Bar Row', 'primary': 'Mid Back,Rhomboids', 'secondary': 'Biceps Long Head,Biceps Short Head', 'equipment': 'Barbell', 'position': 'Full Range', 'unilateral': False, 'profile': 'T-bar row'},
            {'name': 'Machine Row', 'primary': 'Mid Back,Rhomboids', 'secondary': 'Biceps Long Head,Biceps Short Head', 'equipment': 'Machine', 'position': 'Full Range', 'unilateral': True, 'profile': 'Chest supported row'},
            {'name': 'Seal Row', 'primary': 'Mid Back,Rhomboids', 'secondary': 'Biceps Long Head,Biceps Short Head', 'equipment': 'Barbell', 'position': 'Full Range', 'unilateral': False, 'profile': 'Prone seal row'},

            # Traps
            {'name': 'Barbell Shrug', 'primary': 'Upper Traps', 'secondary': None, 'equipment': 'Barbell', 'position': 'Full Range', 'unilateral': False, 'profile': 'Scapular elevation'},
            {'name': 'Dumbbell Shrug', 'primary': 'Upper Traps', 'secondary': None, 'equipment': 'Dumbbells', 'position': 'Full Range', 'unilateral': True, 'profile': 'Dumbbell shrug'},
            {'name': 'Cable Horizontal Shrug', 'primary': 'Mid Traps', 'secondary': 'Rhomboids', 'equipment': 'Cable', 'position': 'Full Range', 'unilateral': True, 'profile': 'Horizontal retraction'},
            {'name': 'Cable Scapular Depression', 'primary': 'Lower Traps', 'secondary': None, 'equipment': 'Cable', 'position': 'Full Range', 'unilateral': True, 'profile': 'Scapular depression'},

            # Erectors & Lower Back (CORRECTED)
            {'name': 'Barbell Deadlift', 'primary': 'Erector Spinae', 'secondary': 'Glutes,Hamstrings', 'equipment': 'Barbell', 'position': 'Full Range', 'unilateral': False, 'profile': 'Conventional deadlift'},
            {'name': 'Romanian Deadlift', 'primary': 'Erector Spinae', 'secondary': 'Glutes', 'equipment': 'Barbell', 'position': 'Lengthened', 'unilateral': False, 'profile': 'Hip hinge - hamstring passive insufficiency'},
            {'name': 'Dumbbell Romanian Deadlift', 'primary': 'Erector Spinae', 'secondary': 'Glutes', 'equipment': 'Dumbbells', 'position': 'Lengthened', 'unilateral': True, 'profile': 'Unilateral RDL - hamstring passive insufficiency'},
            {'name': 'Barbell Stiff Leg Deadlift', 'primary': 'Erector Spinae', 'secondary': 'Glutes', 'equipment': 'Barbell', 'position': 'Lengthened', 'unilateral': False, 'profile': 'Stiff leg - hamstring passive insufficiency'},
            {'name': 'Back Extension', 'primary': 'Erector Spinae', 'secondary': 'Glutes,Hamstrings', 'equipment': 'Bodyweight', 'position': 'Full Range', 'unilateral': False, 'profile': 'Back extension'},
            {'name': 'Weighted Back Extension', 'primary': 'Erector Spinae', 'secondary': 'Glutes,Hamstrings', 'equipment': 'Bodyweight', 'position': 'Full Range', 'unilateral': False, 'profile': 'Loaded back extension'},
            {'name': 'Jefferson Curl', 'primary': 'Erector Spinae', 'secondary': None, 'equipment': 'Barbell', 'position': 'Lengthened', 'unilateral': False, 'profile': 'Spinal flexion stretch'},
            {'name': 'Good Morning', 'primary': 'Erector Spinae', 'secondary': 'Glutes', 'equipment': 'Barbell', 'position': 'Full Range', 'unilateral': False, 'profile': 'Hip hinge good morning'},

            # Serratus
            {'name': 'Cable Serratus Punch', 'primary': 'Serratus Anterior', 'secondary': None, 'equipment': 'Cable', 'position': 'Full Range', 'unilateral': True, 'profile': 'Scapular protraction'},

            # ==================== LEGS ====================
            # Quadriceps
            {'name': 'Leg Extension', 'primary': 'Quadriceps', 'secondary': None, 'equipment': 'Machine', 'position': 'Full Range', 'unilateral': True, 'profile': 'Knee extension isolation'},
            {'name': 'Barbell Back Squat', 'primary': 'Quadriceps', 'secondary': 'Glutes', 'equipment': 'Barbell', 'position': 'Full Range', 'unilateral': False, 'profile': 'Back squat'},
            {'name': 'Barbell Front Squat', 'primary': 'Quadriceps', 'secondary': 'Glutes', 'equipment': 'Barbell', 'position': 'Full Range', 'unilateral': False, 'profile': 'Front squat quad emphasis'},
            {'name': 'Hack Squat', 'primary': 'Quadriceps', 'secondary': 'Glutes', 'equipment': 'Machine', 'position': 'Full Range', 'unilateral': False, 'profile': 'Hack squat machine'},
            {'name': 'Leg Press', 'primary': 'Quadriceps', 'secondary': 'Glutes', 'equipment': 'Machine', 'position': 'Full Range', 'unilateral': False, 'profile': 'Leg press'},
            {'name': 'Leg Press (Unilateral)', 'primary': 'Quadriceps', 'secondary': 'Glutes', 'equipment': 'Machine', 'position': 'Full Range', 'unilateral': True, 'profile': 'Single leg press'},
            {'name': 'Bulgarian Split Squat', 'primary': 'Quadriceps', 'secondary': 'Glutes', 'equipment': 'Dumbbells', 'position': 'Lengthened', 'unilateral': True, 'profile': 'Rear elevated split squat'},
            {'name': 'Walking Lunges', 'primary': 'Quadriceps', 'secondary': 'Glutes', 'equipment': 'Dumbbells', 'position': 'Full Range', 'unilateral': True, 'profile': 'Walking lunge'},
            {'name': 'Sissy Squat', 'primary': 'Quadriceps', 'secondary': None, 'equipment': 'Bodyweight', 'position': 'Lengthened', 'unilateral': False, 'profile': 'Sissy squat lengthened quad'},

            # Hamstrings (CORRECTED - no lengthened seated leg curl)
            {'name': 'Lying Leg Curl', 'primary': 'Hamstrings', 'secondary': None, 'equipment': 'Machine', 'position': 'Full Range', 'unilateral': True, 'profile': 'Lying knee flexion - true lengthened at hip'},
            {'name': 'Seated Leg Curl', 'primary': 'Hamstrings', 'secondary': None, 'equipment': 'Machine', 'position': 'Full Range', 'unilateral': True, 'profile': 'Seated knee flexion - hip flexed position'},
            {'name': 'Nordic Curl', 'primary': 'Hamstrings', 'secondary': None, 'equipment': 'Bodyweight', 'position': 'Lengthened', 'unilateral': False, 'profile': 'Nordic eccentric - true lengthened'},
            {'name': 'Cable Pull-Through', 'primary': 'Hamstrings', 'secondary': 'Glutes', 'equipment': 'Cable', 'position': 'Full Range', 'unilateral': False, 'profile': 'Cable hip hinge'},

            # Glutes
            {'name': 'Barbell Hip Thrust', 'primary': 'Glutes', 'secondary': 'Hamstrings', 'equipment': 'Barbell', 'position': 'Shortened', 'unilateral': False, 'profile': 'Hip thrust shortened peak'},
            {'name': 'Single Leg Hip Thrust', 'primary': 'Glutes', 'secondary': 'Hamstrings', 'equipment': 'Bodyweight', 'position': 'Shortened', 'unilateral': True, 'profile': 'Unilateral hip thrust'},
            {'name': 'Glute Bridge', 'primary': 'Glutes', 'secondary': 'Hamstrings', 'equipment': 'Barbell', 'position': 'Shortened', 'unilateral': False, 'profile': 'Floor glute bridge'},
            {'name': 'Cable Glute Kickback', 'primary': 'Glutes', 'secondary': None, 'equipment': 'Cable', 'position': 'Shortened', 'unilateral': True, 'profile': 'Hip extension isolation'},
            {'name': 'Machine Glute Kickback', 'primary': 'Glutes', 'secondary': None, 'equipment': 'Machine', 'position': 'Full Range', 'unilateral': True, 'profile': 'Machine hip extension'},
            {'name': 'Machine Abduction', 'primary': 'Glute Medius', 'secondary': None, 'equipment': 'Machine', 'position': 'Full Range', 'unilateral': False, 'profile': 'Hip abduction'},
            {'name': 'Cable Abduction', 'primary': 'Glute Medius', 'secondary': None, 'equipment': 'Cable', 'position': 'Full Range', 'unilateral': True, 'profile': 'Cable abduction'},

            # Calves
            {'name': 'Standing Calf Raise', 'primary': 'Gastrocnemius', 'secondary': None, 'equipment': 'Machine', 'position': 'Full Range', 'unilateral': False, 'profile': 'Standing straight leg'},
            {'name': 'Standing Calf Raise (Unilateral)', 'primary': 'Gastrocnemius', 'secondary': None, 'equipment': 'Machine', 'position': 'Full Range', 'unilateral': True, 'profile': 'Single leg calf'},
            {'name': 'Seated Calf Raise', 'primary': 'Soleus', 'secondary': None, 'equipment': 'Machine', 'position': 'Full Range', 'unilateral': False, 'profile': 'Seated bent knee soleus'},
            {'name': 'Leg Press Calf Raise', 'primary': 'Gastrocnemius', 'secondary': None, 'equipment': 'Machine', 'position': 'Full Range', 'unilateral': False, 'profile': 'Leg press calf'},

            # Tibialis
            {'name': 'Tibialis Raise', 'primary': 'Tibialis Anterior', 'secondary': None, 'equipment': 'Bodyweight', 'position': 'Full Range', 'unilateral': False, 'profile': 'Dorsiflexion'},
            {'name': 'Weighted Tibialis Raise', 'primary': 'Tibialis Anterior', 'secondary': None, 'equipment': 'Plate', 'position': 'Full Range', 'unilateral': False, 'profile': 'Loaded dorsiflexion'},
        ]
        
        cursor = self.conn.cursor()
        for ex in system_exercises:
            try:
                cursor.execute("""
                    INSERT OR IGNORE INTO exercises 
                    (name, muscle_group_primary, muscle_group_secondary, equipment, 
                    is_unilateral, resistance_profile, is_system_exercise)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (ex['name'], ex['primary'], ex['secondary'], ex['equipment'],
                    ex['unilateral'], ex['profile'], True))
            except sqlite3.IntegrityError:
                pass
        
        self.conn.commit()
        print(f"âœ… Created {len(system_exercises)} biomechanically correct exercises")
        
        # ===== MISSING DATA RETRIEVAL METHODS =====
    
    def get_training_entries(self, user_id, days=30):
        """Get training entries (placeholder - using workout sessions)"""
        cursor = self.conn.cursor()
        sessions = cursor.execute("""
            SELECT ws.*, wt.name as template_name 
            FROM workout_sessions ws
            LEFT JOIN workout_templates wt ON ws.workout_template_id = wt.id
            WHERE ws.user_id = ? AND ws.session_date >= date('now', '-' || ? || ' days')
            ORDER BY ws.session_date DESC, ws.start_time DESC
        """, (user_id, days)).fetchall()
        return [dict(session) for session in sessions]
        
    def create_demo_users(self):
        """Create demo users with enhanced attributes"""
        demo_users = [
            {
                'username': 'Alex_Beginner',
                'experience_level': 'beginner',
                'weight_kg': 75.0,
                'height_cm': 175.0,
                'body_fat_percentage': 15.0
            },
            {
                'username': 'Sarah_Intermediate',
                'experience_level': 'intermediate',
                'weight_kg': 65.0,
                'height_cm': 165.0,
                'body_fat_percentage': 18.0
            },
            {
                'username': 'Mike_Advanced',
                'experience_level': 'advanced',
                'weight_kg': 85.0,
                'height_cm': 180.0,
                'body_fat_percentage': 12.0
            }
        ]
        
        cursor = self.conn.cursor()
        
        for user_data in demo_users:
            try:
                cursor.execute("""
                    INSERT OR IGNORE INTO users 
                    (username, experience_level, weight_kg, height_cm, body_fat_percentage)
                    VALUES (?, ?, ?, ?, ?)
                """, (user_data['username'], user_data['experience_level'], 
                      user_data['weight_kg'], user_data['height_cm'], 
                      user_data['body_fat_percentage']))
            except sqlite3.IntegrityError:
                pass
                
        self.conn.commit()
        self.create_demo_assessments()
    
    def create_system_exercises(self):
        """Create comprehensive system exercise database"""
        system_exercises = [
            # CHEST
            {'name': 'Barbell Bench Press', 'primary': 'Chest', 'secondary': 'Triceps,Front Delts', 'equipment': 'Barbell', 'type': 'Compound', 'rep_min': 5, 'rep_max': 12, 'rir': 2, 'rest': 180, 'unilateral': False, 'profile': 'Flat', 'difficulty': 2},
            {'name': 'Dumbbell Bench Press', 'primary': 'Chest', 'secondary': 'Triceps,Front Delts', 'equipment': 'Dumbbells', 'type': 'Compound', 'rep_min': 6, 'rep_max': 15, 'rir': 2, 'rest': 180, 'unilateral': False, 'profile': 'Flat', 'difficulty': 2},
            {'name': 'Incline Barbell Press', 'primary': 'Chest', 'secondary': 'Front Delts,Triceps', 'equipment': 'Barbell', 'type': 'Compound', 'rep_min': 6, 'rep_max': 12, 'rir': 2, 'rest': 180, 'unilateral': False, 'profile': 'Incline', 'difficulty': 2},
            {'name': 'Dips', 'primary': 'Chest', 'secondary': 'Triceps', 'equipment': 'Bodyweight', 'type': 'Compound', 'rep_min': 8, 'rep_max': 20, 'rir': 2, 'rest': 120, 'unilateral': False, 'profile': 'Decline', 'difficulty': 3},
            {'name': 'Dumbbell Flyes', 'primary': 'Chest', 'secondary': None, 'equipment': 'Dumbbells', 'type': 'Isolation', 'rep_min': 10, 'rep_max': 20, 'rir': 1, 'rest': 90, 'unilateral': False, 'profile': 'Stretch', 'difficulty': 1},
            
            # BACK
            {'name': 'Deadlift', 'primary': 'Back', 'secondary': 'Glutes,Hamstrings', 'equipment': 'Barbell', 'type': 'Compound', 'rep_min': 3, 'rep_max': 8, 'rir': 2, 'rest': 240, 'unilateral': False, 'profile': 'Hip Hinge', 'difficulty': 4},
            {'name': 'Pull-ups', 'primary': 'Back', 'secondary': 'Biceps', 'equipment': 'Bodyweight', 'type': 'Compound', 'rep_min': 5, 'rep_max': 15, 'rir': 2, 'rest': 180, 'unilateral': False, 'profile': 'Stretch', 'difficulty': 3},
            {'name': 'Barbell Rows', 'primary': 'Back', 'secondary': 'Biceps,Rear Delts', 'equipment': 'Barbell', 'type': 'Compound', 'rep_min': 6, 'rep_max': 12, 'rir': 2, 'rest': 180, 'unilateral': False, 'profile': 'Mid-Range', 'difficulty': 2},
            {'name': 'Lat Pulldown', 'primary': 'Back', 'secondary': 'Biceps', 'equipment': 'Cable', 'type': 'Compound', 'rep_min': 8, 'rep_max': 15, 'rir': 2, 'rest': 120, 'unilateral': False, 'profile': 'Stretch', 'difficulty': 1},
            {'name': 'Cable Rows', 'primary': 'Back', 'secondary': 'Biceps,Rear Delts', 'equipment': 'Cable', 'type': 'Compound', 'rep_min': 8, 'rep_max': 15, 'rir': 2, 'rest': 120, 'unilateral': False, 'profile': 'Stretch', 'difficulty': 1},
            
            # SHOULDERS
            {'name': 'Overhead Press', 'primary': 'Shoulders', 'secondary': 'Triceps', 'equipment': 'Barbell', 'type': 'Compound', 'rep_min': 5, 'rep_max': 12, 'rir': 2, 'rest': 180, 'unilateral': False, 'profile': 'Stretch', 'difficulty': 3},
            {'name': 'Dumbbell Shoulder Press', 'primary': 'Shoulders', 'secondary': 'Triceps', 'equipment': 'Dumbbells', 'type': 'Compound', 'rep_min': 6, 'rep_max': 15, 'rir': 2, 'rest': 150, 'unilateral': False, 'profile': 'Stretch', 'difficulty': 2},
            {'name': 'Lateral Raises', 'primary': 'Shoulders', 'secondary': None, 'equipment': 'Dumbbells', 'type': 'Isolation', 'rep_min': 12, 'rep_max': 25, 'rir': 1, 'rest': 90, 'unilateral': False, 'profile': 'Length', 'difficulty': 1},
            {'name': 'Rear Delt Flyes', 'primary': 'Shoulders', 'secondary': None, 'equipment': 'Dumbbells', 'type': 'Isolation', 'rep_min': 12, 'rep_max': 20, 'rir': 1, 'rest': 90, 'unilateral': False, 'profile': 'Stretch', 'difficulty': 1},
            
            # LEGS
            {'name': 'Squat', 'primary': 'Quadriceps', 'secondary': 'Glutes', 'equipment': 'Barbell', 'type': 'Compound', 'rep_min': 5, 'rep_max': 15, 'rir': 2, 'rest': 240, 'unilateral': False, 'profile': 'Stretch', 'difficulty': 3},
            {'name': 'Romanian Deadlift', 'primary': 'Hamstrings', 'secondary': 'Glutes', 'equipment': 'Barbell', 'type': 'Compound', 'rep_min': 6, 'rep_max': 15, 'rir': 2, 'rest': 180, 'unilateral': False, 'profile': 'Stretch', 'difficulty': 2},
            {'name': 'Bulgarian Split Squat', 'primary': 'Quadriceps', 'secondary': 'Glutes', 'equipment': 'Dumbbells', 'type': 'Compound', 'rep_min': 8, 'rep_max': 20, 'rir': 2, 'rest': 120, 'unilateral': True, 'profile': 'Stretch', 'difficulty': 2},
            {'name': 'Leg Curls', 'primary': 'Hamstrings', 'secondary': None, 'equipment': 'Machine', 'type': 'Isolation', 'rep_min': 10, 'rep_max': 20, 'rir': 1, 'rest': 90, 'unilateral': False, 'profile': 'Stretch', 'difficulty': 1},
            {'name': 'Calf Raises', 'primary': 'Calves', 'secondary': None, 'equipment': 'Machine', 'type': 'Isolation', 'rep_min': 15, 'rep_max': 30, 'rir': 2, 'rest': 60, 'unilateral': False, 'profile': 'Stretch', 'difficulty': 1},
            
            # ARMS
            {'name': 'Barbell Curls', 'primary': 'Biceps', 'secondary': None, 'equipment': 'Barbell', 'type': 'Isolation', 'rep_min': 8, 'rep_max': 15, 'rir': 2, 'rest': 90, 'unilateral': False, 'profile': 'Mid-Range', 'difficulty': 1},
            {'name': 'Incline Dumbbell Curls', 'primary': 'Biceps', 'secondary': None, 'equipment': 'Dumbbells', 'type': 'Isolation', 'rep_min': 8, 'rep_max': 15, 'rir': 2, 'rest': 90, 'unilateral': False, 'profile': 'Stretch', 'difficulty': 2},
            {'name': 'Close Grip Bench Press', 'primary': 'Triceps', 'secondary': 'Chest', 'equipment': 'Barbell', 'type': 'Compound', 'rep_min': 6, 'rep_max': 15, 'rir': 2, 'rest': 150, 'unilateral': False, 'profile': 'Stretch', 'difficulty': 2},
            {'name': 'Overhead Tricep Extension', 'primary': 'Triceps', 'secondary': None, 'equipment': 'Dumbbells', 'type': 'Isolation', 'rep_min': 10, 'rep_max': 20, 'rir': 2, 'rest': 90, 'unilateral': False, 'profile': 'Stretch', 'difficulty': 1}
        ]
        
        cursor = self.conn.cursor()
        for ex in system_exercises:
            try:
                cursor.execute("""
                    INSERT OR IGNORE INTO exercises 
                    (name, muscle_group_primary, muscle_group_secondary, equipment, movement_type,
                     rep_range_min, rep_range_max, suggested_rir, rest_time_seconds, is_unilateral,
                     resistance_profile, difficulty_level, is_system_exercise)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (ex['name'], ex['primary'], ex['secondary'], ex['equipment'], ex['type'],
                      ex['rep_min'], ex['rep_max'], ex['rir'], ex['rest'], ex['unilateral'],
                      ex['profile'], ex['difficulty'], True))
            except sqlite3.IntegrityError:
                pass
        
        self.conn.commit()
        
    def create_demo_assessments(self):
        """Create demo assessment data"""
        cursor = self.conn.cursor()
        
        # Sarah's Tier 1 pass
        sarah_id = cursor.execute("SELECT id FROM users WHERE username = 'Sarah_Intermediate'").fetchone()
        if sarah_id:
            sarah_id = sarah_id[0]
            exists = cursor.execute("SELECT id FROM assessment_history WHERE user_id = ? AND tier_level = 0", (sarah_id,)).fetchone()
            if not exists:
                cursor.execute("""
                    INSERT INTO assessment_history (user_id, tier_level, score, total_questions, percentage, passed)
                    VALUES (?, 0, 16, 20, 80.0, 1)
                """, (sarah_id,))
        
        # Mike's Tier 1 and 2 passes
        mike_id = cursor.execute("SELECT id FROM users WHERE username = 'Mike_Advanced'").fetchone()
        if mike_id:
            mike_id = mike_id[0]
            tier1_exists = cursor.execute("SELECT id FROM assessment_history WHERE user_id = ? AND tier_level = 0", (mike_id,)).fetchone()
            if not tier1_exists:
                cursor.execute("""
                    INSERT INTO assessment_history (user_id, tier_level, score, total_questions, percentage, passed)
                    VALUES (?, 0, 19, 20, 95.0, 1)
                """, (mike_id,))
                
            tier2_exists = cursor.execute("SELECT id FROM assessment_history WHERE user_id = ? AND tier_level = 1", (mike_id,)).fetchone()
            if not tier2_exists:
                cursor.execute("""
                    INSERT INTO assessment_history (user_id, tier_level, score, total_questions, percentage, passed)
                    VALUES (?, 1, 17, 20, 85.0, 1)
                """, (mike_id,))
        
        self.conn.commit()

    # ===== USER MANAGEMENT =====
    
    def get_all_users(self):
        """Get all users with enhanced attributes"""
        cursor = self.conn.cursor()
        users = cursor.execute("SELECT * FROM users ORDER BY created_at DESC").fetchall()
        return [dict(user) for user in users]
    
    def get_user_by_id(self, user_id):
        """Get user by ID"""
        cursor = self.conn.cursor()
        user = cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        return dict(user) if user else None
    
    def create_user(self, username, experience_level, weight_kg, height_cm, body_fat_percentage):
        """Create new user with enhanced attributes"""
        cursor = self.conn.cursor()
        try:
            cursor.execute("""
                INSERT INTO users (username, experience_level, weight_kg, height_cm, body_fat_percentage)
                VALUES (?, ?, ?, ?, ?)
            """, (username, experience_level, weight_kg, height_cm, body_fat_percentage))
            self.conn.commit()
            return cursor.lastrowid
        except sqlite3.IntegrityError:
            raise ValueError(f"User '{username}' already exists")
    
    def delete_user(self, user_id):
        """Delete user and all associated data"""
        cursor = self.conn.cursor()
        
        # Delete in reverse dependency order
        cursor.execute("DELETE FROM exercise_performances WHERE workout_session_id IN (SELECT id FROM workout_sessions WHERE user_id = ?)", (user_id,))
        cursor.execute("DELETE FROM workout_sessions WHERE user_id = ?", (user_id,))
        cursor.execute("DELETE FROM workout_template_exercises WHERE workout_template_id IN (SELECT id FROM workout_templates WHERE user_id = ?)", (user_id,))
        cursor.execute("DELETE FROM workout_templates WHERE user_id = ?", (user_id,))
        cursor.execute("DELETE FROM body_measurements WHERE user_id = ?", (user_id,))
        cursor.execute("DELETE FROM diet_entries WHERE user_id = ?", (user_id,))
        cursor.execute("DELETE FROM sleep_entries WHERE user_id = ?", (user_id,))
        cursor.execute("DELETE FROM assessment_answers WHERE assessment_id IN (SELECT id FROM assessment_history WHERE user_id = ?)", (user_id,))
        cursor.execute("DELETE FROM assessment_history WHERE user_id = ?", (user_id,))
        cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
        
        self.conn.commit()
        return cursor.rowcount > 0

    def update_user_last_active(self, user_id):
        """Update user's last active timestamp"""
        cursor = self.conn.cursor()
        cursor.execute("UPDATE users SET last_active = CURRENT_TIMESTAMP WHERE id = ?", (user_id,))
        self.conn.commit()

    # ===== ASSESSMENT METHODS (unchanged) =====
    
    def save_assessment_result(self, user_id, tier_level, score, total_questions, percentage, passed, answers):
        """Save assessment result"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO assessment_history (user_id, tier_level, score, total_questions, percentage, passed)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (user_id, tier_level, score, total_questions, percentage, passed))
        
        assessment_id = cursor.lastrowid
        for answer in answers:
            cursor.execute("""
                INSERT INTO assessment_answers 
                (assessment_id, question_id, question_text, selected_answer, correct_answer, is_correct)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (assessment_id, answer['question_id'], answer['question_text'],
                  answer['selected_answer'], answer['correct_answer'], answer['is_correct']))
        
        self.conn.commit()
        return assessment_id
    
    def get_user_assessments(self, user_id):
        """Get user assessments"""
        cursor = self.conn.cursor()
        assessments = cursor.execute("""
            SELECT * FROM assessment_history WHERE user_id = ? ORDER BY completion_time DESC
        """, (user_id,)).fetchall()
        return [dict(assessment) for assessment in assessments]
    
    def get_assessment_answers(self, assessment_id):
        """Get assessment answers"""
        cursor = self.conn.cursor()
        answers = cursor.execute("SELECT * FROM assessment_answers WHERE assessment_id = ?", (assessment_id,)).fetchall()
        return [dict(answer) for answer in answers]
    
    def get_user_tier_progress(self, user_id):
        """Calculate tier progression"""
        assessments = self.get_user_assessments(user_id)
        progress = {
            'current_tier': 0, 'tier_1_passed': False, 'tier_2_passed': False, 'tier_3_passed': False,
            'tier_2_unlocked': False, 'tier_3_unlocked': False
        }
        
        for assessment in assessments:
            if assessment['tier_level'] == 0 and assessment['passed']:
                progress['tier_1_passed'] = True
                progress['tier_2_unlocked'] = True
                progress['current_tier'] = max(1, progress['current_tier'])
            elif assessment['tier_level'] == 1 and assessment['passed']:
                progress['tier_2_passed'] = True
                progress['tier_3_unlocked'] = True
                progress['current_tier'] = max(2, progress['current_tier'])
            elif assessment['tier_level'] == 2 and assessment['passed']:
                progress['tier_3_passed'] = True
        
        return progress
    
    # ===== ENHANCED TRACKING METHODS =====
    
    def save_diet_entry(self, user_id, entry_date, **kwargs):
        """Save comprehensive diet entry"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO diet_entries 
            (user_id, entry_date, total_calories, protein_g, carbs_g, fat_g, fiber_g, sugar_g,
             sodium_mg, potassium_mg, calcium_mg, iron_mg, vitamin_d_ug, vitamin_c_mg, b12_ug,
             hydration_liters, meals_count, protein_per_kg, meal_timing, pre_workout_carbs_g,
             post_workout_carbs_g, creatine_g, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (user_id, entry_date,
              kwargs.get('total_calories', 0), kwargs.get('protein_g', 0), kwargs.get('carbs_g', 0),
              kwargs.get('fat_g', 0), kwargs.get('fiber_g'), kwargs.get('sugar_g'),
              kwargs.get('sodium_mg'), kwargs.get('potassium_mg'), kwargs.get('calcium_mg'),
              kwargs.get('iron_mg'), kwargs.get('vitamin_d_ug'), kwargs.get('vitamin_c_mg'),
              kwargs.get('b12_ug'), kwargs.get('hydration_liters'), kwargs.get('meals_count'),
              kwargs.get('protein_per_kg'), kwargs.get('meal_timing'), kwargs.get('pre_workout_carbs_g'),
              kwargs.get('post_workout_carbs_g'), kwargs.get('creatine_g'), kwargs.get('notes')))
        self.conn.commit()
        return cursor.lastrowid
    
    def save_sleep_entry(self, user_id, entry_date, **kwargs):
        """Save comprehensive sleep entry"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO sleep_entries
            (user_id, entry_date, bedtime, wake_time, sleep_duration_hours, sleep_quality,
             time_to_fall_asleep_minutes, awakenings_count, deep_sleep_hours, rem_sleep_hours,
             caffeine_cutoff_time, screen_time_before_bed_minutes, room_temperature_celsius,
             sleep_environment_rating, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (user_id, entry_date, kwargs.get('bedtime'), kwargs.get('wake_time'),
              kwargs.get('sleep_duration_hours', 0), kwargs.get('sleep_quality', 0),
              kwargs.get('time_to_fall_asleep_minutes'), kwargs.get('awakenings_count'),
              kwargs.get('deep_sleep_hours'), kwargs.get('rem_sleep_hours'),
              kwargs.get('caffeine_cutoff_time'), kwargs.get('screen_time_before_bed_minutes'),
              kwargs.get('room_temperature_celsius'), kwargs.get('sleep_environment_rating'),
              kwargs.get('notes')))
        self.conn.commit()
        return cursor.lastrowid

    def save_body_measurement(self, user_id, measurement_date, **kwargs):
        """Save body measurements"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO body_measurements
            (user_id, measurement_date, weight_kg, body_fat_percentage, muscle_mass_kg,
             waist_cm, chest_cm, arm_cm, thigh_cm, neck_cm, forearm_cm, calf_cm, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (user_id, measurement_date, kwargs.get('weight_kg'), kwargs.get('body_fat_percentage'),
              kwargs.get('muscle_mass_kg'), kwargs.get('waist_cm'), kwargs.get('chest_cm'),
              kwargs.get('arm_cm'), kwargs.get('thigh_cm'), kwargs.get('neck_cm'),
              kwargs.get('forearm_cm'), kwargs.get('calf_cm'), kwargs.get('notes')))
        self.conn.commit()
        return cursor.lastrowid

    # ===== WORKOUT SYSTEM METHODS =====
    
    def get_all_exercises(self, user_id=None):
        """Get all exercises (system + user created)"""
        cursor = self.conn.cursor()
        if user_id:
            exercises = cursor.execute("""
                SELECT * FROM exercises 
                WHERE is_system_exercise = 1 OR created_by_user = ?
                ORDER BY name
            """, (user_id,)).fetchall()
        else:
            exercises = cursor.execute("SELECT * FROM exercises WHERE is_system_exercise = 1 ORDER BY name").fetchall()
        return [dict(ex) for ex in exercises]
    
    def create_exercise(self, user_id, **kwargs):
        """Create custom exercise"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO exercises
            (name, muscle_group_primary, muscle_group_secondary, equipment, movement_type,
             rep_range_min, rep_range_max, suggested_rir, rest_time_seconds, is_unilateral,
             limb_priority, resistance_profile, difficulty_level, description, notes,
             created_by_user, is_system_exercise)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
        """, (kwargs.get('name'), kwargs.get('muscle_group_primary'), 
              kwargs.get('muscle_group_secondary'), kwargs.get('equipment'),
              kwargs.get('movement_type'), kwargs.get('rep_range_min'),
              kwargs.get('rep_range_max'), kwargs.get('suggested_rir'),
              kwargs.get('rest_time_seconds'), kwargs.get('is_unilateral'),
              kwargs.get('limb_priority'), kwargs.get('resistance_profile'),
              kwargs.get('difficulty_level', 1), kwargs.get('description'),
              kwargs.get('notes'), user_id))
        self.conn.commit()
        return cursor.lastrowid
    
    def create_workout_template(self, user_id, name, description, workout_type, exercises):
        """Create workout template with exercises"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            INSERT INTO workout_templates (user_id, name, description, workout_type)
            VALUES (?, ?, ?, ?)
        """, (user_id, name, description, workout_type))
        
        template_id = cursor.lastrowid
        
        for i, ex_data in enumerate(exercises):
            cursor.execute("""
                INSERT INTO workout_template_exercises
                (workout_template_id, exercise_id, order_in_workout, target_sets,
                 target_reps_min, target_reps_max, target_rir, rest_seconds, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (template_id, ex_data['exercise_id'], i + 1,
                  ex_data.get('target_sets'), ex_data.get('target_reps_min'),
                  ex_data.get('target_reps_max'), ex_data.get('target_rir'),
                  ex_data.get('rest_seconds'), ex_data.get('notes')))
        
        self.conn.commit()
        return template_id
    
    def get_user_workout_templates(self, user_id):
        """Get user's workout templates"""
        cursor = self.conn.cursor()
        templates = cursor.execute("""
            SELECT * FROM workout_templates WHERE user_id = ? ORDER BY created_at DESC
        """, (user_id,)).fetchall()
        return [dict(template) for template in templates]
    
    def start_workout_session(self, user_id, template_id, session_name, session_date):
        """Start new workout session"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO workout_sessions 
            (user_id, workout_template_id, session_name, session_date, start_time)
            VALUES (?, ?, ?, ?, ?)
        """, (user_id, template_id, session_name, session_date, datetime.now()))
        self.conn.commit()
        return cursor.lastrowid
    
    def log_exercise_performance(self, workout_session_id, exercise_id, set_number, **kwargs):
        """Log individual set performance"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO exercise_performances
            (workout_session_id, exercise_id, set_number, weight_kg, reps_completed,
             rir_actual, rest_seconds, tempo, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (workout_session_id, exercise_id, set_number,
              kwargs.get('weight_kg'), kwargs.get('reps_completed'),
              kwargs.get('rir_actual'), kwargs.get('rest_seconds'),
              kwargs.get('tempo'), kwargs.get('notes')))
        self.conn.commit()
        return cursor.lastrowid
    
    def finish_workout_session(self, session_id, overall_rating, perceived_exertion, notes=""):
        """Complete workout session"""
        cursor = self.conn.cursor()
        cursor.execute("""
            UPDATE workout_sessions 
            SET end_time = ?, overall_rating = ?, perceived_exertion = ?, notes = ?
            WHERE id = ?
        """, (datetime.now(), overall_rating, perceived_exertion, notes, session_id))
        self.conn.commit()
    
    # ===== DATA RETRIEVAL METHODS =====
    
    def get_diet_entries(self, user_id, days=30):
        """Get diet entries"""
        cursor = self.conn.cursor()
        entries = cursor.execute("""
            SELECT * FROM diet_entries 
            WHERE user_id = ? AND entry_date >= date('now', '-' || ? || ' days')
            ORDER BY entry_date DESC
        """, (user_id, days)).fetchall()
        return [dict(entry) for entry in entries]
    
    def get_sleep_entries(self, user_id, days=30):
        """Get sleep entries"""
        cursor = self.conn.cursor()
        entries = cursor.execute("""
            SELECT * FROM sleep_entries
            WHERE user_id = ? AND entry_date >= date('now', '-' || ? || ' days')
            ORDER BY entry_date DESC
        """, (user_id, days)).fetchall()
        return [dict(entry) for entry in entries]
    
    def get_body_measurements(self, user_id, days=90):
        """Get body measurements"""
        cursor = self.conn.cursor()
        measurements = cursor.execute("""
            SELECT * FROM body_measurements
            WHERE user_id = ? AND measurement_date >= date('now', '-' || ? || ' days')
            ORDER BY measurement_date DESC
        """, (user_id, days)).fetchall()
        return [dict(m) for m in measurements]
    
    def get_recent_workout_sessions(self, user_id, days=30):
        """Get recent workout sessions"""
        cursor = self.conn.cursor()
        sessions = cursor.execute("""
            SELECT ws.*, wt.name as template_name 
            FROM workout_sessions ws
            LEFT JOIN workout_templates wt ON ws.workout_template_id = wt.id
            WHERE ws.user_id = ? AND ws.session_date >= date('now', '-' || ? || ' days')
            ORDER BY ws.session_date DESC, ws.start_time DESC
        """, (user_id, days)).fetchall()
        return [dict(session) for session in sessions]
    
    def close(self):
        """Close database connection"""
        self.conn.close()

