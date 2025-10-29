"""
Scientific Hypertrophy Trainer - Enhanced Database Manager

SQLite database with workout system, body measurements, comprehensive tracking
UPDATED FOR TASK 1: ML System Support Added
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
        self.add_missing_columns()  # NEW: Add missing columns
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
                weight_kg REAL,
                height_cm REAL,
                body_fat_percentage REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Assessments table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS assessments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                tier_level INTEGER NOT NULL,
                score INTEGER NOT NULL,
                total_questions INTEGER NOT NULL,
                percentage REAL NOT NULL,
                passed BOOLEAN NOT NULL,
                completed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        
        # Assessment answers (for learning center)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS assessment_answers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                assessment_id INTEGER NOT NULL,
                question_id TEXT NOT NULL,
                user_answer TEXT,
                correct_answer TEXT,
                is_correct BOOLEAN,
                FOREIGN KEY (assessment_id) REFERENCES assessments(id)
            )
        """)
        
        # Diet entries
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS diet_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                entry_date DATE NOT NULL,
                total_calories INTEGER,
                protein_g REAL,
                carbs_g REAL,
                fats_g REAL,
                fiber_g REAL,
                sodium_mg INTEGER,
                sugar_g REAL,
                hydration_liters REAL,
                meal_timing TEXT,
                creatine_taken BOOLEAN DEFAULT FALSE,
                notes TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        
        # Sleep entries
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sleep_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                entry_date DATE NOT NULL,
                sleep_duration_hours REAL,
                sleep_quality INTEGER,
                deep_sleep_hours REAL,
                rem_sleep_hours REAL,
                sleep_latency_minutes INTEGER,
                awakenings INTEGER,
                sleep_environment_rating INTEGER,
                notes TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id)
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
                chest_cm REAL,
                waist_cm REAL,
                hips_cm REAL,
                thigh_cm REAL,
                calf_cm REAL,
                biceps_cm REAL,
                forearm_cm REAL,
                shoulders_cm REAL,
                notes TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        
        # Exercises database
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS exercises (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                category TEXT NOT NULL,
                equipment TEXT,
                muscle_group_primary TEXT,
                muscle_groups_secondary TEXT,
                difficulty_level TEXT,
                is_compound BOOLEAN,
                instructions TEXT
            )
        """)
        
        # Workout sessions
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS workout_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                session_date DATE NOT NULL,
                session_name TEXT,
                total_duration_minutes INTEGER,
                notes TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        
        # Exercise performances within workout sessions
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS exercise_performances (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                workout_session_id INTEGER NOT NULL,
                exercise_id INTEGER NOT NULL,
                set_number INTEGER NOT NULL,
                reps_completed INTEGER,
                weight_kg REAL,
                rir_actual INTEGER,
                rpe_actual INTEGER,
                tempo TEXT,
                rest_seconds INTEGER,
                notes TEXT,
                FOREIGN KEY (workout_session_id) REFERENCES workout_sessions(id),
                FOREIGN KEY (exercise_id) REFERENCES exercises(id)
            )
        """)
        
        # ==========================================
        # NEW ML SYSTEM TABLES - TASK 1
        # ==========================================
        
        # Exercise Performance History (detailed per-exercise tracking)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS exercise_performance_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                exercise_id INTEGER NOT NULL,
                session_date DATE NOT NULL,
                session_number INTEGER DEFAULT 1,
                
                -- Performance metrics
                best_set_weight_kg REAL NOT NULL,
                best_set_reps INTEGER NOT NULL,
                best_set_rir INTEGER,
                total_sets INTEGER NOT NULL,
                total_volume_kg REAL,
                average_rir REAL,
                average_rpe REAL,
                
                -- Context
                rest_between_sets INTEGER,
                exercise_order INTEGER DEFAULT 1,
                notes TEXT,
                
                -- Timestamps
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                
                FOREIGN KEY (user_id) REFERENCES users(id),
                FOREIGN KEY (exercise_id) REFERENCES exercises(id)
            )
        """)
        
        # ML Predictions Storage
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ml_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                exercise_id INTEGER NOT NULL,
                
                -- Prediction metadata
                prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                target_session_date DATE NOT NULL,
                sessions_ahead INTEGER NOT NULL,
                
                -- Predictions
                predicted_weight_kg REAL NOT NULL,
                predicted_reps INTEGER NOT NULL,
                predicted_rir INTEGER NOT NULL,
                confidence REAL NOT NULL,
                
                -- Model info
                model_version TEXT DEFAULT 'v1.0',
                prediction_method TEXT DEFAULT 'lstm_bayesian',
                similar_users_json TEXT,
                
                FOREIGN KEY (user_id) REFERENCES users(id),
                FOREIGN KEY (exercise_id) REFERENCES exercises(id)
            )
        """)
        
        # Prediction Outcomes (accuracy tracking)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS prediction_outcomes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_id INTEGER NOT NULL,
                
                -- Actual results
                actual_weight_kg REAL,
                actual_reps INTEGER,
                actual_rir INTEGER,
                
                -- Error metrics
                weight_error_percent REAL,
                reps_error INTEGER,
                rir_error INTEGER,
                
                -- Accuracy flags
                weight_within_5pct BOOLEAN,
                reps_exact_match BOOLEAN,
                rir_within_1 BOOLEAN,
                
                -- Timestamps
                recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                
                FOREIGN KEY (prediction_id) REFERENCES ml_predictions(id)
            )
        """)
        
        # User Embeddings (collaborative filtering)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER UNIQUE NOT NULL,
                
                -- 20-dimensional embedding vector
                embedding_json TEXT NOT NULL,
                
                -- Component breakdowns
                demographics_json TEXT,
                diet_patterns_json TEXT,
                sleep_patterns_json TEXT,
                training_style_json TEXT,
                knowledge_json TEXT,
                
                -- Metadata
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                data_completeness_score REAL DEFAULT 0.0,
                
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        
        # Enhanced Supplements System
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS supplement_types (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                category TEXT NOT NULL,
                typical_dosage_mg INTEGER,
                optimal_timing TEXT,
                evidence_level TEXT DEFAULT 'moderate',
                interactions_json TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS supplement_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                supplement_id INTEGER NOT NULL,
                entry_date DATE NOT NULL,
                
                -- Dosage
                amount_mg REAL NOT NULL,
                timing TEXT NOT NULL,
                
                -- Context
                taken_with_food BOOLEAN DEFAULT FALSE,
                workout_day BOOLEAN DEFAULT FALSE,
                perceived_effect INTEGER,
                notes TEXT,
                
                FOREIGN KEY (user_id) REFERENCES users(id),
                FOREIGN KEY (supplement_id) REFERENCES supplement_types(id)
            )
        """)
        
        # Model Performance Metrics
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_version TEXT NOT NULL,
                evaluation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                
                -- Accuracy metrics
                weight_mae REAL,
                weight_mape REAL,
                reps_accuracy REAL,
                rir_accuracy REAL,
                
                -- Confidence calibration
                overconfident_rate REAL,
                underconfident_rate REAL,
                
                -- Sample sizes
                total_predictions INTEGER,
                horizon_1_accuracy REAL,
                horizon_2_accuracy REAL,
                horizon_4_accuracy REAL,
                horizon_10_accuracy REAL
            )
        """)
        
        self.conn.commit()
    
    def add_missing_columns(self):
        """Add any missing columns to existing tables"""
        cursor = self.conn.cursor()
        
        try:
            # Add age column to users table if it doesn't exist
            cursor.execute("ALTER TABLE users ADD COLUMN age INTEGER DEFAULT 25")
            self.conn.commit()
            print("✅ Added age column to users table")
        except sqlite3.OperationalError:
            # Column already exists
            pass
    
    def create_demo_data(self):
        """Create demo users and system exercises"""
        self.create_demo_users()
        self.create_system_exercises()
        self.seed_supplement_types()  # NEW: Seed supplements
    
    def seed_supplement_types(self):
        """Populate supplement types with evidence-based data"""
        supplements = [
            # Performance supplements
            ('Creatine Monohydrate', 'performance', 5000, 'anytime', 'strong', '[]'),
            ('Caffeine', 'performance', 200, 'pre_workout', 'strong', '[]'),
            ('Beta-Alanine', 'performance', 3000, 'pre_workout', 'moderate', '[]'),
            ('Citrulline Malate', 'performance', 6000, 'pre_workout', 'moderate', '[]'),
            ('HMB', 'performance', 3000, 'post_workout', 'weak', '[]'),
            
            # Recovery supplements  
            ('Whey Protein', 'recovery', 25000, 'post_workout', 'strong', '[]'),
            ('Casein Protein', 'recovery', 25000, 'evening', 'moderate', '[]'),
            ('Magnesium', 'recovery', 400, 'evening', 'moderate', '[]'),
            ('Zinc', 'recovery', 15, 'evening', 'moderate', '[]'),
            ('Taurine', 'recovery', 2000, 'post_workout', 'weak', '[]'),
            
            # Health supplements
            ('Vitamin D3', 'health', 2000, 'morning', 'strong', '[]'),
            ('Omega-3 Fish Oil', 'health', 2000, 'anytime', 'strong', '[]'),
            ('Multivitamin', 'health', 1, 'morning', 'moderate', '[]'),
            ('Ashwagandha', 'health', 600, 'evening', 'moderate', '[]'),
            ('Rhodiola Rosea', 'health', 400, 'morning', 'weak', '[]')
        ]
        
        cursor = self.conn.cursor()
        supplements_added = 0
        
        for supp_data in supplements:
            try:
                cursor.execute("""
                    INSERT OR IGNORE INTO supplement_types
                    (name, category, typical_dosage_mg, optimal_timing, evidence_level, interactions_json)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, supp_data)
                if cursor.rowcount > 0:
                    supplements_added += 1
            except sqlite3.IntegrityError:
                pass
        
        self.conn.commit()
        if supplements_added > 0:
            print(f"✅ Created {supplements_added} evidence-based supplements")
    
    def create_demo_users(self):
        """Create demo users if they don't exist"""
        demo_users = [
            {
                'username': 'Alex_Beginner',
                'experience_level': 'beginner',
                'primary_goal': 'hypertrophy',
                'weight_kg': 75.0,
                'height_cm': 175.0,
                'body_fat_percentage': 18.0
            },
            {
                'username': 'Sarah_Intermediate',
                'experience_level': 'intermediate',
                'primary_goal': 'strength',
                'weight_kg': 62.0,
                'height_cm': 165.0,
                'body_fat_percentage': 22.0
            },
            {
                'username': 'Marcus_Advanced',
                'experience_level': 'advanced',
                'primary_goal': 'hypertrophy',
                'weight_kg': 85.0,
                'height_cm': 180.0,
                'body_fat_percentage': 12.0
            }
        ]
        
        cursor = self.conn.cursor()
        
        for user_data in demo_users:
            try:
                cursor.execute("""
                    INSERT INTO users (username, experience_level, primary_goal, 
                                     weight_kg, height_cm, body_fat_percentage)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (user_data['username'], user_data['experience_level'], 
                      user_data['primary_goal'], user_data['weight_kg'],
                      user_data['height_cm'], user_data['body_fat_percentage']))
            except sqlite3.IntegrityError:
                # User already exists
                pass
        
        self.conn.commit()
    
    def create_system_exercises(self):
        """Create comprehensive exercise database"""
        # This would be a very long list - keeping it abbreviated
        exercises = [
            # Chest exercises
            ('Barbell Bench Press', 'push', 'barbell', 'chest', 'shoulders,triceps', 'intermediate', True, 'Compound pressing movement'),
            ('Incline Dumbbell Press', 'push', 'dumbbell', 'chest', 'shoulders,triceps', 'intermediate', True, 'Upper chest focus'),
            ('Cable Chest Fly', 'push', 'cable', 'chest', '', 'beginner', False, 'Isolation movement'),
            
            # Back exercises
            ('Deadlift', 'pull', 'barbell', 'back', 'hamstrings,glutes', 'advanced', True, 'Full posterior chain'),
            ('Pull-ups', 'pull', 'bodyweight', 'back', 'biceps', 'intermediate', True, 'Vertical pulling'),
            ('Barbell Row', 'pull', 'barbell', 'back', 'biceps', 'intermediate', True, 'Horizontal pulling'),
            
            # Leg exercises
            ('Barbell Squat', 'legs', 'barbell', 'quads', 'glutes,hamstrings', 'intermediate', True, 'King of exercises'),
            ('Romanian Deadlift', 'legs', 'barbell', 'hamstrings', 'glutes,back', 'intermediate', True, 'Hamstring focus'),
            ('Leg Press', 'legs', 'machine', 'quads', 'glutes', 'beginner', True, 'Quad dominant'),
            
            # Shoulder exercises
            ('Overhead Press', 'push', 'barbell', 'shoulders', 'triceps', 'intermediate', True, 'Vertical press'),
            ('Lateral Raise', 'push', 'dumbbell', 'shoulders', '', 'beginner', False, 'Lateral delt isolation'),
            
            # Arms
            ('Barbell Curl', 'pull', 'barbell', 'biceps', '', 'beginner', False, 'Bicep mass builder'),
            ('Tricep Pushdown', 'push', 'cable', 'triceps', '', 'beginner', False, 'Tricep isolation')
        ]
        
        cursor = self.conn.cursor()
        
        for exercise in exercises:
            try:
                cursor.execute("""
                    INSERT INTO exercises 
                    (name, category, equipment, muscle_group_primary, 
                     muscle_groups_secondary, difficulty_level, is_compound, instructions)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, exercise)
            except sqlite3.IntegrityError:
                # Exercise already exists
                pass
        
        self.conn.commit()
    
    # ==========================================
    # EXISTING METHODS (UNCHANGED)
    # ==========================================
    
    def create_user(self, username, experience_level, primary_goal='hypertrophy', **kwargs):
        """Create a new user"""
        cursor = self.conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO users (username, experience_level, primary_goal, 
                                 weight_kg, height_cm, body_fat_percentage)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (username, experience_level, primary_goal,
                  kwargs.get('weight_kg'), kwargs.get('height_cm'), 
                  kwargs.get('body_fat_percentage')))
            
            self.conn.commit()
            return cursor.lastrowid
        except sqlite3.IntegrityError:
            return None
    
    def get_user_by_username(self, username):
        """Get user by username"""
        cursor = self.conn.cursor()
        user = cursor.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        return dict(user) if user else None
    
    def get_user_by_id(self, user_id):
        """Get user by ID"""
        cursor = self.conn.cursor()
        user = cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        return dict(user) if user else None
    
    def get_all_users(self):
        """Get all users"""
        cursor = self.conn.cursor()
        users = cursor.execute("SELECT * FROM users ORDER BY created_at DESC").fetchall()
        return [dict(user) for user in users]
    
    def save_assessment_result(self, user_id, tier_level, score, total_questions, 
                               percentage, passed, answers):
        """Save assessment results"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            INSERT INTO assessments 
            (user_id, tier_level, score, total_questions, percentage, passed)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (user_id, tier_level, score, total_questions, percentage, passed))
        
        assessment_id = cursor.lastrowid
        
        for answer in answers:
            cursor.execute("""
                INSERT INTO assessment_answers
                (assessment_id, question_id, user_answer, correct_answer, is_correct)
                VALUES (?, ?, ?, ?, ?)
            """, (assessment_id, answer['question_id'], answer['user_answer'],
                  answer['correct_answer'], answer['is_correct']))
        
        self.conn.commit()
        return assessment_id
    
    def get_user_assessments(self, user_id):
        """Get all assessments for a user"""
        cursor = self.conn.cursor()
        assessments = cursor.execute("""
            SELECT * FROM assessments 
            WHERE user_id = ? 
            ORDER BY tier_level, completed_at DESC
        """, (user_id,)).fetchall()
        return [dict(assessment) for assessment in assessments]
    
    def get_assessment_answers(self, assessment_id):
        """Get answers for a specific assessment"""
        cursor = self.conn.cursor()
        answers = cursor.execute("""
            SELECT * FROM assessment_answers 
            WHERE assessment_id = ?
        """, (assessment_id,)).fetchall()
        return [dict(answer) for answer in answers]
    
    def save_diet_entry(self, user_id, entry_date, **kwargs):
        """Save diet entry"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO diet_entries
            (user_id, entry_date, total_calories, protein_g, carbs_g, fats_g,
             fiber_g, sodium_mg, sugar_g, hydration_liters, meal_timing, 
             creatine_taken, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (user_id, entry_date, kwargs.get('total_calories'), 
              kwargs.get('protein_g'), kwargs.get('carbs_g'), kwargs.get('fats_g'),
              kwargs.get('fiber_g'), kwargs.get('sodium_mg'), kwargs.get('sugar_g'),
              kwargs.get('hydration_liters'), kwargs.get('meal_timing'),
              kwargs.get('creatine_taken', False), kwargs.get('notes')))
        self.conn.commit()
        return cursor.lastrowid
    
    def get_diet_entries(self, user_id, days=30):
        """Get diet entries for user"""
        cursor = self.conn.cursor()
        entries = cursor.execute("""
            SELECT * FROM diet_entries
            WHERE user_id = ? AND entry_date >= date('now', '-{} days')
            ORDER BY entry_date DESC
        """.format(days), (user_id,)).fetchall()
        return [dict(entry) for entry in entries]
    
    def save_sleep_entry(self, user_id, entry_date, **kwargs):
        """Save sleep entry"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO sleep_entries
            (user_id, entry_date, sleep_duration_hours, sleep_quality,
             deep_sleep_hours, rem_sleep_hours, sleep_latency_minutes,
             awakenings, sleep_environment_rating, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (user_id, entry_date, kwargs.get('sleep_duration_hours'),
              kwargs.get('sleep_quality'), kwargs.get('deep_sleep_hours'),
              kwargs.get('rem_sleep_hours'), kwargs.get('sleep_latency_minutes'),
              kwargs.get('awakenings'), kwargs.get('sleep_environment_rating'),
              kwargs.get('notes')))
        self.conn.commit()
        return cursor.lastrowid
    
    def get_sleep_entries(self, user_id, days=30):
        """Get sleep entries for user"""
        cursor = self.conn.cursor()
        entries = cursor.execute("""
            SELECT * FROM sleep_entries
            WHERE user_id = ? AND entry_date >= date('now', '-{} days')
            ORDER BY entry_date DESC
        """.format(days), (user_id,)).fetchall()
        return [dict(entry) for entry in entries]
    
    def save_body_measurement(self, user_id, measurement_date, **kwargs):
        """Save body measurements"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO body_measurements
            (user_id, measurement_date, weight_kg, body_fat_percentage,
             chest_cm, waist_cm, hips_cm, thigh_cm, calf_cm,
             biceps_cm, forearm_cm, shoulders_cm, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (user_id, measurement_date, kwargs.get('weight_kg'),
              kwargs.get('body_fat_percentage'), kwargs.get('chest_cm'),
              kwargs.get('waist_cm'), kwargs.get('hips_cm'),
              kwargs.get('thigh_cm'), kwargs.get('calf_cm'),
              kwargs.get('biceps_cm'), kwargs.get('forearm_cm'),
              kwargs.get('shoulders_cm'), kwargs.get('notes')))
        self.conn.commit()
        return cursor.lastrowid
    
    def get_body_measurements(self, user_id, days=90):
        """Get body measurements for user"""
        cursor = self.conn.cursor()
        measurements = cursor.execute("""
            SELECT * FROM body_measurements
            WHERE user_id = ? AND measurement_date >= date('now', '-{} days')
            ORDER BY measurement_date DESC
        """.format(days), (user_id,)).fetchall()
        return [dict(measurement) for measurement in measurements]
    
    def get_all_exercises(self):
        """Get all exercises"""
        cursor = self.conn.cursor()
        exercises = cursor.execute("SELECT * FROM exercises ORDER BY name").fetchall()
        return [dict(exercise) for exercise in exercises]
    
    def get_exercise_by_id(self, exercise_id):
        """Get exercise by ID"""
        cursor = self.conn.cursor()
        exercise = cursor.execute("SELECT * FROM exercises WHERE id = ?", (exercise_id,)).fetchone()
        return dict(exercise) if exercise else None
    
    def create_workout_session(self, user_id, session_date, session_name=None, notes=None):
        """Create a new workout session"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO workout_sessions (user_id, session_date, session_name, notes)
            VALUES (?, ?, ?, ?)
        """, (user_id, session_date, session_name, notes))
        self.conn.commit()
        return cursor.lastrowid
    
    def add_exercise_performance(self, workout_session_id, exercise_id, set_number, **kwargs):
        """Add exercise performance to workout session"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO exercise_performances
            (workout_session_id, exercise_id, set_number, reps_completed,
             weight_kg, rir_actual, rpe_actual, tempo, rest_seconds, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (workout_session_id, exercise_id, set_number,
              kwargs.get('reps_completed'), kwargs.get('weight_kg'),
              kwargs.get('rir_actual'), kwargs.get('rpe_actual'),
              kwargs.get('tempo'), kwargs.get('rest_seconds'), kwargs.get('notes')))
        self.conn.commit()
        return cursor.lastrowid
    
    def get_recent_workout_sessions(self, user_id, days=30):
        """Get recent workout sessions"""
        cursor = self.conn.cursor()
        sessions = cursor.execute("""
            SELECT * FROM workout_sessions
            WHERE user_id = ? AND session_date >= date('now', '-{} days')
            ORDER BY session_date DESC
        """.format(days), (user_id,)).fetchall()
        return [dict(session) for session in sessions]
    
    def get_workout_session_details(self, session_id):
        """Get detailed workout session with all exercises"""
        cursor = self.conn.cursor()
        
        session = cursor.execute("""
            SELECT * FROM workout_sessions WHERE id = ?
        """, (session_id,)).fetchone()
        
        if not session:
            return None
        
        performances = cursor.execute("""
            SELECT ep.*, e.name as exercise_name
            FROM exercise_performances ep
            JOIN exercises e ON ep.exercise_id = e.id
            WHERE ep.workout_session_id = ?
            ORDER BY ep.set_number
        """, (session_id,)).fetchall()
        
        return {
            'session': dict(session),
            'performances': [dict(perf) for perf in performances]
        }
    
    # ==========================================
    # NEW ML SYSTEM METHODS - TASK 1
    # ==========================================
    
    def store_exercise_performance(self, user_id, exercise_id, session_date, 
                                  weight_kg, reps, rir, total_sets, 
                                  exercise_order=1, rest_seconds=None, notes=None):
        """Store detailed exercise performance for ML training"""
        cursor = self.conn.cursor()
        
        # Calculate total volume
        total_volume = weight_kg * reps * total_sets if weight_kg and reps and total_sets else None
        
        cursor.execute("""
            INSERT OR REPLACE INTO exercise_performance_history 
            (user_id, exercise_id, session_date, best_set_weight_kg, best_set_reps, 
             best_set_rir, total_sets, total_volume_kg, exercise_order, 
             rest_between_sets, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (user_id, exercise_id, session_date, weight_kg, reps, rir, 
              total_sets, total_volume, exercise_order, rest_seconds, notes))
        
        self.conn.commit()
        return cursor.lastrowid
    
    def get_exercise_performance_history(self, user_id, exercise_id=None, days=90):
        """Get historical performance for exercises"""
        cursor = self.conn.cursor()
        
        if exercise_id:
            # Specific exercise
            return cursor.execute("""
                SELECT eph.*, e.name as exercise_name 
                FROM exercise_performance_history eph
                JOIN exercises e ON eph.exercise_id = e.id
                WHERE eph.user_id = ? AND eph.exercise_id = ?
                AND eph.session_date >= date('now', '-{} days')
                ORDER BY eph.session_date DESC
            """.format(days), (user_id, exercise_id)).fetchall()
        else:
            # All exercises
            return cursor.execute("""
                SELECT eph.*, e.name as exercise_name 
                FROM exercise_performance_history eph
                JOIN exercises e ON eph.exercise_id = e.id
                WHERE eph.user_id = ?
                AND eph.session_date >= date('now', '-{} days')
                ORDER BY eph.session_date DESC
            """.format(days), (user_id,)).fetchall()
    
    def store_ml_prediction(self, user_id, exercise_id, target_date, sessions_ahead,
                           predicted_weight, predicted_reps, predicted_rir, 
                           confidence, method='lstm_bayesian', similar_users=None):
        """Store ML prediction for later accuracy validation"""
        cursor = self.conn.cursor()
        
        similar_users_json = json.dumps(similar_users) if similar_users else None
        
        cursor.execute("""
            INSERT INTO ml_predictions 
            (user_id, exercise_id, target_session_date, sessions_ahead,
             predicted_weight_kg, predicted_reps, predicted_rir, confidence,
             prediction_method, similar_users_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (user_id, exercise_id, target_date, sessions_ahead,
              predicted_weight, predicted_reps, predicted_rir, confidence,
              method, similar_users_json))
        
        self.conn.commit()
        return cursor.lastrowid
    
    def record_prediction_outcome(self, prediction_id, actual_weight, actual_reps, actual_rir):
        """Record actual outcome to validate prediction accuracy"""
        cursor = self.conn.cursor()
        
        # Get original prediction
        prediction = cursor.execute("""
            SELECT predicted_weight_kg, predicted_reps, predicted_rir 
            FROM ml_predictions WHERE id = ?
        """, (prediction_id,)).fetchone()
        
        if not prediction:
            return None
        
        # Calculate errors
        weight_error_pct = ((actual_weight - prediction['predicted_weight_kg']) 
                           / prediction['predicted_weight_kg'] * 100) if prediction['predicted_weight_kg'] > 0 else 0
        reps_error = actual_reps - prediction['predicted_reps']
        rir_error = actual_rir - prediction['predicted_rir']
        
        # Accuracy flags
        weight_within_5pct = abs(weight_error_pct) <= 5
        reps_exact = reps_error == 0
        rir_within_1 = abs(rir_error) <= 1
        
        cursor.execute("""
            INSERT INTO prediction_outcomes
            (prediction_id, actual_weight_kg, actual_reps, actual_rir,
             weight_error_percent, reps_error, rir_error,
             weight_within_5pct, reps_exact_match, rir_within_1)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (prediction_id, actual_weight, actual_reps, actual_rir,
              weight_error_pct, reps_error, rir_error,
              weight_within_5pct, reps_exact, rir_within_1))
        
        self.conn.commit()
        return cursor.lastrowid
    
    def get_user_ml_data(self, user_id, days=30):
        """Extract ML features for user from last N days"""
        cursor = self.conn.cursor()
        
        # Diet entries
        diet_entries = cursor.execute("""
            SELECT * FROM diet_entries 
            WHERE user_id = ? AND entry_date >= date('now', '-{} days')
            ORDER BY entry_date
        """.format(days), (user_id,)).fetchall()
        
        # Sleep entries
        sleep_entries = cursor.execute("""
            SELECT * FROM sleep_entries
            WHERE user_id = ? AND entry_date >= date('now', '-{} days')
            ORDER BY entry_date
        """.format(days), (user_id,)).fetchall()
        
        # Body measurements
        body_entries = cursor.execute("""
            SELECT * FROM body_measurements
            WHERE user_id = ? AND measurement_date >= date('now', '-{} days')
            ORDER BY measurement_date
        """.format(days), (user_id,)).fetchall()
        
        # Exercise performances
        exercise_performances = cursor.execute("""
            SELECT eph.*, e.name as exercise_name, e.muscle_group_primary
            FROM exercise_performance_history eph
            JOIN exercises e ON eph.exercise_id = e.id
            WHERE eph.user_id = ? AND eph.session_date >= date('now', '-{} days')
            ORDER BY eph.session_date
        """.format(days), (user_id,)).fetchall()
        
        # Supplement entries
        supplement_entries = cursor.execute("""
            SELECT se.*, st.name as supplement_name, st.category
            FROM supplement_entries se
            JOIN supplement_types st ON se.supplement_id = st.id
            WHERE se.user_id = ? AND se.entry_date >= date('now', '-{} days')
            ORDER BY se.entry_date
        """.format(days), (user_id,)).fetchall()
        
        return {
            'diet': [dict(d) for d in diet_entries],
            'sleep': [dict(s) for s in sleep_entries],
            'body': [dict(b) for b in body_entries],
            'workouts': [dict(w) for w in exercise_performances],
            'supplements': [dict(s) for s in supplement_entries]
        }
    
    def store_user_embedding(self, user_id, embedding_vector, component_data):
        """Store user embedding for collaborative filtering"""
        cursor = self.conn.cursor()
        
        embedding_json = json.dumps(embedding_vector.tolist() if hasattr(embedding_vector, 'tolist') else embedding_vector)
        demographics_json = json.dumps(component_data.get('demographics', []))
        diet_json = json.dumps(component_data.get('diet_patterns', []))
        sleep_json = json.dumps(component_data.get('sleep_patterns', []))
        training_json = json.dumps(component_data.get('training_style', []))
        knowledge_json = json.dumps(component_data.get('knowledge', []))
        
        cursor.execute("""
            INSERT OR REPLACE INTO user_embeddings
            (user_id, embedding_json, demographics_json, diet_patterns_json,
             sleep_patterns_json, training_style_json, knowledge_json,
             data_completeness_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (user_id, embedding_json, demographics_json, diet_json,
              sleep_json, training_json, knowledge_json,
              component_data.get('completeness_score', 0.0)))
        
        self.conn.commit()
    
    def get_user_embedding(self, user_id):
        """Retrieve user embedding"""
        cursor = self.conn.cursor()
        result = cursor.execute("SELECT * FROM user_embeddings WHERE user_id = ?", (user_id,)).fetchone()
        return dict(result) if result else None
    
    def save_supplement_entry(self, user_id, supplement_id, entry_date, amount_mg, timing, **kwargs):
        """Save supplement entry"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO supplement_entries
            (user_id, supplement_id, entry_date, amount_mg, timing, taken_with_food, 
             workout_day, perceived_effect, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (user_id, supplement_id, entry_date, amount_mg, timing,
              kwargs.get('taken_with_food', False), kwargs.get('workout_day', False),
              kwargs.get('perceived_effect'), kwargs.get('notes')))
        self.conn.commit()
        return cursor.lastrowid
    
    def get_supplement_entries(self, user_id, days=30):
        """Get supplement entries"""
        cursor = self.conn.cursor()
        entries = cursor.execute("""
            SELECT se.*, st.name as supplement_name, st.category, st.typical_dosage_mg
            FROM supplement_entries se
            JOIN supplement_types st ON se.supplement_id = st.id
            WHERE se.user_id = ? AND se.entry_date >= date('now', '-{} days')
            ORDER BY se.entry_date DESC
        """.format(days), (user_id,)).fetchall()
        return [dict(entry) for entry in entries]
    
    def get_all_supplement_types(self):
        """Get all available supplement types"""
        cursor = self.conn.cursor()
        supplements = cursor.execute("SELECT * FROM supplement_types ORDER BY category, name").fetchall()
        return [dict(supp) for supp in supplements]
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()