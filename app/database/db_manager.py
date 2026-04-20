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
        self.add_missing_columns()
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
                age INTEGER DEFAULT 25,
                last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP, 
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
                question_text TEXT,
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
        # Workout Templates (The "Folders")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS workout_templates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER, -- NULL for system templates
                name TEXT NOT NULL,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Exercises inside a Template
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS template_exercises (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                template_id INTEGER,
                exercise_id INTEGER,
                order_index INTEGER,
                target_sets INTEGER,
                target_reps TEXT, -- e.g. "8-12"
                FOREIGN KEY(template_id) REFERENCES workout_templates(id),
                FOREIGN KEY(exercise_id) REFERENCES exercises(id)
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
        
        columns_to_add = [
            ("users", "age", "INTEGER DEFAULT 25"),
            ("users", "last_active", "TEXT"),
            ("exercises", "is_unilateral", "BOOLEAN DEFAULT 0"),
            ("exercises", "stability_score", "INTEGER"),
            ("exercises", "resistance_profile", "TEXT"),
            ("exercises", "regional_bias", "TEXT"),
            ("assessment_answers", "question_text", "TEXT"),
        ]
        
        for table, column, col_type in columns_to_add:
            try:
                cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
                self.conn.commit()
            except sqlite3.OperationalError:
                pass

    def create_demo_data(self):
        """Create demo users and system exercises"""
        self.create_demo_users()
        self.seed_scientific_exercises()
        self.seed_workout_templates()
        self.seed_supplement_types()
    
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
        """
        Saves the main result AND the individual answers for the Learning Center.
        """
        cursor = self.conn.cursor()
        
        try:
            # 1. Save Main Assessment Record
            cursor.execute("""
                INSERT INTO assessments 
                (user_id, tier_level, score, total_questions, percentage, passed)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (user_id, tier_level, score, total_questions, percentage, passed))
            
            assessment_id = cursor.lastrowid
            
            # 2. Save Individual Answers (The Missing Link)
            for ans in answers:
                # Map Engine keys to Database columns
                # Engine uses 'selected_answer', DB uses 'user_answer'
                q_id = ans.get('question_id', 'Unknown')
                user_ans = ans.get('selected_answer') or ans.get('user_answer')
                corr_ans = ans.get('correct_answer')
                is_right = ans.get('is_correct')
                q_text = ans.get('question_text', '') # Save text for context
                
                # We save the question text in the question_id column if needed, 
                # or create a new column. For now, let's append text to ID to be safe
                # or just save the ID if your logic relies on lookups. 
                # Let's save the full text in a way we can retrieve it.
                
                cursor.execute("""
                    INSERT INTO assessment_answers
                    (assessment_id, question_id, user_answer, correct_answer, is_correct, question_text)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (assessment_id, q_id, user_ans, corr_ans, is_right, q_text))
            
            self.conn.commit()
            print(f"✅ Saved Assessment {assessment_id} with {len(answers)} answer details.")
            return assessment_id
            
        except Exception as e:
            print(f"❌ DB Error saving assessment: {e}")
            self.conn.rollback()
            return None
    
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
        """Save or update diet entry for a given date (upsert)."""
        cursor = self.conn.cursor()
        fats = kwargs.get('fats_g') or kwargs.get('fat_g')
        existing = cursor.execute(
            "SELECT id FROM diet_entries WHERE user_id=? AND entry_date=?",
            (user_id, entry_date)).fetchone()
        if existing:
            cursor.execute("""
                UPDATE diet_entries SET total_calories=?, protein_g=?, carbs_g=?, fats_g=?,
                fiber_g=?, sodium_mg=?, sugar_g=?, hydration_liters=?, meal_timing=?,
                creatine_taken=?, notes=? WHERE id=?
            """, (kwargs.get('total_calories'), kwargs.get('protein_g'),
                  kwargs.get('carbs_g'), fats,
                  kwargs.get('fiber_g'), kwargs.get('sodium_mg'), kwargs.get('sugar_g'),
                  kwargs.get('hydration_liters'), kwargs.get('meal_timing'),
                  kwargs.get('creatine_taken', False), kwargs.get('notes'),
                  existing[0]))
            self.conn.commit()
            return existing[0]
        else:
            cursor.execute("""
                INSERT INTO diet_entries
                (user_id, entry_date, total_calories, protein_g, carbs_g, fats_g,
                 fiber_g, sodium_mg, sugar_g, hydration_liters, meal_timing,
                 creatine_taken, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (user_id, entry_date, kwargs.get('total_calories'),
                  kwargs.get('protein_g'), kwargs.get('carbs_g'), fats,
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
        """Save or update sleep entry for a given date (upsert)."""
        cursor = self.conn.cursor()
        existing = cursor.execute(
            "SELECT id FROM sleep_entries WHERE user_id=? AND entry_date=?",
            (user_id, entry_date)).fetchone()
        if existing:
            cursor.execute("""
                UPDATE sleep_entries SET sleep_duration_hours=?, sleep_quality=?,
                deep_sleep_hours=?, rem_sleep_hours=?, sleep_latency_minutes=?,
                awakenings=?, sleep_environment_rating=?, notes=? WHERE id=?
            """, (kwargs.get('sleep_duration_hours'), kwargs.get('sleep_quality'),
                  kwargs.get('deep_sleep_hours'), kwargs.get('rem_sleep_hours'),
                  kwargs.get('sleep_latency_minutes'), kwargs.get('awakenings'),
                  kwargs.get('sleep_environment_rating'), kwargs.get('notes'),
                  existing[0]))
            self.conn.commit()
            return existing[0]
        else:
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

    def update_user_last_active(self, user_id):
        """Updates the last_active timestamp for the user."""
        from datetime import datetime
        try:
            cursor = self.conn.cursor()
            timestamp = datetime.now().isoformat()
            cursor.execute(
                "UPDATE users SET last_active = ? WHERE id = ?",
                (timestamp, user_id)
            )
            self.conn.commit()
        except Exception as e:
            print(f"Warning: Could not update last_active for user {user_id}: {e}")

    def get_user_tier_progress(self, user_id):
        """Get user's assessment tier progress for the user manager."""
        try:
            cursor = self.conn.cursor()
            result = cursor.execute(
                "SELECT MAX(tier_level) FROM assessments WHERE user_id = ? AND passed = 1",
                (user_id,)
            ).fetchone()
            
            # FIX: If result is None, they haven't passed anything (-1). 
            # If they passed Tier 1 (which is index 0), max_passed is 0.
            max_passed = int(result[0]) if result and result[0] is not None else -1
            
            # The tier they are currently allowed to take
            current_tier = min(max_passed + 1, 2)
            
            return {
                'current_tier': current_tier,
                'tier_1_passed': max_passed >= 0,
                'tier_2_unlocked': max_passed >= 0,  # Unlocks if Tier 1 (0) is passed
                'tier_2_passed': max_passed >= 1,
                'tier_3_unlocked': max_passed >= 1,  # Unlocks if Tier 2 (1) is passed
                'tier_3_passed': max_passed >= 2,
                'tier_4_unlocked': max_passed >= 2,
            }
        except Exception as e:
            print(f"Warning: Could not get tier progress for user {user_id}: {e}")
            return {
                'current_tier': 0, 'tier_1_passed': False, 'tier_2_unlocked': False,
                'tier_2_passed': False, 'tier_3_unlocked': False, 'tier_3_passed': False,
                'tier_4_unlocked': False,
            }
            
    def get_training_entries(self, user_id, days=7):
        """Get recent training/workout entries for user"""
        cursor = self.conn.cursor()
        
        # Get recent workout sessions with exercise details
        sessions = cursor.execute("""
            SELECT ws.*, 
                   COUNT(DISTINCT ep.exercise_id) as exercise_count,
                   SUM(ep.reps_completed * ep.weight_kg) as total_volume
            FROM workout_sessions ws
            LEFT JOIN exercise_performances ep ON ws.id = ep.workout_session_id
            WHERE ws.user_id = ? 
            AND ws.session_date >= date('now', '-{} days')
            GROUP BY ws.id
            ORDER BY ws.session_date DESC
        """.format(days), (user_id,)).fetchall()
        
        return [dict(session) for session in sessions]


    def get_user_workout_history_df(self, user_id, exercise_name):
        """
        Retrieves training history for a specific exercise + daily biometrics.
        Returns a Pandas DataFrame ready for the ML Engine.
        Uses ws.session_date for the date column.
        """
        import pandas as pd
        
        # We join exercise_performances (ep) to workout_sessions (ws) to get the date
        query = """
            SELECT 
                ws.session_date as date,
                ep.weight_kg,
                ep.reps_completed as reps,
                ep.rir_actual as rir,
                s.sleep_duration_hours,
                s.sleep_quality,
                d.protein_g,
                d.total_calories as calories
            FROM exercise_performances ep
            JOIN exercises e ON ep.exercise_id = e.id
            JOIN workout_sessions ws ON ep.workout_session_id = ws.id
            LEFT JOIN sleep_entries s ON ws.session_date = s.entry_date AND ws.user_id = s.user_id
            LEFT JOIN diet_entries d ON ws.session_date = d.entry_date AND ws.user_id = d.user_id
            WHERE ws.user_id = ? AND e.name = ?
            ORDER BY ws.session_date ASC
        """
        
        try:
            df = pd.read_sql_query(query, self.conn, params=(user_id, exercise_name))
            
            if df.empty:
                return df
                
            # Aggregate to one row per day (taking the max weight used that day)
            df = df.groupby('date').agg({
                'weight_kg': 'max',
                'reps': 'mean',
                'rir': 'mean',
                'sleep_duration_hours': 'first',
                'sleep_quality': 'first',
                'protein_g': 'first',
                'calories': 'first'
            }).reset_index()
            
            # Fill missing values so the AI doesn't crash
            defaults = {
                'sleep_duration_hours': 7.5,
                'sleep_quality': 7,
                'protein_g': 150,
                'calories': 2500,
                'rir': 2
            }
            df.fillna(defaults, inplace=True)
            
            return df
            
        except Exception as e:
            print(f"❌ DB Error getting AI history: {e}")
            return pd.DataFrame()


    def get_user_ml_profile(self, user_id):
        """Returns static user features for the AI."""
        user = self.get_user_by_id(user_id)
        if not user: return {}
        
        # Determine Knowledge Tier
        tier_info = self.get_user_tier_progress(user_id)
        current_tier = tier_info.get('current_tier', 0)
        
        # Convert tier to 0.0-1.0 literacy score
        literacy_index = 0.2 + (current_tier * 0.25)
        
        return {
            'age': user.get('age', 25),
            'weight_kg_user': user.get('weight_kg', 75),
            'height_cm': user.get('height_cm', 175),
            'body_fat_pct': user.get('body_fat_percentage', 15),
            'assessment_score': literacy_index * 100,
            'training_literacy_index': literacy_index,
            'load_management_score': literacy_index,
            'technique_score': literacy_index,
            'recovery_knowledge': literacy_index,
            # Placeholder defaults for the 35-feature vector
            'creatine': 0, 'pre_workout': 0, 'protein_powder': 0, 'caffeine_mg': 0,
            'stress_level': 5, 'soreness_level': 5, 'fatigue_level': 5, 
            'readiness_score': 5, 'hrv': 60, 'resting_heart_rate': 60, 
            'session_rpe': 7, 'recovery_quality': 5, 'days_since_last_session': 2
        }

        
    def add_custom_exercise(self, name, category, muscle, equipment, difficulty, is_compound, description, is_unilateral):
        """Adds a user-defined exercise with full AI metadata"""
        cursor = self.conn.cursor()
        try:
            cursor.execute("""
                INSERT INTO exercises 
                (name, category, muscle_group_primary, equipment, difficulty_level, 
                 is_compound, instructions, is_unilateral)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (name, category, muscle, equipment, difficulty, is_compound, description, is_unilateral))
            self.conn.commit()
            return cursor.lastrowid
        except Exception as e:
            print(f"Error adding exercise: {e}")
            return None

    def seed_scientific_exercises(self):
        """
        Populates DB with evidence-based hypertrophy exercises.
        Includes biomechanics metadata (Beardsley 2020):
        - resistance_profile: ascending/descending/bell/constant
        - stability_score: 1-10 (10 = most stable, e.g. machine)
        - regional_bias: which sub-region of the muscle is most loaded
        - sfr_rating: stimulus-to-fatigue ratio (1-10)
        - lengthened_bias: 0/1 (legacy field, not used for exercise selection)
        """
        cursor = self.conn.cursor()
        count = cursor.execute("SELECT count(*) FROM exercises").fetchone()[0]
        if count > 20:
            return

        exercises = [
            # (name, cat, muscle, equip, diff, compound, instructions, unilateral,
            #  stability, resistance_profile, regional_bias, sfr, lengthened_bias)

            # ─── CHEST ───
            ("Flat Barbell Bench Press", "Push", "Chest", "Barbell", "Intermediate", True,
             "Ascending resistance profile. Peak tension at lockout. Sternal head dominant. "
             "High absolute load capacity but shoulder impingement risk at end-range.",
             False, 7, "ascending", "sternal_mid", 6, 0),
            ("Flat DB Bench Press", "Push", "Chest", "Dumbbell", "Intermediate", True,
             "Greater ROM than barbell. More adduction at top. Higher stability demand "
             "means slightly less absolute load but superior per-fiber tension for pecs.",
             True, 5, "bell", "sternal_mid", 7, 0),
            ("Incline Barbell Press", "Push", "Chest", "Barbell", "Intermediate", True,
             "30-45 degree incline biases clavicular (upper) pec fibers. "
             "Fixed bar path allows high loading. Anterior deltoid is a significant synergist.",
             False, 7, "ascending", "clavicular_upper", 6, 0),
            ("Incline DB Press", "Push", "Chest", "Dumbbell", "Intermediate", True,
             "Incline + dumbbell = upper pec bias with greater ROM and adduction. "
             "Peak tension in mid-range. Superior upper chest stimulus vs barbell variant.",
             True, 4, "bell", "clavicular_upper", 7, 0),
            ("Cable Fly (Low-to-High)", "Push", "Chest", "Cable", "Beginner", False,
             "Constant tension throughout ROM. Low-to-high angle biases clavicular fibers. "
             "Excellent shortened-position overload for upper chest.",
             True, 8, "constant", "clavicular_upper", 9, 0),
            ("Cable Fly (High-to-Low)", "Push", "Chest", "Cable", "Beginner", False,
             "Targets sternal and costal pec fibers. Constant tension. "
             "Loads the muscle through the full ROM without joint stress.",
             True, 8, "constant", "sternal_lower", 9, 0),
            ("Pec Deck / Machine Fly", "Push", "Chest", "Machine", "Beginner", False,
             "Maximum stability. Constant tension. Pure horizontal adduction. "
             "Highest SFR chest exercise — almost zero systemic fatigue.",
             False, 10, "constant", "sternal_mid", 10, 0),
            ("Dip (Chest Bias)", "Push", "Chest", "Bodyweight", "Advanced", True,
             "Forward lean + wide grip emphasizes sternal pec fibers. Very high mechanical "
             "tension at bottom of ROM. Shoulder injury risk if ROM is pushed beyond "
             "individual capacity. Low stability = high CNS cost. Moderate SFR.",
             False, 3, "descending", "sternal_lower", 5, 0),

            # ─── BACK ───
            ("Chest-Supported T-Bar Row", "Pull", "Back", "Machine", "Beginner", True,
             "Removes spinal erector demand entirely. All neural drive to lats/rhomboids/traps. "
             "Best SFR back compound by far.",
             True, 9, "bell", "mid_back", 9, 0),
            ("Lat Pulldown (Wide)", "Pull", "Back", "Cable", "Beginner", True,
             "Vertical pull biasing lat width. Pronated wide grip emphasizes upper lat fibers. "
             "Constant tension from cable.",
             True, 8, "constant", "upper_lat", 8, 0),
            ("Lat Pulldown (Neutral Close)", "Pull", "Back", "Cable", "Beginner", True,
             "Neutral grip improves leverage and biases lower lats. Reduced shoulder impingement. "
             "Excellent for lat thickness.",
             True, 8, "constant", "lower_lat", 8, 0),
            ("Seated Cable Row", "Pull", "Back", "Cable", "Beginner", True,
             "Horizontal pull with constant tension. Targets mid-back (rhomboids, mid traps). "
             "Moderate stability demand.",
             True, 7, "constant", "mid_back", 8, 0),
            ("Straight Arm Pulldown", "Pull", "Back", "Cable", "Intermediate", False,
             "Lat isolation with zero bicep involvement. Constant cable tension. "
             "Full ROM through shoulder flexion/extension. High SFR, excellent for "
             "accumulating lat volume without systemic fatigue.",
             False, 9, "constant", "full_lat", 9, 0),
            ("Barbell Bent-Over Row", "Pull", "Back", "Barbell", "Advanced", True,
             "Heavy horizontal pull but imposes massive spinal erector and hamstring isometric demand. "
             "Good for absolute strength but poor SFR for pure back hypertrophy.",
             False, 3, "ascending", "mid_back", 4, 0),
            ("Pull-Up / Chin-Up", "Pull", "Back", "Bodyweight", "Advanced", True,
             "Vertical pull. Chin-up (supinated) involves more biceps. Pull-up (pronated) isolates "
             "lats more. Both require significant strength relative to bodyweight.",
             False, 3, "descending", "upper_lat", 5, 1),

            # ─── QUADS ───
            ("Hack Squat", "Legs", "Quads", "Machine", "Intermediate", True,
             "Fixed path removes balance demand. Allows maximal quad loading without spinal "
             "compression. Vastus lateralis and medialis dominant.",
             False, 9, "ascending", "vastus_group", 8, 0),
            ("Leg Press", "Legs", "Quads", "Machine", "Beginner", True,
             "Highest absolute load quad exercise. Very high stability. Ascending resistance. "
             "Foot placement modulates quad vs glute bias.",
             False, 10, "ascending", "vastus_group", 7, 0),
            ("Leg Extension", "Legs", "Quads", "Machine", "Beginner", False,
             "Only exercise that fully loads the rectus femoris (biarticular quad head) in its "
             "shortened position. Essential for complete quad development. Bell-curve resistance.",
             False, 10, "bell", "rectus_femoris", 10, 0),
            ("Bulgarian Split Squat", "Legs", "Quads", "Dumbbell", "Advanced", True,
             "Unilateral compound. Deep ROM loads VMO and adductors. High stability demand "
             "reduces absolute load but addresses bilateral strength imbalances. Moderate SFR.",
             True, 3, "descending", "vmo_adductor", 5, 0),
            ("Barbell Back Squat", "Legs", "Quads", "Barbell", "Advanced", True,
             "King of compound lifts. High spinal load and CNS demand. Good for overall leg "
             "development but poor SFR if hypertrophy is the only goal. Consider hack squat instead.",
             False, 4, "ascending", "full_quad", 4, 0),
            ("Sissy Squat", "Legs", "Quads", "Bodyweight", "Advanced", False,
             "Extreme knee flexion isolates the rectus femoris through a large ROM. "
             "Very high force demands on the knee joint — requires healthy knees and "
             "progressive loading. Low stability, moderate SFR. Use as an accessory, not a staple.",
             False, 2, "descending", "rectus_femoris", 6, 0),

            # ─── HAMSTRINGS ───
            ("Seated Leg Curl", "Legs", "Hamstrings", "Machine", "Beginner", False,
             "Hip flexion avoids active insufficiency, allowing hamstrings to produce "
             "MORE FORCE per rep than lying curl. This is a biomechanical advantage, not "
             "a stretch effect. Constant tension. Highest SFR hamstring exercise.",
             False, 10, "constant", "full_hamstring", 10, 0),
            ("Lying Leg Curl", "Legs", "Hamstrings", "Machine", "Beginner", False,
             "Hips extended = hamstrings in active insufficiency. Less effective than seated variant "
             "for hypertrophy but still useful. Often cramping occurs due to shortened position.",
             False, 10, "bell", "distal_hamstring", 8, 0),
            ("Romanian Deadlift", "Legs", "Hamstrings", "Barbell", "Advanced", True,
             "Heavy hip hinge loading proximal hamstrings and glutes. High absolute force "
             "production but extremely high systemic cost (spinal erectors, CNS). Poor SFR "
             "for hamstring hypertrophy specifically — seated leg curl produces more hamstring "
             "stimulus per unit of fatigue. Use sparingly, not as a hamstring staple.",
             False, 4, "descending", "proximal_hamstring", 5, 0),
            ("Nordic Hamstring Curl", "Legs", "Hamstrings", "Bodyweight", "Advanced", False,
             "Extreme eccentric overload. Shown to reduce hamstring injury rates (Al Attar 2017). "
             "Very high force demands. Requires significant strength — modify with band assist. "
             "High muscle damage potential — use low volume (2 sets max).",
             False, 3, "descending", "full_hamstring", 5, 0),

            # ─── GLUTES ───
            ("Hip Thrust", "Legs", "Glutes", "Barbell", "Intermediate", True,
             "Peak tension at full hip extension (shortened position). Directly loads glute max. "
             "Ascending resistance matches glute strength curve.",
             False, 7, "ascending", "glute_max", 8, 0),
            ("Cable Pull-Through", "Legs", "Glutes", "Cable", "Beginner", True,
             "Constant tension hip hinge. Loads glutes at long length unlike hip thrust. "
             "Lower absolute load but better length-tension matching.",
             False, 7, "constant", "glute_max", 8, 1),

            # ─── SHOULDERS ───
            ("Cable Lateral Raise", "Push", "Shoulders", "Cable", "Intermediate", False,
             "Constant tension profile vs dumbbells (which have zero tension at bottom). "
             "Superior for lateral deltoid hypertrophy. Use slight forward lean for long head bias.",
             True, 8, "constant", "lateral_delt", 10, 0),
            ("DB Lateral Raise", "Push", "Shoulders", "Dumbbell", "Beginner", False,
             "Bell curve resistance (hardest at 90 degrees). Zero tension at bottom of ROM. "
             "Cable variant is superior but this is more accessible.",
             True, 6, "bell", "lateral_delt", 8, 0),
            ("Face Pull", "Pull", "Shoulders", "Cable", "Beginner", False,
             "Rear deltoid + external rotators. Essential for structural balance against pressing "
             "volume. Constant tension. Very high SFR.",
             False, 8, "constant", "rear_delt", 10, 0),
            ("Overhead Press (DB)", "Push", "Shoulders", "Dumbbell", "Intermediate", True,
             "Anterior deltoid dominant. Greater ROM than barbell. Ascending resistance. "
             "Moderate SFR — imposes significant trap/core stabilization demand.",
             True, 4, "ascending", "anterior_delt", 6, 0),
            ("Reverse Pec Deck", "Pull", "Shoulders", "Machine", "Beginner", False,
             "Maximum stability rear delt isolation. Constant tension. "
             "Highest SFR rear delt exercise. Use for high-frequency rear delt work.",
             False, 10, "constant", "rear_delt", 10, 0),

            # ─── BICEPS ───
            ("Incline Dumbbell Curl", "Pull", "Biceps", "Dumbbell", "Intermediate", False,
             "Shoulder extension avoids active insufficiency on the biceps long head, allowing "
             "it to produce maximal force through full ROM. Superior to standing curls for long "
             "head development due to biomechanical advantage, not 'stretch.'",
             True, 6, "bell", "long_head", 9, 0),
            ("Preacher Curl", "Pull", "Biceps", "Dumbbell", "Beginner", False,
             "Shoulder flexion shortens the long head, biasing the short head. "
             "Peak tension at mid-range. Good for short head width.",
             True, 8, "bell", "short_head", 9, 0),
            ("Cable Curl", "Pull", "Biceps", "Cable", "Beginner", False,
             "Constant tension throughout ROM. Cable direction can be adjusted to modify "
             "the resistance profile (overhead for lengthened, low for shortened emphasis).",
             True, 8, "constant", "full_biceps", 9, 0),

            # ─── TRICEPS ───
            ("Overhead Cable Triceps Extension", "Push", "Triceps", "Cable", "Intermediate", False,
             "Shoulder flexion avoids active insufficiency on the long head (largest triceps "
             "head), allowing it to produce maximal force. Constant cable tension. The single "
             "best triceps exercise for overall mass because it loads the head that pushdowns miss.",
             True, 7, "constant", "long_head", 9, 0),
            ("Cable Pushdown", "Push", "Triceps", "Cable", "Beginner", False,
             "Shoulder in neutral = long head shortened, biasing lateral and medial heads. "
             "Good complement to overhead work but NOT a replacement for it.",
             True, 8, "constant", "lateral_head", 8, 0),
            ("Close-Grip Bench Press", "Push", "Triceps", "Barbell", "Intermediate", True,
             "Compound triceps movement. High absolute load but chest is a significant synergist. "
             "Use for strength; prefer isolations for triceps hypertrophy.",
             False, 7, "ascending", "full_triceps", 5, 0),

            # ─── CALVES ───
            ("Seated Calf Raise", "Legs", "Calves", "Machine", "Beginner", False,
             "Knee flexion shortens the gastrocnemius, isolating the soleus (60% of calf mass). "
             "Essential — standing raises alone miss the soleus.",
             False, 10, "ascending", "soleus", 9, 0),
            ("Standing Calf Raise", "Legs", "Calves", "Machine", "Beginner", False,
             "Knee extension allows gastrocnemius to produce maximal force (avoids active "
             "insufficiency). Full ROM with controlled eccentrics.",
             False, 9, "ascending", "gastrocnemius", 8, 1),

            # ─── CHEST — Angle Variants ───
            ("30° Incline DB Press", "Push", "Chest", "Dumbbell", "Intermediate", True,
             "30-degree incline targets clavicular (upper) pec fibers. DB allows independent "
             "arm movement and greater ROM than barbell. Bell curve resistance with peak "
             "tension at mid-range. Moderate stability demand keeps SFR reasonable.",
             True, 5, "bell", "clavicular_upper", 7, 0),
            ("Low Incline DB Press", "Push", "Chest", "Dumbbell", "Intermediate", True,
             "15-20 degree incline targets the transition zone between upper and mid pec. "
             "Minimal anterior delt involvement compared to steeper inclines. Each arm "
             "works independently through full ROM.",
             True, 5, "bell", "upper_mid_transition", 7, 0),
            ("Decline DB Press", "Push", "Chest", "Dumbbell", "Intermediate", True,
             "Decline angle biases lower sternal pec fibers. Reduced shoulder stress "
             "compared to flat and incline pressing. Good option for lifters with "
             "shoulder impingement issues. Each arm loads independently.",
             True, 5, "bell", "sternal_lower", 7, 0),
            ("Machine Chest Press", "Push", "Chest", "Machine", "Beginner", True,
             "Maximum stability allows full neural drive to pecs with zero stabilizer "
             "demand. Constant tension on most models. One of the highest SFR compound "
             "chest exercises available, minimal systemic fatigue.",
             False, 10, "constant", "sternal_mid", 9, 0),
            ("Smith Machine Incline Press", "Push", "Chest", "Smith Machine", "Intermediate", True,
             "Fixed bar path provides high stability for safe overloading of clavicular "
             "pec fibers. Ascending resistance profile. Good for progressive overload "
             "when a spotter is unavailable.",
             False, 9, "ascending", "clavicular_upper", 7, 0),

            # ─── BACK — More Variants ───
            ("Single Arm Cable Row", "Pull", "Back", "Cable", "Intermediate", True,
             "Unilateral horizontal pull with constant cable tension. Addresses side-to-side "
             "imbalances. Allows slight trunk rotation for full scapular retraction. "
             "Great SFR for mid-back development.",
             True, 7, "constant", "mid_back", 8, 0),
            ("Single Arm Lat Pulldown", "Pull", "Back", "Cable", "Intermediate", True,
             "Unilateral vertical pull addressing lat imbalances. Constant cable tension "
             "through full ROM. Greater ROM per side than bilateral pulldown. Focuses "
             "neural drive on one lat at a time.",
             True, 7, "constant", "full_lat", 8, 0),
            ("Prone Incline DB Row", "Pull", "Back", "Dumbbell", "Beginner", True,
             "Chest supported on incline bench removes lower back and spinal erector demand "
             "entirely. All force production goes to lats, rhomboids, and rear delts. "
             "Very high SFR. Each arm works independently.",
             True, 8, "bell", "mid_back", 9, 0),
            ("Machine Row", "Pull", "Back", "Machine", "Beginner", True,
             "Highest stability horizontal pull. Fixed path removes all stabilizer demand. "
             "Full neural drive to target muscles. Excellent SFR for back hypertrophy, "
             "comparable to chest-supported rows.",
             False, 10, "constant", "mid_back", 9, 0),
            ("Meadows Row", "Pull", "Back", "Barbell", "Advanced", True,
             "Landmine single-arm row with a unique pulling angle. Perpendicular stance "
             "to the bar changes the line of resistance. Targets the lower lats and teres "
             "major from an angle other rows miss. Moderate stability demand.",
             True, 5, "bell", "lower_lat", 6, 0),

            # ─── SHOULDERS — Anterior ───
            ("Machine Shoulder Press", "Push", "Shoulders", "Machine", "Beginner", True,
             "High stability overhead press, primarily anterior delt dominant (60-70% of "
             "force production even with wide grip, per Campos & Silva 2014). Ascending "
             "resistance. Safe overloading without spotter. High SFR for anterior delt.",
             False, 10, "ascending", "anterior_delt", 9, 0),
            ("Smith Machine Overhead Press", "Push", "Shoulders", "Smith Machine", "Intermediate", True,
             "Fixed bar path allows safe heavy overloading of anterior delts. Ascending "
             "resistance. Removes stabilizer demand compared to free weight overhead press. "
             "Good for progressive overload on pressing.",
             False, 9, "ascending", "anterior_delt", 8, 0),
            ("DB Front Raise", "Push", "Shoulders", "Dumbbell", "Beginner", False,
             "Anterior delt isolation. Bell curve resistance. Low priority exercise since "
             "pressing movements already heavily load the anterior delt. Only useful if "
             "anterior delt is specifically lagging.",
             True, 6, "bell", "anterior_delt", 7, 0),

            # ─── SHOULDERS — Medial ───
            ("Machine Lateral Raise", "Push", "Shoulders", "Machine", "Beginner", False,
             "Highest SFR medial delt exercise when available. Constant tension through "
             "full ROM eliminates the dead zone at the bottom that DB lateral raises have. "
             "Maximum stability, zero momentum possible.",
             False, 10, "constant", "lateral_delt", 10, 0),
            ("Behind-the-Body Cable Lateral Raise", "Push", "Shoulders", "Cable", "Intermediate", False,
             "Cable passes behind the body, shifting the resistance curve to load the "
             "medial delt harder at the start of ROM where DB lateral raises have zero "
             "tension. Single arm, constant tension. Superior resistance profile to "
             "standard cable lateral raise.",
             True, 7, "constant", "lateral_delt", 9, 0),

            # ─── SHOULDERS — Posterior ───
            ("Cable Rear Delt Fly", "Pull", "Shoulders", "Cable", "Beginner", False,
             "Constant cable tension throughout ROM for rear delt. When performed with "
             "slight external rotation, minimizes trap dominance and isolates the posterior "
             "deltoid effectively. Can be done single or dual arm.",
             True, 8, "constant", "rear_delt", 9, 0),
            ("Prone Incline Rear Delt Raise", "Pull", "Shoulders", "Dumbbell", "Beginner", False,
             "Chest on incline bench removes momentum and trunk movement entirely. Isolates "
             "rear delt through horizontal abduction. Bell curve resistance. Controlled "
             "tempo eliminates trap compensation.",
             True, 8, "bell", "rear_delt", 9, 0),

            # ─── BICEPS — Cable/EZ Variants ───
            ("EZ Bar Curl", "Pull", "Biceps", "Barbell", "Beginner", False,
             "Cambered bar reduces wrist strain from full supination, allowing heavier "
             "loading than straight bar. Targets both biceps heads. Bell curve resistance "
             "with peak tension at mid-range.",
             False, 6, "bell", "full_biceps", 8, 0),
            ("EZ Bar Cable Curl", "Pull", "Biceps", "Cable", "Beginner", False,
             "EZ attachment on low cable combines wrist-friendly grip angle with constant "
             "cable tension. Eliminates the dead zone at top and bottom that free weight "
             "curls have. Good SFR for biceps mass.",
             False, 8, "constant", "full_biceps", 9, 0),
            ("Hammer Cable Curl", "Pull", "Biceps", "Cable", "Beginner", False,
             "Rope attachment with neutral grip emphasizes brachialis and brachioradialis. "
             "Constant cable tension. Brachialis development pushes the biceps peak up for "
             "a wider arm appearance. Also builds forearm thickness.",
             False, 8, "constant", "brachialis", 9, 0),
            ("Single Arm Cable Curl", "Pull", "Biceps", "Cable", "Beginner", False,
             "Unilateral cable curl with constant tension throughout ROM. Addresses bicep "
             "imbalances. Cable angle can be adjusted to shift peak tension to different "
             "parts of the ROM for varied stimulus.",
             True, 7, "constant", "full_biceps", 9, 0),
            ("Bayesian Cable Curl", "Pull", "Biceps", "Cable", "Intermediate", False,
             "Behind-body single arm cable curl with shoulder in extension. This position "
             "avoids active insufficiency on the biceps long head, allowing maximal force "
             "production through full ROM. Best cable curl variant for long head.",
             True, 7, "constant", "long_head", 9, 0),
            ("Spider Curl", "Pull", "Biceps", "Dumbbell", "Intermediate", False,
             "Chest against incline bench (prone). Bell curve resistance with peak tension "
             "at mid-range. Shoulder flexion shortens the long head, biasing the short head. "
             "Eliminates momentum completely.",
             True, 8, "bell", "short_head", 9, 0),
            ("Concentration Curl", "Pull", "Biceps", "Dumbbell", "Beginner", False,
             "Seated single arm curl braced against inner thigh. Eliminates momentum and "
             "body english entirely. Shoulder flexion biases the short head. Good for "
             "mind-muscle connection and addressing imbalances.",
             True, 7, "bell", "short_head", 8, 0),

            # ─── TRICEPS — More Variants ───
            ("Single Arm Overhead Cable Extension", "Push", "Triceps", "Cable", "Intermediate", False,
             "Unilateral overhead extension addressing tricep imbalances. Shoulder flexion "
             "avoids active insufficiency on the long head, allowing maximal force production "
             "per arm. Constant cable tension throughout ROM.",
             True, 7, "constant", "long_head", 9, 0),
            ("EZ Bar Skull Crusher", "Push", "Triceps", "Barbell", "Intermediate", False,
             "Lying EZ bar extension loading triceps through elbow extension. Loads all "
             "three heads. EZ bar reduces wrist strain. Mild shoulder flexion at bottom "
             "provides some long head contribution. Moderate stability demand.",
             False, 6, "bell", "full_triceps", 7, 0),
            ("Cable Kickback", "Push", "Triceps", "Cable", "Beginner", False,
             "Single arm cable kickback with constant tension. Lateral head emphasis when "
             "performed with full elbow extension against resistance. Useful accessory "
             "for lateral head detail work.",
             True, 7, "constant", "lateral_head", 8, 0),
            ("Triceps Dip Machine", "Push", "Triceps", "Machine", "Beginner", True,
             "High stability dip with adjustable resistance. Reduces shoulder stress "
             "compared to bodyweight dips. Loads all three triceps heads. Ascending "
             "resistance profile. Good SFR for triceps compound work.",
             False, 10, "ascending", "full_triceps", 8, 0),
            ("JM Press", "Push", "Triceps", "Barbell", "Advanced", True,
             "Hybrid between close-grip bench press and skull crusher. Very high force "
             "production on all three triceps heads. Advanced movement requiring excellent "
             "elbow tracking. High mechanical tension per rep.",
             False, 5, "ascending", "full_triceps", 6, 0),
            ("Rope Pushdown", "Push", "Triceps", "Cable", "Beginner", False,
             "Rope attachment allows wrist pronation at the bottom, increasing lateral "
             "head recruitment. Constant cable tension. Shoulder neutral position biases "
             "lateral and medial heads over long head.",
             False, 8, "constant", "lateral_head", 8, 0),

            # ─── LEGS — Machine & Unilateral Variants ───
            ("Pendulum Squat", "Legs", "Quads", "Machine", "Intermediate", True,
             "Machine squat with pendulum arc path. Very high stability removes all balance "
             "demand. Ascending resistance matching quad strength curve. Quad dominant with "
             "minimal spinal load. Better SFR than barbell squat for hypertrophy.",
             False, 9, "ascending", "full_quad", 9, 0),
            ("Belt Squat", "Legs", "Quads", "Machine", "Intermediate", True,
             "Load hangs from a belt at the hips, removing spinal compression entirely. "
             "Quad and glute focus without CNS cost of spinal loading. Excellent SFR. "
             "Allows high quad volume without accumulating back fatigue.",
             False, 8, "ascending", "full_quad", 9, 0),
            ("Single Leg Leg Press", "Legs", "Quads", "Machine", "Intermediate", True,
             "Unilateral leg press addressing quad and glute imbalances between legs. "
             "Machine provides moderate stability while each leg works independently. "
             "High load capacity per leg. Ascending resistance.",
             True, 8, "ascending", "vastus_group", 7, 0),
            ("Single Leg Leg Extension", "Legs", "Quads", "Machine", "Beginner", False,
             "Unilateral leg extension isolating each quad independently. Identifies and "
             "corrects side-to-side strength imbalances. Same bell curve resistance as "
             "bilateral variant. Full rectus femoris loading per leg.",
             True, 10, "bell", "rectus_femoris", 10, 0),
            ("Single Leg Seated Leg Curl", "Legs", "Hamstrings", "Machine", "Beginner", False,
             "Unilateral seated leg curl addressing hamstring imbalances. Seated position "
             "(hips flexed) avoids active insufficiency for maximal hamstring force per rep. "
             "Constant tension. Each leg works independently.",
             True, 10, "constant", "full_hamstring", 10, 0),
            ("Leg Press Calf Raise", "Legs", "Calves", "Machine", "Beginner", False,
             "Calf raises on the leg press platform. Allows very high load capacity. "
             "Straight knee position targets gastrocnemius (avoids active insufficiency). "
             "High stability from the machine sled.",
             False, 9, "ascending", "gastrocnemius", 8, 0),
        ]

        for ex in exercises:
            try:
                cursor.execute("""
                    INSERT INTO exercises (name, category, muscle_group_primary, equipment,
                    difficulty_level, is_compound, instructions, is_unilateral,
                    stability_score, resistance_profile, regional_bias)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, ex[:11])
            except (sqlite3.IntegrityError, sqlite3.OperationalError):
                pass
        self.conn.commit()

    def seed_workout_templates(self):
        """
        Evidence-based protocols designed around hypertrophy PRINCIPLES, not body parts.

        Design rationale:
        - Exercise selection prioritizes HIGH SFR movements (machines/cables over
          free weights where possible — Beardsley 2020)
        - Force production optimization: exercises chosen for maximal motor unit
          recruitment via biomechanical advantage (avoiding active insufficiency)
        - 2-3 hard sets per exercise (Krieger 2010 dose-response — diminishing
          returns beyond set 3 in a single session)
        - Regional coverage: each muscle hit from multiple force-length positions
          across the week (not in one session)
        - Active insufficiency avoided: seated leg curl over lying, overhead
          triceps over pushdown for long head
        - Systemic fatigue managed: high-SFR isolations placed after compounds,
          no redundant compound stacking (e.g., no squat + leg press same day)
        - Volume distributed across frequency (2x/week per muscle) rather than
          crammed into one giant session
        """
        cursor = self.conn.cursor()
        count = cursor.execute("SELECT count(*) FROM workout_templates WHERE user_id IS NULL").fetchone()[0]
        if count > 0:
            return

        def get_ex_id(name):
            row = cursor.execute("SELECT id FROM exercises WHERE name = ?", (name,)).fetchone()
            return row[0] if row else None

        templates = [
            # ─────────────────────────────────────────────
            # PROTOCOL 1: Upper A — Compound Force Production
            # ─────────────────────────────────────────────
            ("Upper A — Max Force Compounds",
             "PRINCIPLE: Mechanical tension through high-force compound movements. "
             "Flat DB bench allows full ROM with high load and explosive concentric "
             "intent. Chest-supported row removes spinal erector bottleneck so all "
             "force goes to lats/rhomboids. Overhead DB press for anterior delts — "
             "moderate stability demand but high force output. Incline curl avoids "
             "biceps long head active insufficiency. Overhead triceps extension lets "
             "the long head produce maximal force (shoulder flexion avoids active "
             "insufficiency). 2-3 hard sets, explosive concentrics, 0-2 RIR.",
             [
                 ("Flat DB Bench Press", 3),
                 ("Chest-Supported T-Bar Row", 3),
                 ("Overhead Press (DB)", 3),
                 ("Cable Lateral Raise", 3),
                 ("Incline Dumbbell Curl", 3),
                 ("Overhead Cable Triceps Extension", 3),
             ]),

            # ─────────────────────────────────────────────
            # PROTOCOL 2: High-SFR Upper B
            # ─────────────────────────────────────────────
            ("Upper B — High SFR / Low Fatigue",
             "PRINCIPLE: Maximize stimulus-to-fatigue ratio. Every movement is machine "
             "or cable — maximal stability means 100% of neural drive goes to target "
             "muscles with near-zero systemic cost. This session can be recovered from "
             "quickly, enabling higher weekly frequency. Pec deck for chest (constant "
             "tension, zero joint stress). Lat pulldown for width. Cable row for "
             "thickness. Cable lateral raise for side delts (superior resistance "
             "profile vs DB — no dead zone at bottom). Preacher curl for short head. "
             "Cable pushdown for lateral/medial heads. 2-3 sets, 0-1 RIR.",
             [
                 ("Pec Deck / Machine Fly", 3),
                 ("Lat Pulldown (Neutral Close)", 3),
                 ("Seated Cable Row", 3),
                 ("Cable Lateral Raise", 3),
                 ("Reverse Pec Deck", 3),
                 ("Preacher Curl", 2),
                 ("Cable Pushdown", 2),
             ]),

            # ─────────────────────────────────────────────
            # PROTOCOL 3: Lower A — Quad/Glute Bias
            # ─────────────────────────────────────────────
            ("Lower A — Quad & Glute Focus",
             "PRINCIPLE: Quad hypertrophy requires both the vastus group AND the rectus "
             "femoris, which are loaded differently. Hack squat (ascending resistance, "
             "high stability) for vastus lateralis/medialis — NOT barbell squat, because "
             "the squat's limiting factor is spinal erector fatigue, not quad stimulus. "
             "Leg extension for rectus femoris (biarticular head that only gets fully "
             "loaded in open-chain knee extension). Hip thrust for glutes at shortened "
             "position (ascending profile matches glute strength curve). Seated calf "
             "raise for soleus (60% of calf mass, missed by standing raises). "
             "2-3 sets, 1-2 RIR on compounds, 0-1 RIR on isolations.",
             [
                 ("Hack Squat", 3),
                 ("Leg Extension", 3),
                 ("Hip Thrust", 3),
                 ("Seated Leg Curl", 3),
                 ("Seated Calf Raise", 3),
             ]),

            # ─────────────────────────────────────────────
            # PROTOCOL 4: Lower B — Hamstring/Posterior Bias
            # ─────────────────────────────────────────────
            ("Lower B — Posterior Chain & Force Optimization",
             "PRINCIPLE: Hamstring hypertrophy requires BOTH knee flexion AND hip "
             "extension exercises. The key nuance is FORCE PRODUCTION, not length. "
             "Seated leg curl (hips flexed) avoids active insufficiency — the muscle "
             "can contract HARDER, producing more force per rep than lying curl. "
             "RDL is a high-force hip hinge but costs massive systemic fatigue "
             "(spinal erectors) — placed FIRST when CNS is fresh, limited to 2-3 "
             "sets. Leg press for quad maintenance. Standing calf raise for "
             "gastrocnemius (straight knee avoids active insufficiency). Cable "
             "pull-through for glutes with constant tension.",
             [
                 ("Romanian Deadlift", 3),
                 ("Seated Leg Curl", 3),
                 ("Leg Press", 3),
                 ("Cable Pull-Through", 2),
                 ("Standing Calf Raise", 3),
             ]),

            # ─────────────────────────────────────────────
            # PROTOCOL 5: Shoulder/Arm Specialization
            # ─────────────────────────────────────────────
            ("Arms & Delts — Regional Force Coverage",
             "PRINCIPLE: Regional hypertrophy via maximizing force production for each "
             "muscle head. Incline curl puts the biceps long head in a position where "
             "it can contract hardest (avoids active insufficiency). Preacher curl "
             "biases the short head. Overhead triceps extension lets the long head "
             "produce maximal force (pushdowns cannot do this — the long head is in "
             "active insufficiency with shoulder neutral). Cable lateral raises have "
             "constant tension — DB raises have zero resistance in the bottom 40% of "
             "ROM, wasting half the set. Rear delts need direct work (reverse pec deck) "
             "because they get minimal activation from rows despite common belief. All "
             "isolations, all high SFR, 0-1 RIR. Explosive concentrics on every rep.",
             [
                 ("Incline Dumbbell Curl", 3),
                 ("Preacher Curl", 2),
                 ("Overhead Cable Triceps Extension", 3),
                 ("Cable Pushdown", 2),
                 ("Cable Lateral Raise", 3),
                 ("Reverse Pec Deck", 3),
                 ("Face Pull", 2),
             ]),

            # ─────────────────────────────────────────────
            # PROTOCOL 6: Full Body — Minimum Effective Volume
            # ─────────────────────────────────────────────
            ("Full Body MEV — 3x/Week",
             "PRINCIPLE: Minimum Effective Volume (Israetel). Every muscle hit with "
             "just 2 sets per session, 3x/week = 6 sets/week/muscle. This is the floor "
             "for measurable hypertrophy. Each exercise is the single HIGHEST SFR "
             "option for its target muscle — machines and cables only. Zero free weights, "
             "zero wasted recovery capacity. This is not a 'light' session — each set "
             "is taken to 0-1 RIR. The low volume per session means recovery is fast, "
             "enabling the high frequency that keeps MPS elevated more often "
             "(Schoenfeld 2016). Ideal for beginners, deload phases, or resensitization "
             "blocks where you want to restore mTOR sensitivity.",
             [
                 ("Hack Squat", 2),
                 ("Seated Leg Curl", 2),
                 ("Pec Deck / Machine Fly", 2),
                 ("Lat Pulldown (Wide)", 2),
                 ("Cable Lateral Raise", 2),
                 ("Seated Calf Raise", 2),
             ]),

            # ─────────────────────────────────────────────
            # PROTOCOL 7: Chest/Back Antagonist Pairing
            # ─────────────────────────────────────────────
            ("Chest + Back — Antagonist Supersets",
             "PRINCIPLE: Antagonist pairing. Chest and back are biomechanical opposites "
             "— training one potentiates the other via reciprocal inhibition. Supersetting "
             "a press with a row (30-60s rest between) maintains performance on both "
             "while cutting session time nearly in half. Exercise selection: pec deck "
             "paired with chest-supported row (both high stability, constant tension). "
             "Incline DB press paired with lat pulldown (both target upper fibers). "
             "Cable fly paired with straight-arm pulldown (both high-SFR isolations). "
             "No barbell movements — stability is intentionally maximized so you can "
             "push hard on both without CNS bottleneck.",
             [
                 ("Pec Deck / Machine Fly", 3),
                 ("Chest-Supported T-Bar Row", 3),
                 ("Incline DB Press", 3),
                 ("Lat Pulldown (Wide)", 3),
                 ("Cable Fly (High-to-Low)", 2),
                 ("Straight Arm Pulldown", 2),
             ]),

            # ─────────────────────────────────────────────
            # PROTOCOL 8: Accumulation Block — Volume Push
            # ─────────────────────────────────────────────
            ("Accumulation Block — Upper (Wk 3-4 Meso)",
             "PRINCIPLE: Periodized mesocycle. This is the HIGH VOLUME template for "
             "weeks 3-4 of an accumulation block, when you are pushing toward MRV. "
             "3 sets per exercise across 7 exercises = 21 sets upper body in one session. "
             "This is NOT sustainable long-term — it is designed to be run for 1-2 weeks "
             "before a deload drops volume to ~50%. Exercise selection: same high-SFR "
             "movements from Upper A/B but combined into one dense session. RIR 1-2 to "
             "manage fatigue at this volume. If performance drops (weight/reps decline), "
             "you have overshot MRV and need to deload immediately.",
             [
                 ("Pec Deck / Machine Fly", 3),
                 ("Incline DB Press", 3),
                 ("Lat Pulldown (Neutral Close)", 3),
                 ("Chest-Supported T-Bar Row", 3),
                 ("Cable Lateral Raise", 3),
                 ("Incline Dumbbell Curl", 3),
                 ("Overhead Cable Triceps Extension", 3),
             ]),
        ]

        for name, desc, exercises in templates:
            cursor.execute(
                "INSERT INTO workout_templates (user_id, name, description) VALUES (NULL, ?, ?)",
                (name, desc))
            tid = cursor.lastrowid
            for order, (ex_name, sets) in enumerate(exercises):
                ex_id = get_ex_id(ex_name)
                if ex_id:
                    cursor.execute(
                        "INSERT INTO template_exercises (template_id, exercise_id, order_index, target_sets) "
                        "VALUES (?, ?, ?, ?)", (tid, ex_id, order, sets))

        self.conn.commit()

    def get_workout_templates(self, user_id=None):
        """Fetch system templates and user-created ones"""
        try:
            if user_id:
                rows = self.conn.execute("SELECT * FROM workout_templates WHERE user_id IS NULL OR user_id=?", (user_id,)).fetchall()
            else:
                rows = self.conn.execute("SELECT * FROM workout_templates WHERE user_id IS NULL").fetchall()
            return [dict(row) for row in rows]
        except Exception:
            return []

    def get_template_details(self, template_id):
        """Fetch all exercises for a specific folder/template"""
        query = """
            SELECT te.*, e.name, e.is_unilateral 
            FROM template_exercises te
            JOIN exercises e ON te.exercise_id = e.id
            WHERE te.template_id = ?
            ORDER BY te.order_index ASC
        """
        try:
            rows = self.conn.execute(query, (template_id,)).fetchall()
            return [dict(row) for row in rows]
        except Exception:
            return []

    def create_custom_template(self, user_id, name, exercises_data):
        """Saves a protocol/folder of exercises as a routine"""
        cursor = self.conn.cursor()
        cursor.execute("INSERT INTO workout_templates (user_id, name, description) VALUES (?, ?, ?)",
                       (user_id, name, "Custom Protocol"))
        tid = cursor.lastrowid
        for i, ex in enumerate(exercises_data):
            cursor.execute("""
                INSERT INTO template_exercises (template_id, exercise_id, order_index, target_sets)
                VALUES (?, ?, ?, ?)
            """, (tid, ex['id'], i, ex['sets']))
        self.conn.commit()