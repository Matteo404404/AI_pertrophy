"""
Database seeder for exercises and workout templates.
"""

import sqlite3
import os

def seed_database():
    db_path = "data/users.db" 
    if not os.path.exists(db_path):
        if os.path.exists("app/data/users.db"):
            db_path = "app/data/users.db"
        else:
            os.makedirs("data", exist_ok=True)

    print(f"🔌 Connecting to database at: {db_path}")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 1. ROBUST SCHEMA MIGRATION
    new_columns =[
        ("is_unilateral", "BOOLEAN DEFAULT 0"),
        ("resistance_profile", "TEXT"),
        ("stability_score", "INTEGER"),
        ("lengthened_bias", "BOOLEAN"),
        ("regional_bias", "TEXT")
    ]

    for col_name, col_type in new_columns:
        try:
            cursor.execute(f"ALTER TABLE exercises ADD COLUMN {col_name} {col_type}")
        except: pass

    # Ensure template tables exist
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS workout_templates (
            id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, name TEXT, description TEXT
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS template_exercises (
            id INTEGER PRIMARY KEY AUTOINCREMENT, template_id INTEGER, exercise_id INTEGER,
            order_index INTEGER, target_sets INTEGER
        )
    """)

    # 2. CLEAR OLD DATA
    print("🧹 Cleaning old data...")
    cursor.execute("DELETE FROM template_exercises")
    cursor.execute("DELETE FROM workout_templates")
    cursor.execute("DELETE FROM exercises")
    conn.commit()

    # 3. INJECT EXERCISES FIRST
    science_data =[
        ("Incline DB Press (30°)", "Push", "Chest", "Dumbbell", "Intermediate", 1, "<b>Biomechanics:</b> The 30° incline aligns the humerus with the clavicular fibers.", 1, "Bell Curve", 7, 0, "Clavicular Pectoralis"),
        ("Machine Chest Press (Converging)", "Push", "Chest", "Machine", "Beginner", 1, "<b>Biomechanics:</b> High stability removes stabilization demands.", 1, "Descending", 10, 0, "Sternal/Costal Pectoralis"),
        ("Cable Fly (Crossover)", "Push", "Chest", "Cable", "Intermediate", 0, "<b>Biomechanics:</b> Cables provide constant tension.", 1, "Constant Tension", 8, 1, "Sternal Pectoralis"),
        ("Iliac Lat Pulldown (Single Arm)", "Pull", "Back", "Cable", "Advanced", 0, "<b>Biomechanics:</b> Biases the Iliac (Lower) Lats.", 1, "Descending", 8, 1, "Iliac Lats"),
        ("Chest-Supported Row", "Pull", "Back", "Machine", "Beginner", 1, "<b>Biomechanics:</b> Removing lower back limits increases output.", 1, "Ascending", 10, 0, "Upper Back/Traps"),
        ("Leg Extension", "Legs", "Quads", "Machine", "Beginner", 0, "<b>Biomechanics:</b> The ONLY exercise loading Rectus Femoris in shortened position.", 1, "Ascending", 10, 0, "Rectus Femoris"),
        ("Hack Squat", "Legs", "Quads", "Machine", "Intermediate", 1, "<b>Biomechanics:</b> Superior to barbell squats for hypertrophy due to stability.", 0, "Linear", 10, 1, "Vastus Lateralis"),
        ("Seated Leg Curl", "Legs", "Hamstrings", "Machine", "Beginner", 0, "<b>Science:</b> Hip flexion puts hamstrings at optimal length.", 1, "Descending", 9, 1, "Total Hamstrings"),
        ("Romanian Deadlift", "Legs", "Hamstrings", "Barbell", "Advanced", 1, "<b>Biomechanics:</b> Pure hip hinge. Massive mechanical tension.", 0, "Lengthened Overload", 6, 1, "Proximal Hamstrings"),
        ("Cable Lateral Raise (Behind)", "Push", "Shoulders", "Cable", "Intermediate", 0, "<b>Biomechanics:</b> Cable behind back keeps tension in the first 30°.", 1, "Constant", 8, 1, "Lateral Deltoid")
    ]
    print("🧠 Injecting Exercises...")
    cursor.executemany("INSERT INTO exercises (name, category, muscle_group_primary, equipment, difficulty_level, is_compound, instructions, is_unilateral, resistance_profile, stability_score, lengthened_bias, regional_bias) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", science_data)
    conn.commit()

    # 4. INJECT TEMPLATES SECOND
    print("📋 Injecting Evidence-Based Protocols...")
    cursor.execute("INSERT INTO workout_templates (name, description) VALUES ('Schoenfeld Essentials', 'Scientific 3-day full body split focusing on mechanical tension.')")
    t1_id = cursor.lastrowid
    
    essential_exercises =[
        ("Hack Squat", 3),
        ("Machine Chest Press (Converging)", 3),
        ("Chest-Supported Row", 3),
        ("Romanian Deadlift", 3),
        ("Seated Leg Curl", 2)
    ]
    
    for i, (name, sets) in enumerate(essential_exercises):
        ex_id_res = cursor.execute("SELECT id FROM exercises WHERE name=?", (name,)).fetchone()
        if ex_id_res:
            cursor.execute("INSERT INTO template_exercises (template_id, exercise_id, order_index, target_sets) VALUES (?, ?, ?, ?)", 
                           (t1_id, ex_id_res[0], i, sets))

    conn.commit()
    conn.close()
    print("✨ DATABASE UPGRADE COMPLETE.")

if __name__ == "__main__":
    seed_database()