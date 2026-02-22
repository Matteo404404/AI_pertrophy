"""
SCIENTIFIC HYPERTROPHY DATABASE SEEDER v2 (FIXED)
- Robust schema migration (adds columns one by one)
- Injects advanced neuromechanical data
"""

import sqlite3
import os

def seed_database():
    # Ensure we look in the right place for the DB
    db_path = "data/users.db" 
    
    # If not found, try looking inside app/data just in case
    if not os.path.exists(db_path):
        if os.path.exists("app/data/users.db"):
            db_path = "app/data/users.db"
        else:
            # Fallback: create in data if it doesn't exist
            os.makedirs("data", exist_ok=True)

    print(f"🔌 Connecting to database at: {db_path}")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 1. ROBUST SCHEMA MIGRATION
    # We add columns one by one. If they exist, we ignore the error.
    new_columns = [
        ("is_unilateral", "BOOLEAN DEFAULT 0"),
        ("resistance_profile", "TEXT"),
        ("stability_score", "INTEGER"),
        ("lengthened_bias", "BOOLEAN"),
        ("regional_bias", "TEXT")
    ]

    print("🛠️  Updating Schema...")
    for col_name, col_type in new_columns:
        try:
            cursor.execute(f"ALTER TABLE exercises ADD COLUMN {col_name} {col_type}")
            print(f"   ✅ Added column: {col_name}")
        except sqlite3.OperationalError as e:
            if "duplicate column name" in str(e):
                print(f"   🔹 Column '{col_name}' already exists (Skipping)")
            else:
                print(f"   ⚠️ Error adding '{col_name}': {e}")

    # 2. CLEAR OLD DATA to ensure clean state
    print("🧹 Cleaning old exercise data...")
    cursor.execute("DELETE FROM exercises")

    # 3. INJECT THE SCIENCE
    science_data = [
        # --- CHEST ---
        ("Incline DB Press (30°)", "Push", "Chest", "Dumbbell", "Intermediate", 1, 
         "<b>Biomechanics:</b> The 30° incline aligns the humerus with the clavicular fibers. <br><b>Cue:</b> Keep elbows at 45° to maximize moment arm for the chest.", 
         1, "Bell Curve", 7, 0, "Clavicular Pectoralis"),

        ("Machine Chest Press (Converging)", "Push", "Chest", "Machine", "Beginner", 1, 
         "<b>Biomechanics:</b> High stability removes stabilization demands, allowing 100% output. Converging path shortens pecs fully.", 
         1, "Descending", 10, 0, "Sternal/Costal Pectoralis"),
         
        ("Cable Fly (Crossover)", "Push", "Chest", "Cable", "Intermediate", 0, 
         "<b>Biomechanics:</b> Cables provide constant tension. Set at shoulder height to bias sternal fibers.<br><b>Focus:</b> Deep stretch.", 
         1, "Constant Tension", 8, 1, "Sternal Pectoralis"),

        # --- BACK ---
        ("Iliac Lat Pulldown (Single Arm)", "Pull", "Back", "Cable", "Advanced", 0, 
         "<b>Biomechanics:</b> Biases the Iliac (Lower) Lats. Lateral spinal flexion maximizes shortening.", 
         1, "Descending", 8, 1, "Iliac Lats"),

        ("Chest-Supported Row", "Pull", "Back", "Machine", "Beginner", 1, 
         "<b>Biomechanics:</b> Removing lower back limits increases output. Elbows 45° for Upper Back, Tucked for Lats.", 
         1, "Ascending", 10, 0, "Upper Back/Traps"),

        # --- QUADS ---
        ("Leg Extension", "Legs", "Quads", "Machine", "Beginner", 0, 
         "<b>Biomechanics:</b> The ONLY exercise loading Rectus Femoris in shortened position. Essential for full development.", 
         1, "Ascending", 10, 0, "Rectus Femoris"),

        ("Hack Squat", "Legs", "Quads", "Machine", "Intermediate", 1, 
         "<b>Biomechanics:</b> Superior to barbell squats for hypertrophy due to stability. Allows deep knee flexion.", 
         0, "Linear", 10, 1, "Vastus Lateralis"),

        # --- HAMSTRINGS ---
        ("Seated Leg Curl", "Legs", "Hamstrings", "Machine", "Beginner", 0, 
         "<b>Science:</b> Hip flexion puts hamstrings at optimal length. Research shows greater hypertrophy than lying curls.", 
         1, "Descending", 9, 1, "Total Hamstrings"),

        ("Romanian Deadlift", "Legs", "Hamstrings", "Barbell", "Advanced", 1, 
         "<b>Biomechanics:</b> Pure hip hinge. Massive mechanical tension in lengthened position.<br><b>Warning:</b> High fatigue cost.", 
         0, "Lengthened Overload", 6, 1, "Proximal Hamstrings"),

        # --- SHOULDERS ---
        ("Cable Lateral Raise (Behind)", "Push", "Shoulders", "Cable", "Intermediate", 0, 
         "<b>Biomechanics:</b> Cable behind back keeps tension in the first 30° of abduction. Overloads the lengthened range.", 
         1, "Constant", 8, 1, "Lateral Deltoid"),
    ]

    print("🧠 Injecting Neuromechanical Data...")
    
    try:
        cursor.executemany("""
            INSERT INTO exercises (name, category, muscle_group_primary, equipment, 
            difficulty_level, is_compound, instructions, is_unilateral, 
            resistance_profile, stability_score, lengthened_bias, regional_bias)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, science_data)
        conn.commit()
        print(f"✅ Successfully inserted {len(science_data)} scientific exercises.")
    except Exception as e:
        print(f"❌ Insert Error: {e}")

    conn.close()
    print("✨ DATABASE UPGRADE COMPLETE.")

if __name__ == "__main__":
    seed_database()