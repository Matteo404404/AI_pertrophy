"""
Scientific Hypertrophy Trainer - Tracking Interface v3.3
- Fixed: Hard import for ExerciseLibraryDialog
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame, 
    QTabWidget, QCalendarWidget, QDoubleSpinBox, QSpinBox, QTextEdit,
    QComboBox, QGroupBox, QScrollArea, QMessageBox, QTimeEdit, 
    QTableWidget, QTableWidgetItem, QHeaderView, QDialog, QCheckBox,
    QGridLayout, QSizePolicy
)
from PyQt6.QtCore import Qt, QDate
from datetime import date
import numpy as np
import json

# Matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# --- STRICT IMPORT: EXERCISE LIBRARY ---
# This ensures we see the error immediately if the file is missing
from gui.exercise_library import ExerciseLibraryDialog

# AI Engine
try:
    from ml_engine.inference.hybrid_predictor import HybridPredictor
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False


# --- MICRONUTRIENT CONFIG ---
MICROS_CONFIG = {
    "Vitamins (Fat Soluble)": [
        ("Vitamin A", "IU"), ("Vitamin D3", "IU"), ("Vitamin E", "mg"), ("Vitamin K1/K2", "mcg")
    ],
    "Vitamins (Water Soluble)": [
        ("Vitamin C", "mg"), ("Thiamin (B1)", "mg"), ("Riboflavin (B2)", "mg"), 
        ("Niacin (B3)", "mg"), ("B6", "mg"), ("Folate (B9)", "mcg"), ("B12", "mcg")
    ],
    "Minerals (Macro)": [
        ("Calcium", "mg"), ("Magnesium", "mg"), ("Potassium", "mg"), ("Sodium", "mg"), ("Chloride", "mg")
    ],
    "Minerals (Trace)": [
        ("Iron", "mg"), ("Zinc", "mg"), ("Copper", "mg"), ("Selenium", "mcg"), ("Iodine", "mcg")
    ],
    "Other": [
        ("Water", "L"), ("Fiber", "g"), ("Sat. Fat", "g"), ("Omega-3", "g"), ("Cholesterol", "mg")
    ]
}

class TrackingWidget(QWidget):
    def __init__(self, db_manager, tracking_system, user_manager):
        super().__init__()
        self.db = db_manager
        self.user_manager = user_manager
        self.selected_date = date.today()
        self.current_session_sets = []
        self.micro_inputs = {} 
        
        if ML_AVAILABLE:
            model_path = "ml_engine/models/strength_predictor.pt"
            self.ai_predictor = HybridPredictor(self.db, model_path)
        else:
            self.ai_predictor = None
            
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)
        
        # Header
        header = QLabel("Scientific Tracking Console")
        header.setStyleSheet("font-size: 28px; font-weight: 800; color: #89b4fa; letter-spacing: 1px;")
        main_layout.addWidget(header)
        
        content_layout = QHBoxLayout()
        
        # Calendar (Left Sidebar)
        self.create_calendar_section(content_layout)
        
        # Tabs (Right Content)
        self.tabs = QTabWidget()
        self.tabs.addTab(self.create_diet_tab(), "🧪 Nutrition")
        self.tabs.addTab(self.create_workout_tab(), "🏋️ Hypertrophy") 
        self.tabs.addTab(self.create_sleep_tab(), "😴 Recovery")
        
        content_layout.addWidget(self.tabs, 1)
        main_layout.addLayout(content_layout)

    def create_calendar_section(self, parent_layout):
        frame = QFrame()
        frame.setFixedWidth(320)
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(15, 20, 15, 20)
        
        self.calendar = QCalendarWidget()
        self.calendar.setSelectedDate(QDate.currentDate())
        self.calendar.clicked.connect(self.on_date_selected)
        layout.addWidget(self.calendar)
        
        self.date_info_label = QLabel(f"{date.today().strftime('%A, %B %d')}")
        self.date_info_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #fab387; margin-top: 15px;")
        self.date_info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.date_info_label)
        
        today_btn = QPushButton("Jump to Today")
        today_btn.setProperty("class", "action_btn")
        today_btn.clicked.connect(self.jump_to_today)
        layout.addWidget(today_btn)
        
        layout.addStretch()
        parent_layout.addWidget(frame)

    # ==========================================
    # 🧪 ADVANCED NUTRITION TAB
    # ==========================================
    def create_diet_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        
        form_widget = QWidget()
        form_layout = QVBoxLayout(form_widget)
        form_layout.setSpacing(25)
        form_layout.setContentsMargins(10, 10, 20, 10)

        # 1. Macros
        macros_group = QGroupBox("Macronutrients")
        grid = QGridLayout(macros_group)
        grid.setSpacing(15)
        
        self.diet_protein = self._create_macro_input("Protein", "g", grid, 0, 0)
        self.diet_carbs = self._create_macro_input("Carbs", "g", grid, 0, 2)
        self.diet_fats = self._create_macro_input("Fats", "g", grid, 1, 0)
        
        # Total Calories
        self.diet_calories = QSpinBox()
        self.diet_calories.setRange(0, 10000)
        self.diet_calories.setSuffix(" kcal")
        self.diet_calories.setStyleSheet("font-size: 16px; color: #a6e3a1; font-weight: bold;")
        
        lbl = QLabel("🔥 Total Energy:")
        lbl.setStyleSheet("font-size: 14px; font-weight: bold; color: #a6e3a1;")
        grid.addWidget(lbl, 1, 2)
        grid.addWidget(self.diet_calories, 1, 3)
        
        form_layout.addWidget(macros_group)

        # 2. Dynamic Micros
        for category, items in MICROS_CONFIG.items():
            group = QGroupBox(category)
            g_layout = QGridLayout(group)
            g_layout.setSpacing(10)
            
            row, col = 0, 0
            for name, unit in items:
                g_layout.addWidget(QLabel(name + ":"), row, col)
                inp = QDoubleSpinBox()
                inp.setRange(0, 50000)
                inp.setSuffix(f" {unit}")
                inp.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.NoButtons)
                
                self.micro_inputs[name] = inp
                g_layout.addWidget(inp, row, col + 1)
                
                col += 2
                if col >= 6:
                    col = 0
                    row += 1
            form_layout.addWidget(group)
        
        save_btn = QPushButton("💾 Save Nutrition Log")
        save_btn.setProperty("class", "success_btn")
        save_btn.setMinimumHeight(50)
        save_btn.clicked.connect(self.save_diet_entry)
        form_layout.addWidget(save_btn)
        
        scroll.setWidget(form_widget)
        layout.addWidget(scroll)
        return tab

    def _create_macro_input(self, label, suffix, layout, row, col):
        lbl = QLabel(f"{label}:")
        inp = QDoubleSpinBox()
        inp.setRange(0, 1000)
        inp.setSuffix(f" {suffix}")
        inp.valueChanged.connect(self.auto_calculate_calories)
        layout.addWidget(lbl, row, col)
        layout.addWidget(inp, row, col + 1)
        return inp

    def auto_calculate_calories(self):
        p = self.diet_protein.value() * 4
        c = self.diet_carbs.value() * 4
        f = self.diet_fats.value() * 9
        self.diet_calories.setValue(int(p + c + f))

    # ==========================================
    # 🏋️ WORKOUT TAB
    # ==========================================
    def create_workout_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # 1. Exercise Selector
        sel_frame = QFrame()
        sel_frame.setStyleSheet("background: #262639; border-radius: 8px; padding: 15px; border: 1px solid #313244;")
        sel_layout = QHBoxLayout(sel_frame)
        
        # Display selected exercise
        self.lbl_selected_exercise = QLabel("No Exercise Selected")
        self.lbl_selected_exercise.setStyleSheet("font-size: 18px; font-weight: bold; color: #89b4fa;")
        
        # The Library Button
        self.btn_library = QPushButton("📖 Open Exercise Library")
        self.btn_library.setProperty("class", "action_btn")
        self.btn_library.clicked.connect(self.open_exercise_library)
        
        sel_layout.addWidget(self.lbl_selected_exercise)
        sel_layout.addStretch()
        sel_layout.addWidget(self.btn_library)
        layout.addWidget(sel_frame)
        
        # Store current ID
        self.current_exercise_id = None


        # Input
        input_group = QGroupBox("Set Data")
        grid = QGridLayout(input_group)
        
        self.weight_input = QDoubleSpinBox()
        self.weight_input.setRange(0, 500)
        self.weight_input.setSuffix(" kg")
        grid.addWidget(QLabel("Load:"), 0, 0)
        grid.addWidget(self.weight_input, 0, 1)
        
        self.reps_input = QSpinBox()
        self.reps_input.setRange(0, 100)
        grid.addWidget(QLabel("Reps:"), 0, 2)
        grid.addWidget(self.reps_input, 0, 3)
        
        self.rir_input = QSpinBox()
        self.rir_input.setRange(0, 10)
        grid.addWidget(QLabel("RIR:"), 0, 4)
        grid.addWidget(self.rir_input, 0, 5)
        
        self.unilateral_check = QCheckBox("Unilateral (Single Limb)")
        grid.addWidget(self.unilateral_check, 1, 0, 1, 2)
        
        add_btn = QPushButton("Add Set")
        add_btn.setProperty("class", "action_btn")
        add_btn.clicked.connect(self.add_set_to_list)
        grid.addWidget(add_btn, 1, 4, 1, 2)
        
        layout.addWidget(input_group)
        
        # Table
        self.sets_table = QTableWidget()
        self.sets_table.setColumnCount(5)
        self.sets_table.setHorizontalHeaderLabels(["Exercise", "Load", "Reps", "RIR", "Mode"])
        self.sets_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.sets_table)
        
        # Footer
        btn_row = QHBoxLayout()
        save_btn = QPushButton("✅ Commit Session")
        save_btn.setProperty("class", "success_btn")
        save_btn.clicked.connect(self.save_workout_session)
        btn_row.addWidget(save_btn)
        
        self.predict_btn = QPushButton("🧠 AI Trajectory Analysis")
        self.predict_btn.setProperty("class", "action_btn")
        self.predict_btn.clicked.connect(self.on_predict_click)
        btn_row.addWidget(self.predict_btn)
        
        layout.addLayout(btn_row)
        return tab

    # ==========================================
    # 😴 SLEEP TAB (RESTORED & STYLED)
    # ==========================================
    def create_sleep_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        form_widget = QWidget()
        form_layout = QVBoxLayout(form_widget)
        
        group = QGroupBox("Sleep Metrics")
        grid = QGridLayout(group)
        
        self.sleep_duration = QDoubleSpinBox()
        self.sleep_duration.setRange(0, 24)
        self.sleep_duration.setSuffix(" hrs")
        grid.addWidget(QLabel("Duration:"), 0, 0)
        grid.addWidget(self.sleep_duration, 0, 1)
        
        self.sleep_quality = QSpinBox()
        self.sleep_quality.setRange(1, 10)
        grid.addWidget(QLabel("Quality (1-10):"), 1, 0)
        grid.addWidget(self.sleep_quality, 1, 1)
        
        form_layout.addWidget(group)
        
        save_btn = QPushButton("Save Recovery Data")
        save_btn.setProperty("class", "success_btn")
        save_btn.clicked.connect(self.save_sleep_entry)
        form_layout.addWidget(save_btn)
        
        form_layout.addStretch()
        scroll.setWidget(form_widget)
        layout.addWidget(scroll)
        return tab

    # ==========================================
    # CRITICAL: DATA HANDLING METHODS
    # ==========================================
    def refresh_data(self):
        """CRITICAL FIX: Called by MainWindow to refresh data when tab is opened"""
        self.load_existing_data()

    def load_existing_data(self):
        """Loads Diet, Sleep, and Workouts for selected date"""
        self.clear_all_forms() # Reset first
        
        user = self.user_manager.get_current_user()
        if not user: return
        date_str = self.selected_date.strftime('%Y-%m-%d')
        cursor = self.db.conn.cursor()
        
        # 1. Load Diet
        row = cursor.execute("SELECT * FROM diet_entries WHERE user_id=? AND entry_date=?", (user['id'], date_str)).fetchone()
        if row:
            self.diet_calories.setValue(row['total_calories'] or 0)
            self.diet_protein.setValue(row['protein_g'] or 0)
            self.diet_carbs.setValue(row['carbs_g'] or 0)
            self.diet_fats.setValue(row['fats_g'] or 0)
            
            # Load Micros from JSON if present
            if row['notes'] and "MICROS_DATA::" in row['notes']:
                try:
                    data = json.loads(row['notes'].split("MICROS_DATA::")[1])
                    for k, v in data.items():
                        if k in self.micro_inputs: self.micro_inputs[k].setValue(v)
                except: pass

        # 2. Load Sleep
        row = cursor.execute("SELECT * FROM sleep_entries WHERE user_id=? AND entry_date=?", (user['id'], date_str)).fetchone()
        if row:
            self.sleep_duration.setValue(row['sleep_duration_hours'] or 0)
            self.sleep_quality.setValue(row['sleep_quality'] or 0)

        # 3. Load Workouts (Populate Table)
        session = cursor.execute("SELECT id FROM workout_sessions WHERE user_id=? AND session_date=?", (user['id'], date_str)).fetchone()
        if session:
            exercises = cursor.execute("""
                SELECT ep.*, e.name 
                FROM exercise_performances ep 
                JOIN exercises e ON ep.exercise_id = e.id 
                WHERE workout_session_id=?
            """, (session['id'],)).fetchall()
            
            for ex in exercises:
                self._add_row_to_table(ex['name'], ex['weight_kg'], ex['reps_completed'], ex['rir_actual'], False)

    def save_diet_entry(self):
        user = self.user_manager.get_current_user()
        if not user: return
        
        # Pack micros
        micros = {k: v.value() for k, v in self.micro_inputs.items() if v.value() > 0}
        note = f"MICROS_DATA::{json.dumps(micros)}"
        
        try:
            self.db.save_diet_entry(
                user['id'], self.selected_date,
                total_calories=self.diet_calories.value(),
                protein_g=self.diet_protein.value(),
                carbs_g=self.diet_carbs.value(),
                fat_g=self.diet_fats.value(),
                notes=note
            )
            QMessageBox.information(self, "Saved", "Nutrition Logged Successfully")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def save_workout_session(self):
        if not self.current_session_sets: return
        user = self.user_manager.get_current_user()
        
        try:
            sid = self.db.create_workout_session(user['id'], self.selected_date, notes="v3 Log")
            for i, s in enumerate(self.current_session_sets):
                self.db.add_exercise_performance(
                    sid, s['id'], i+1, s['w'], s['r'], s['rir']
                )
            QMessageBox.information(self, "Saved", "Workout Logged Successfully")
            self.current_session_sets = [] # Clear memory
            # Don't clear table immediately so user can see what they just saved
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def save_sleep_entry(self):
        user = self.user_manager.get_current_user()
        try:
            self.db.save_sleep_entry(
                user['id'], self.selected_date,
                sleep_duration_hours=self.sleep_duration.value(),
                sleep_quality=self.sleep_quality.value()
            )
            QMessageBox.information(self, "Saved", "Sleep Data Logged")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    # ==========================================
    # UTILS
    # ==========================================
    def refresh_exercise_list(self):
        self.exercise_combo.clear()
        exs = self.db.get_all_exercises()
        if exs:
            for e in exs: self.exercise_combo.addItem(e['name'], e['id'])
        else:
            self.exercise_combo.addItem("Bench Press", 1)

    def add_set_to_list(self):
        if not self.current_exercise_id:
            QMessageBox.warning(self, "Selection", "Please select an exercise from the library first.")
            return
        w = self.weight_input.value()
        r = self.reps_input.value()
        
        if w==0 and r==0: return
        
        uni = self.unilateral_check.isChecked()
        name = self.lbl_selected_exercise.text()
        ex_id = self.current_exercise_id
        
        self.current_session_sets.append({
            'id': ex_id, 'w': w, 'r': r, 'rir': self.rir_input.value(), 'uni': uni
        })
        self._add_row_to_table(name, w, r, self.rir_input.value(), uni)

    def _add_row_to_table(self, name, w, r, rir, uni):
        row = self.sets_table.rowCount()
        self.sets_table.insertRow(row)
        self.sets_table.setItem(row, 0, QTableWidgetItem(name))
        self.sets_table.setItem(row, 1, QTableWidgetItem(str(w)))
        self.sets_table.setItem(row, 2, QTableWidgetItem(str(r)))
        self.sets_table.setItem(row, 3, QTableWidgetItem(str(rir)))
        self.sets_table.setItem(row, 4, QTableWidgetItem("Single" if uni else "Bi-lat"))

    def clear_all_forms(self):
        self.diet_calories.setValue(0)
        self.diet_protein.setValue(0)
        self.diet_carbs.setValue(0)
        self.diet_fats.setValue(0)
        for w in self.micro_inputs.values(): w.setValue(0)
        self.sets_table.setRowCount(0)
        self.current_session_sets = []
        self.sleep_duration.setValue(0)
        self.sleep_quality.setValue(1)

    def on_date_selected(self):
        self.selected_date = self.calendar.selectedDate().toPyDate()
        self.date_info_label.setText(f"{self.selected_date.strftime('%A, %B %d')}")
        self.load_existing_data()

    def jump_to_today(self):
        self.calendar.setSelectedDate(QDate.currentDate())
        self.on_date_selected()

    # ==========================================
    # AI PREDICTION
    # ==========================================
    def on_predict_click(self):
        if not self.ai_predictor: return
        ex_name = self.exercise_combo.currentText()
        user = self.user_manager.get_current_user()
        
        self.predict_btn.setText("⏳ Computing...")
        try:
            result = self.ai_predictor.predict(user['id'], ex_name)
            self.show_prediction_results(ex_name, result)
        except Exception as e:
            QMessageBox.warning(self, "AI Error", str(e))
        finally:
            self.predict_btn.setText("🧠 AI Trajectory Analysis")

    def show_prediction_results(self, exercise, result):
        dialog = QDialog(self)
        dialog.setWindowTitle(f"AI Analytics: {exercise}")
        dialog.setMinimumSize(750, 600)
        layout = QVBoxLayout()
        
        # Header
        layout.addWidget(QLabel(f"Model: {result['type']}"))
        outcome = QLabel(f"Forecast: {result['prediction'].upper()}")
        outcome.setStyleSheet(f"font-size: 22px; font-weight: 800; color: {'#28a745' if 'LIKELY' in result['prediction'].upper() else '#dc3545'};")
        layout.addWidget(outcome)
        
        # Graph
        fig = Figure(figsize=(6, 4), dpi=100)
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        
        user = self.user_manager.get_current_user()
        history_df = self.db.get_user_workout_history_df(user['id'], exercise)
        
        if not history_df.empty and len(history_df) > 1:
            dates = range(len(history_df))
            weights = history_df['weight_kg'].values
            ax.plot(dates, weights, color='#007bff', linewidth=2, marker='o', label='History')
            
            last_x = dates[-1]
            pred_y = result.get('predicted_weight') or weights[-1]
            ax.plot([last_x, last_x+1], [weights[-1], pred_y], 'k--', alpha=0.5)
            ax.plot(last_x+1, pred_y, 'r*', markersize=15, label='Forecast')
        else:
            ax.text(0.5, 0.5, "Insufficient Data for Trendline", ha='center')
            
        ax.grid(True, alpha=0.3)
        ax.legend()
        layout.addWidget(canvas)
        
        # Info
        layout.addWidget(QLabel(f"Recommendation: {result['recommendations']['rir']} RIR"))
        layout.addWidget(QLabel(f"<i>{result['reason']}</i>"))
        
        btn = QPushButton("Close")
        btn.clicked.connect(dialog.accept)
        layout.addWidget(btn)
        dialog.setLayout(layout)
        dialog.exec()

    def open_exercise_library(self):
        """Opens the advanced exercise selector"""
        dialog = ExerciseLibraryDialog(self.db, self)
        dialog.exercise_selected.connect(self.on_exercise_selected)
        dialog.exec()

    def on_exercise_selected(self, ex_id, name, is_unilateral):
        """Callback when exercise is chosen from library"""
        self.lbl_selected_exercise.setText(name)
        self.current_exercise_id = ex_id
        
        # Auto-set the unilateral checkbox based on exercise capability
        self.unilateral_check.setChecked(is_unilateral)
        
        # If it's unilateral, maybe prompt user or auto-adjust logic?
        if is_unilateral:
            self.unilateral_check.setText("Unilateral (One side) - Recommended")
        else:
            self.unilateral_check.setText("Unilateral (One side)")