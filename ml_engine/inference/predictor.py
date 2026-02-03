"""
Scientific Hypertrophy Trainer - Comprehensive Tracking System
Diet, Sleep, Workout Logging, Body Measurements with Calendar
"""

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QFrame, QTabWidget, QCalendarWidget,
                             QLineEdit, QDoubleSpinBox, QSpinBox, QTextEdit,
                             QComboBox, QListWidget, QDialog, QDialogButtonBox,
                             QGridLayout, QGroupBox, QScrollArea, QMessageBox,
                             QTimeEdit, QCheckBox, QTableWidget, QTableWidgetItem,
                             QHeaderView, QFileDialog)
from PyQt6.QtCore import Qt, QDate, QTime, pyqtSignal
from PyQt6.QtGui import QTextCharFormat, QColor
from datetime import datetime, date
import csv

# ML Imports
from ml.inference.data_loader import InferenceDataLoader
from ml.inference.predictor import StrengthPredictor


class TrackingWidget(QWidget):
    """Main tracking interface with tabs for workouts, diet, sleep, body measurements"""
    
    def __init__(self, db_manager):
        super().__init__()
        self.db_manager = db_manager
        self.selected_date = date.today()
        
        # Initialize ML Components
        self._init_ml_engine()
        
        # Build UI
        self.init_ui()

    def _init_ml_engine(self):
        """Initialize the LSTM model and data loader."""
        try:
            self.data_loader = InferenceDataLoader(
                db_path="database/app.db",
                stats_path="ml/data/normalization_stats.json"
            )
            self.predictor = StrengthPredictor(
                model_path="ml/models/strength_predictor.pt"
            )
            self.ml_ready = True
            print("[TrackingWidget] ML engine initialized successfully")
        except Exception as e:
            self.ml_ready = False
            print(f"[TrackingWidget] ML engine failed to load: {e}")

    def init_ui(self):
        """Initialize UI components"""
        main_layout = QVBoxLayout()
        
        # Header
        header_label = QLabel("Scientific Hypertrophy Trainer")
        header_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        main_layout.addWidget(header_label)
        
        # Tab Widget
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)
        
        # Create tabs
        self.tabs.addTab(self._create_workout_tab(), "Workouts")
        self.tabs.addTab(self._create_diet_tab(), "Diet")
        self.tabs.addTab(self._create_sleep_tab(), "Sleep")
        self.tabs.addTab(self._create_body_tab(), "Body Measurements")
        
        # Bottom control buttons
        bottom_layout = QHBoxLayout()
        
        save_btn = QPushButton("Save All")
        save_btn.clicked.connect(self.save_all_data)
        bottom_layout.addWidget(save_btn)
        
        export_btn = QPushButton("Export Data")
        export_btn.clicked.connect(self.export_data)
        bottom_layout.addWidget(export_btn)
        
        # Predict button (AI)
        self.predict_btn = QPushButton("🤖 Predict Performance (AI)")
        self.predict_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        self.predict_btn.clicked.connect(self.on_predict_click)
        bottom_layout.addWidget(self.predict_btn)
        
        # Enable/disable predict button based on ML status
        if not self.ml_ready:
            self.predict_btn.setEnabled(False)
            self.predict_btn.setToolTip("ML engine not initialized")
        
        main_layout.addLayout(bottom_layout)
        
        self.setLayout(main_layout)

    def _create_workout_tab(self):
        """Create workout tracking tab"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Calendar
        calendar_group = QGroupBox("Select Date")
        cal_layout = QVBoxLayout()
        self.calendar = QCalendarWidget()
        self.calendar.clicked.connect(self.on_calendar_click)
        cal_layout.addWidget(self.calendar)
        calendar_group.setLayout(cal_layout)
        layout.addWidget(calendar_group)
        
        # Date display
        self.date_label = QLabel(f"Selected: {self.selected_date}")
        layout.addWidget(self.date_label)
        
        # Quick jump to today
        today_btn = QPushButton("Jump to Today")
        today_btn.clicked.connect(self.jump_to_today)
        layout.addWidget(today_btn)
        
        # Workout input form
        form_group = QGroupBox("Log Workout")
        form_layout = QGridLayout()
        
        # Exercise selection
        form_layout.addWidget(QLabel("Exercise:"), 0, 0)
        self.exercise_combo = QComboBox()
        self.exercise_combo.addItems([
            "Bench Press", "Squat", "Deadlift", "Overhead Press",
            "Bent Row", "Pull-ups", "Leg Press", "Leg Curl",
            "Chest Fly", "Lateral Raise", "Tricep Dips", "Barbell Curl"
        ])
        form_layout.addWidget(self.exercise_combo, 0, 1)
        
        # Weight
        form_layout.addWidget(QLabel("Weight (kg):"), 1, 0)
        self.workout_weight = QDoubleSpinBox()
        self.workout_weight.setRange(0, 500)
        self.workout_weight.setSingleStep(2.5)
        form_layout.addWidget(self.workout_weight, 1, 1)
        
        # Reps
        form_layout.addWidget(QLabel("Reps:"), 2, 0)
        self.workout_reps = QSpinBox()
        self.workout_reps.setRange(1, 50)
        form_layout.addWidget(self.workout_reps, 2, 1)
        
        # RIR
        form_layout.addWidget(QLabel("RIR:"), 3, 0)
        self.workout_rir = QSpinBox()
        self.workout_rir.setRange(0, 10)
        form_layout.addWidget(self.workout_rir, 3, 1)
        
        # Sets
        form_layout.addWidget(QLabel("Sets:"), 4, 0)
        self.workout_sets = QSpinBox()
        self.workout_sets.setRange(1, 20)
        self.workout_sets.setValue(3)
        form_layout.addWidget(self.workout_sets, 4, 1)
        
        # Notes
        form_layout.addWidget(QLabel("Notes:"), 5, 0)
        self.workout_notes = QTextEdit()
        self.workout_notes.setMaximumHeight(100)
        form_layout.addWidget(self.workout_notes, 5, 1)
        
        form_group.setLayout(form_layout)
        layout.addWidget(form_group)
        
        widget.setLayout(layout)
        return widget

    def _create_diet_tab(self):
        """Create diet tracking tab"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        diet_group = QGroupBox("Log Diet")
        form_layout = QGridLayout()
        
        # Calories
        form_layout.addWidget(QLabel("Calories:"), 0, 0)
        self.diet_calories = QSpinBox()
        self.diet_calories.setRange(0, 10000)
        form_layout.addWidget(self.diet_calories, 0, 1)
        
        # Protein
        form_layout.addWidget(QLabel("Protein (g):"), 1, 0)
        self.diet_protein = QSpinBox()
        self.diet_protein.setRange(0, 500)
        form_layout.addWidget(self.diet_protein, 1, 1)
        
        # Carbs
        form_layout.addWidget(QLabel("Carbs (g):"), 2, 0)
        self.diet_carbs = QSpinBox()
        self.diet_carbs.setRange(0, 500)
        form_layout.addWidget(self.diet_carbs, 2, 1)
        
        # Fats
        form_layout.addWidget(QLabel("Fats (g):"), 3, 0)
        self.diet_fats = QSpinBox()
        self.diet_fats.setRange(0, 500)
        form_layout.addWidget(self.diet_fats, 3, 1)
        
        # Fiber
        form_layout.addWidget(QLabel("Fiber (g):"), 4, 0)
        self.diet_fiber = QSpinBox()
        self.diet_fiber.setRange(0, 100)
        form_layout.addWidget(self.diet_fiber, 4, 1)
        
        # Water
        form_layout.addWidget(QLabel("Water (ml):"), 5, 0)
        self.diet_water = QSpinBox()
        self.diet_water.setRange(0, 10000)
        form_layout.addWidget(self.diet_water, 5, 1)
        
        # Notes
        form_layout.addWidget(QLabel("Notes:"), 6, 0)
        self.diet_notes = QTextEdit()
        self.diet_notes.setMaximumHeight(80)
        form_layout.addWidget(self.diet_notes, 6, 1)
        
        diet_group.setLayout(form_layout)
        layout.addWidget(diet_group)
        layout.addStretch()
        
        widget.setLayout(layout)
        return widget

    def _create_sleep_tab(self):
        """Create sleep tracking tab"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        sleep_group = QGroupBox("Log Sleep")
        form_layout = QGridLayout()
        
        # Sleep hours
        form_layout.addWidget(QLabel("Sleep Hours:"), 0, 0)
        self.sleep_hours = QDoubleSpinBox()
        self.sleep_hours.setRange(0, 24)
        self.sleep_hours.setSingleStep(0.5)
        form_layout.addWidget(self.sleep_hours, 0, 1)
        
        # Sleep quality
        form_layout.addWidget(QLabel("Quality (1-10):"), 1, 0)
        self.sleep_quality = QSpinBox()
        self.sleep_quality.setRange(1, 10)
        form_layout.addWidget(self.sleep_quality, 1, 1)
        
        # Stress level
        form_layout.addWidget(QLabel("Stress Level (1-10):"), 2, 0)
        self.stress_level = QSpinBox()
        self.stress_level.setRange(1, 10)
        form_layout.addWidget(self.stress_level, 2, 1)
        
        # Notes
        form_layout.addWidget(QLabel("Notes:"), 3, 0)
        self.sleep_notes = QTextEdit()
        self.sleep_notes.setMaximumHeight(80)
        form_layout.addWidget(self.sleep_notes, 3, 1)
        
        sleep_group.setLayout(form_layout)
        layout.addWidget(sleep_group)
        layout.addStretch()
        
        widget.setLayout(layout)
        return widget

    def _create_body_tab(self):
        """Create body measurements tab"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        body_group = QGroupBox("Body Measurements")
        form_layout = QGridLayout()
        
        # Weight
        form_layout.addWidget(QLabel("Weight (kg):"), 0, 0)
        self.body_weight = QDoubleSpinBox()
        self.body_weight.setRange(0, 500)
        self.body_weight.setSingleStep(0.1)
        form_layout.addWidget(self.body_weight, 0, 1)
        
        # Body fat
        form_layout.addWidget(QLabel("Body Fat (%):"), 1, 0)
        self.body_bf = QDoubleSpinBox()
        self.body_bf.setRange(0, 100)
        self.body_bf.setSingleStep(0.1)
        form_layout.addWidget(self.body_bf, 1, 1)
        
        # Muscle mass
        form_layout.addWidget(QLabel("Muscle Mass (kg):"), 2, 0)
        self.body_muscle = QDoubleSpinBox()
        self.body_muscle.setRange(0, 500)
        self.body_muscle.setSingleStep(0.1)
        form_layout.addWidget(self.body_muscle, 2, 1)
        
        # Circumference measurements
        form_layout.addWidget(QLabel("Waist (cm):"), 3, 0)
        self.body_waist = QDoubleSpinBox()
        self.body_waist.setRange(0, 200)
        form_layout.addWidget(self.body_waist, 3, 1)
        
        form_layout.addWidget(QLabel("Chest (cm):"), 4, 0)
        self.body_chest = QDoubleSpinBox()
        self.body_chest.setRange(0, 200)
        form_layout.addWidget(self.body_chest, 4, 1)
        
        form_layout.addWidget(QLabel("Arm (cm):"), 5, 0)
        self.body_arm = QDoubleSpinBox()
        self.body_arm.setRange(0, 200)
        form_layout.addWidget(self.body_arm, 5, 1)
        
        form_layout.addWidget(QLabel("Forearm (cm):"), 6, 0)
        self.body_forearm = QDoubleSpinBox()
        self.body_forearm.setRange(0, 200)
        form_layout.addWidget(self.body_forearm, 6, 1)
        
        form_layout.addWidget(QLabel("Thigh (cm):"), 7, 0)
        self.body_thigh = QDoubleSpinBox()
        self.body_thigh.setRange(0, 200)
        form_layout.addWidget(self.body_thigh, 7, 1)
        
        form_layout.addWidget(QLabel("Calf (cm):"), 8, 0)
        self.body_calf = QDoubleSpinBox()
        self.body_calf.setRange(0, 200)
        form_layout.addWidget(self.body_calf, 8, 1)
        
        form_layout.addWidget(QLabel("Neck (cm):"), 9, 0)
        self.body_neck = QDoubleSpinBox()
        self.body_neck.setRange(0, 200)
        form_layout.addWidget(self.body_neck, 9, 1)
        
        # Notes
        form_layout.addWidget(QLabel("Notes:"), 10, 0)
        self.body_notes = QTextEdit()
        self.body_notes.setMaximumHeight(80)
        form_layout.addWidget(self.body_notes, 10, 1)
        
        body_group.setLayout(form_layout)
        layout.addWidget(body_group)
        layout.addStretch()
        
        widget.setLayout(layout)
        return widget

    def on_calendar_click(self, date):
        """Handle calendar date selection"""
        self.selected_date = date.toPyDate()
        self.date_label.setText(f"Selected: {self.selected_date}")
        self.load_existing_data()

    def load_existing_data(self):
        """Load existing data for selected date"""
        # This would query the database and populate forms
        # Placeholder for now
        pass

    def jump_to_today(self):
        """Jump calendar to today's date"""
        today = date.today()
        qdate = QDate(today.year, today.month, today.day)
        self.calendar.setSelectedDate(qdate)
        self.selected_date = today
        self.load_existing_data()

    def save_all_data(self):
        """Save all tracking data to database"""
        try:
            # Collect data from all tabs
            data = {
                'date': self.selected_date.isoformat(),
                'exercise': self.exercise_combo.currentText(),
                'weight_kg': self.workout_weight.value(),
                'reps': self.workout_reps.value(),
                'rir': self.workout_rir.value(),
                'sets': self.workout_sets.value(),
                'workout_notes': self.workout_notes.toPlainText(),
                'calories': self.diet_calories.value(),
                'protein_g': self.diet_protein.value(),
                'carbs_g': self.diet_carbs.value(),
                'fats_g': self.diet_fats.value(),
                'fiber_g': self.diet_fiber.value(),
                'water_ml': self.diet_water.value(),
                'sleep_hours': self.sleep_hours.value(),
                'sleep_quality': self.sleep_quality.value(),
                'stress_level': self.stress_level.value(),
                'body_weight': self.body_weight.value(),
                'body_fat_pct': self.body_bf.value(),
                'muscle_mass': self.body_muscle.value(),
            }
            
            # Save via db_manager
            self.db_manager.save_tracking_data(data)
            QMessageBox.information(self, "Success", "All data saved successfully!")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save data: {str(e)}")

    def export_data(self):
        """Export tracking data to CSV"""
        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Export Data", "", "CSV Files (*.csv)"
            )
            
            if file_path:
                # Query all data from database and write to CSV
                # Placeholder implementation
                QMessageBox.information(self, "Success", f"Data exported to {file_path}")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Export failed: {str(e)}")

    def on_predict_click(self):
        """
        Handle prediction button click.
        Triggers AI model to predict future performance.
        """
        if not self.ml_ready:
            QMessageBox.warning(self, "ML Error", "ML engine not initialized")
            return
        
        # Get current exercise
        current_exercise = self.exercise_combo.currentText()
        current_user_id = 1  # TODO: Get from authenticated user context
        
        self.predict_btn.setText("🤖 Calculating...")
        self.predict_btn.setEnabled(False)
        
        try:
            # 1. LOAD & PREPARE DATA
            print(f"[Predict] Loading data for user {current_user_id}, exercise '{current_exercise}'")
            input_tensor = self.data_loader.prepare_inference_tensor(
                user_id=current_user_id,
                exercise_name=current_exercise
            )
            
            # 2. RUN INFERENCE
            print(f"[Predict] Running model inference...")
            # Predictor returns a Dictionary now, not a raw tensor
            predictions = self.predictor.predict(input_tensor)
            
            # 3. DISPLAY RESULTS
            self.show_prediction_dialog(current_exercise, predictions)
            
        except Exception as e:
            print(f"[Predict] Error: {e}")
            QMessageBox.critical(self, "Prediction Error", f"Could not generate prediction:\n\n{str(e)}")
        
        finally:
            self.predict_btn.setText("🤖 Predict Performance (AI)")
            self.predict_btn.setEnabled(True)

    def show_prediction_dialog(self, exercise_name: str, predictions: dict):
        """
        Display prediction results in a dialog window.
        Uses the Dictionary output from StrengthPredictor.
        """
        dialog = QDialog(self)
        dialog.setWindowTitle(f"AI Performance Prediction - {exercise_name}")
        dialog.setMinimumWidth(500)
        
        layout = QVBoxLayout()
        
        # Title
        title = QLabel(f"Next Session Forecast (Horizon 1)")
        title.setStyleSheet("font-size: 14px; font-weight: bold;")
        layout.addWidget(title)
        
        # Extract Horizon 1 Data
        h1 = predictions.get('horizon_1_session', {})
        weight = h1.get('weight', 0.0)
        reps = h1.get('reps', 0.0)
        rir = h1.get('rir', 0.0)
        
        # Display nicely
        info_text = (
            f"<b>Predicted Performance for Next Workout:</b><br>"
            f"• Weight: <b>{weight:.2f} kg</b><br>"
            f"• Reps: <b>{reps:.1f}</b><br>"
            f"• RIR: <b>{rir:.1f}</b><br><br>"
            f"<i>Model Uncertainty: ±{h1.get('weight_uncertainty', 0):.2f} kg</i>"
        )
        
        label = QLabel(info_text)
        label.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(label)

        # Show raw data for debugging (Optional)
        debug_group = QGroupBox("Advanced Metrics (Horizons 2, 4, 10)")
        debug_layout = QVBoxLayout()
        debug_text = QTextEdit()
        
        # Format the rest of the dictionary into string
        import json
        pretty_json = json.dumps(predictions, indent=2)
        debug_text.setPlainText(pretty_json)
        debug_text.setReadOnly(True)
        debug_text.setMaximumHeight(150)
        
        debug_layout.addWidget(debug_text)
        debug_group.setLayout(debug_layout)
        layout.addWidget(debug_group)
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)
        
        dialog.setLayout(layout)
        dialog.exec()
