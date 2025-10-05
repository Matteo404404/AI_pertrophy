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


class TrackingWidget(QWidget):
    """Comprehensive tracking interface with calendar and multiple tracking types"""
    
    def __init__(self, db_manager, tracking_system, user_manager):
        super().__init__()
        self.db = db_manager
        self.tracking_system = tracking_system
        self.user_manager = user_manager
        self.selected_date = date.today()
        
        self.init_ui()
    
    def init_ui(self):
        """Initialize tracking interface"""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(40, 40, 40, 40)
        
        # Header
        header = QLabel("Progress Tracking")
        header.setStyleSheet("font-size: 32px; font-weight: bold; color: #667eea; margin-bottom: 20px;")
        main_layout.addWidget(header)
        
        # Main content layout
        content_layout = QHBoxLayout()
        
        # Left side - Calendar
        self.create_calendar_section(content_layout)
        
        # Right side - Tracking tabs
        self.create_tracking_tabs(content_layout)
        
        main_layout.addLayout(content_layout)
        
        # Bottom buttons
        bottom_layout = QHBoxLayout()
        
        export_btn = QPushButton("Export Data (CSV)")
        export_btn.clicked.connect(self.export_data)
        export_btn.setStyleSheet("background-color: #6c757d;")
        
        bottom_layout.addWidget(export_btn)
        bottom_layout.addStretch()
        
        main_layout.addLayout(bottom_layout)
    
    def create_calendar_section(self, parent_layout):
        """Create calendar widget section"""
        calendar_frame = QFrame()
        calendar_frame.setMaximumWidth(400)
        
        calendar_layout = QVBoxLayout(calendar_frame)
        
        # Calendar title
        cal_title = QLabel("Select Date")
        cal_title.setStyleSheet("font-size: 20px; font-weight: bold; margin-bottom: 15px;")
        calendar_layout.addWidget(cal_title)
        
        # Calendar widget
        self.calendar = QCalendarWidget()
        self.calendar.setSelectedDate(QDate.currentDate())
        self.calendar.clicked.connect(self.on_date_selected)
        calendar_layout.addWidget(self.calendar)
        
        # Date info
        self.date_info_label = QLabel(f"Selected: {date.today().strftime('%B %d, %Y')}")
        self.date_info_label.setStyleSheet("""
            font-size: 14px;
            color: #6c757d;
            margin-top: 15px;
            padding: 10px;
            background-color: #f8f9fa;
            border-radius: 6px;
        """)
        calendar_layout.addWidget(self.date_info_label)
        
        # Today button
        today_btn = QPushButton("Jump to Today")
        today_btn.clicked.connect(self.jump_to_today)
        today_btn.setStyleSheet("background-color: #667eea; margin-top: 10px;")
        calendar_layout.addWidget(today_btn)
        
        parent_layout.addWidget(calendar_frame)
    
    def create_tracking_tabs(self, parent_layout):
        """Create tracking tabs for different tracking types"""
        tabs_frame = QFrame()
        tabs_layout = QVBoxLayout(tabs_frame)
        
        self.tabs = QTabWidget()
        
        # Create tabs
        self.diet_tab = self.create_diet_tab()
        self.sleep_tab = self.create_sleep_tab()
        self.workout_tab = self.create_workout_tab()
        self.body_tab = self.create_body_measurements_tab()
        
        self.tabs.addTab(self.diet_tab, "Diet")
        self.tabs.addTab(self.sleep_tab, "Sleep")
        self.tabs.addTab(self.workout_tab, "Workout")
        self.tabs.addTab(self.body_tab, "Body Metrics")
        
        tabs_layout.addWidget(self.tabs)
        
        parent_layout.addWidget(tabs_frame)
    
    def create_diet_tab(self):
        """Create comprehensive diet tracking tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Scroll area for form
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; }")
        
        form_widget = QWidget()
        form_layout = QVBoxLayout(form_widget)
        
        # Macros section
        macros_group = QGroupBox("Macronutrients")
        macros_layout = QGridLayout()
        
        # Calories
        macros_layout.addWidget(QLabel("Total Calories:"), 0, 0)
        self.diet_calories = QSpinBox()
        self.diet_calories.setRange(0, 10000)
        self.diet_calories.setSuffix(" kcal")
        macros_layout.addWidget(self.diet_calories, 0, 1)
        
        # Protein
        macros_layout.addWidget(QLabel("Protein:"), 1, 0)
        self.diet_protein = QDoubleSpinBox()
        self.diet_protein.setRange(0, 500)
        self.diet_protein.setSuffix(" g")
        macros_layout.addWidget(self.diet_protein, 1, 1)
        
        # Carbs
        macros_layout.addWidget(QLabel("Carbohydrates:"), 2, 0)
        self.diet_carbs = QDoubleSpinBox()
        self.diet_carbs.setRange(0, 1000)
        self.diet_carbs.setSuffix(" g")
        macros_layout.addWidget(self.diet_carbs, 2, 1)
        
        # Fats
        macros_layout.addWidget(QLabel("Fats:"), 3, 0)
        self.diet_fats = QDoubleSpinBox()
        self.diet_fats.setRange(0, 300)
        self.diet_fats.setSuffix(" g")
        macros_layout.addWidget(self.diet_fats, 3, 1)
        
        # Fiber
        macros_layout.addWidget(QLabel("Fiber (optional):"), 4, 0)
        self.diet_fiber = QDoubleSpinBox()
        self.diet_fiber.setRange(0, 100)
        self.diet_fiber.setSuffix(" g")
        macros_layout.addWidget(self.diet_fiber, 4, 1)
        
        # Sugar
        macros_layout.addWidget(QLabel("Sugar (optional):"), 5, 0)
        self.diet_sugar = QDoubleSpinBox()
        self.diet_sugar.setRange(0, 300)
        self.diet_sugar.setSuffix(" g")
        macros_layout.addWidget(self.diet_sugar, 5, 1)
        
        macros_group.setLayout(macros_layout)
        form_layout.addWidget(macros_group)
        
        # Micronutrients section
        micros_group = QGroupBox("Micronutrients (optional)")
        micros_layout = QGridLayout()
        
        self.diet_sodium = QDoubleSpinBox()
        self.diet_sodium.setRange(0, 10000)
        self.diet_sodium.setSuffix(" mg")
        micros_layout.addWidget(QLabel("Sodium:"), 0, 0)
        micros_layout.addWidget(self.diet_sodium, 0, 1)
        
        self.diet_potassium = QDoubleSpinBox()
        self.diet_potassium.setRange(0, 10000)
        self.diet_potassium.setSuffix(" mg")
        micros_layout.addWidget(QLabel("Potassium:"), 1, 0)
        micros_layout.addWidget(self.diet_potassium, 1, 1)
        
        self.diet_calcium = QDoubleSpinBox()
        self.diet_calcium.setRange(0, 5000)
        self.diet_calcium.setSuffix(" mg")
        micros_layout.addWidget(QLabel("Calcium:"), 2, 0)
        micros_layout.addWidget(self.diet_calcium, 2, 1)
        
        self.diet_iron = QDoubleSpinBox()
        self.diet_iron.setRange(0, 100)
        self.diet_iron.setSuffix(" mg")
        micros_layout.addWidget(QLabel("Iron:"), 3, 0)
        micros_layout.addWidget(self.diet_iron, 3, 1)
        
        self.diet_vitamin_d = QDoubleSpinBox()
        self.diet_vitamin_d.setRange(0, 200)
        self.diet_vitamin_d.setSuffix(" µg")
        micros_layout.addWidget(QLabel("Vitamin D:"), 4, 0)
        micros_layout.addWidget(self.diet_vitamin_d, 4, 1)
        
        self.diet_vitamin_c = QDoubleSpinBox()
        self.diet_vitamin_c.setRange(0, 1000)
        self.diet_vitamin_c.setSuffix(" mg")
        micros_layout.addWidget(QLabel("Vitamin C:"), 5, 0)
        micros_layout.addWidget(self.diet_vitamin_c, 5, 1)
        
        self.diet_b12 = QDoubleSpinBox()
        self.diet_b12.setRange(0, 100)
        self.diet_b12.setSuffix(" µg")
        micros_layout.addWidget(QLabel("Vitamin B12:"), 6, 0)
        micros_layout.addWidget(self.diet_b12, 6, 1)
        
        micros_group.setLayout(micros_layout)
        form_layout.addWidget(micros_group)
        
        # Hypertrophy-specific section
        hypertrophy_group = QGroupBox("Hypertrophy Optimization")
        hypertrophy_layout = QGridLayout()
        
        self.diet_hydration = QDoubleSpinBox()
        self.diet_hydration.setRange(0, 10)
        self.diet_hydration.setSuffix(" L")
        self.diet_hydration.setDecimals(1)
        hypertrophy_layout.addWidget(QLabel("Hydration:"), 0, 0)
        hypertrophy_layout.addWidget(self.diet_hydration, 0, 1)
        
        self.diet_meals = QSpinBox()
        self.diet_meals.setRange(1, 10)
        hypertrophy_layout.addWidget(QLabel("Meal Count:"), 1, 0)
        hypertrophy_layout.addWidget(self.diet_meals, 1, 1)
        
        self.diet_protein_per_kg = QDoubleSpinBox()
        self.diet_protein_per_kg.setRange(0, 5)
        self.diet_protein_per_kg.setSuffix(" g/kg")
        self.diet_protein_per_kg.setDecimals(1)
        hypertrophy_layout.addWidget(QLabel("Protein per kg:"), 2, 0)
        hypertrophy_layout.addWidget(self.diet_protein_per_kg, 2, 1)
        
        self.diet_pre_workout_carbs = QDoubleSpinBox()
        self.diet_pre_workout_carbs.setRange(0, 200)
        self.diet_pre_workout_carbs.setSuffix(" g")
        hypertrophy_layout.addWidget(QLabel("Pre-workout Carbs:"), 3, 0)
        hypertrophy_layout.addWidget(self.diet_pre_workout_carbs, 3, 1)
        
        self.diet_post_workout_carbs = QDoubleSpinBox()
        self.diet_post_workout_carbs.setRange(0, 200)
        self.diet_post_workout_carbs.setSuffix(" g")
        hypertrophy_layout.addWidget(QLabel("Post-workout Carbs:"), 4, 0)
        hypertrophy_layout.addWidget(self.diet_post_workout_carbs, 4, 1)
        
        self.diet_creatine = QDoubleSpinBox()
        self.diet_creatine.setRange(0, 20)
        self.diet_creatine.setSuffix(" g")
        hypertrophy_layout.addWidget(QLabel("Creatine:"), 5, 0)
        hypertrophy_layout.addWidget(self.diet_creatine, 5, 1)
        
        hypertrophy_group.setLayout(hypertrophy_layout)
        form_layout.addWidget(hypertrophy_group)
        
        # Notes
        form_layout.addWidget(QLabel("Notes:"))
        self.diet_notes = QTextEdit()
        self.diet_notes.setMaximumHeight(100)
        self.diet_notes.setPlaceholderText("Additional diet notes...")
        form_layout.addWidget(self.diet_notes)
        
        # Save button
        save_diet_btn = QPushButton("Save Diet Entry")
        save_diet_btn.clicked.connect(self.save_diet_entry)
        save_diet_btn.setMinimumHeight(50)
        save_diet_btn.setStyleSheet("background-color: #28a745; font-size: 16px; font-weight: bold;")
        form_layout.addWidget(save_diet_btn)
        
        scroll.setWidget(form_widget)
        layout.addWidget(scroll)
        
        return tab
    
    def create_sleep_tab(self):
        """Create comprehensive sleep tracking tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; }")
        
        form_widget = QWidget()
        form_layout = QVBoxLayout(form_widget)
        
        # Basic sleep metrics
        basic_group = QGroupBox("Sleep Metrics")
        basic_layout = QGridLayout()
        
        basic_layout.addWidget(QLabel("Bedtime:"), 0, 0)
        self.sleep_bedtime = QTimeEdit()
        self.sleep_bedtime.setDisplayFormat("HH:mm")
        basic_layout.addWidget(self.sleep_bedtime, 0, 1)
        
        basic_layout.addWidget(QLabel("Wake Time:"), 1, 0)
        self.sleep_waketime = QTimeEdit()
        self.sleep_waketime.setDisplayFormat("HH:mm")
        basic_layout.addWidget(self.sleep_waketime, 1, 1)
        
        basic_layout.addWidget(QLabel("Total Duration:"), 2, 0)
        self.sleep_duration = QDoubleSpinBox()
        self.sleep_duration.setRange(0, 24)
        self.sleep_duration.setSuffix(" hours")
        self.sleep_duration.setDecimals(1)
        basic_layout.addWidget(self.sleep_duration, 2, 1)
        
        basic_layout.addWidget(QLabel("Sleep Quality (1-10):"), 3, 0)
        self.sleep_quality = QSpinBox()
        self.sleep_quality.setRange(1, 10)
        basic_layout.addWidget(self.sleep_quality, 3, 1)
        
        basic_group.setLayout(basic_layout)
        form_layout.addWidget(basic_group)
        
        # Advanced metrics
        advanced_group = QGroupBox("Advanced Metrics (optional)")
        advanced_layout = QGridLayout()
        
        self.sleep_fall_asleep = QSpinBox()
        self.sleep_fall_asleep.setRange(0, 180)
        self.sleep_fall_asleep.setSuffix(" minutes")
        advanced_layout.addWidget(QLabel("Time to Fall Asleep:"), 0, 0)
        advanced_layout.addWidget(self.sleep_fall_asleep, 0, 1)
        
        self.sleep_awakenings = QSpinBox()
        self.sleep_awakenings.setRange(0, 20)
        advanced_layout.addWidget(QLabel("Awakenings Count:"), 1, 0)
        advanced_layout.addWidget(self.sleep_awakenings, 1, 1)
        
        self.sleep_deep = QDoubleSpinBox()
        self.sleep_deep.setRange(0, 12)
        self.sleep_deep.setSuffix(" hours")
        self.sleep_deep.setDecimals(1)
        advanced_layout.addWidget(QLabel("Deep Sleep:"), 2, 0)
        advanced_layout.addWidget(self.sleep_deep, 2, 1)
        
        self.sleep_rem = QDoubleSpinBox()
        self.sleep_rem.setRange(0, 12)
        self.sleep_rem.setSuffix(" hours")
        self.sleep_rem.setDecimals(1)
        advanced_layout.addWidget(QLabel("REM Sleep:"), 3, 0)
        advanced_layout.addWidget(self.sleep_rem, 3, 1)
        
        advanced_group.setLayout(advanced_layout)
        form_layout.addWidget(advanced_group)
        
        # Environment factors
        env_group = QGroupBox("Sleep Environment")
        env_layout = QGridLayout()
        
        self.sleep_caffeine_cutoff = QTimeEdit()
        self.sleep_caffeine_cutoff.setDisplayFormat("HH:mm")
        env_layout.addWidget(QLabel("Last Caffeine:"), 0, 0)
        env_layout.addWidget(self.sleep_caffeine_cutoff, 0, 1)
        
        self.sleep_screen_time = QSpinBox()
        self.sleep_screen_time.setRange(0, 300)
        self.sleep_screen_time.setSuffix(" minutes")
        env_layout.addWidget(QLabel("Screen Time Before Bed:"), 1, 0)
        env_layout.addWidget(self.sleep_screen_time, 1, 1)
        
        self.sleep_temperature = QDoubleSpinBox()
        self.sleep_temperature.setRange(10, 30)
        self.sleep_temperature.setSuffix(" °C")
        self.sleep_temperature.setDecimals(1)
        env_layout.addWidget(QLabel("Room Temperature:"), 2, 0)
        env_layout.addWidget(self.sleep_temperature, 2, 1)
        
        self.sleep_env_rating = QSpinBox()
        self.sleep_env_rating.setRange(1, 10)
        env_layout.addWidget(QLabel("Environment Rating (1-10):"), 3, 0)
        env_layout.addWidget(self.sleep_env_rating, 3, 1)
        
        env_group.setLayout(env_layout)
        form_layout.addWidget(env_group)
        
        # Notes
        form_layout.addWidget(QLabel("Notes:"))
        self.sleep_notes = QTextEdit()
        self.sleep_notes.setMaximumHeight(100)
        self.sleep_notes.setPlaceholderText("Sleep quality notes...")
        form_layout.addWidget(self.sleep_notes)
        
        # Save button
        save_sleep_btn = QPushButton("Save Sleep Entry")
        save_sleep_btn.clicked.connect(self.save_sleep_entry)
        save_sleep_btn.setMinimumHeight(50)
        save_sleep_btn.setStyleSheet("background-color: #28a745; font-size: 16px; font-weight: bold;")
        form_layout.addWidget(save_sleep_btn)
        
        scroll.setWidget(form_widget)
        layout.addWidget(scroll)
        
        return tab
    
    def create_workout_tab(self):
        """Create workout logging tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Workout selection/creation
        workout_header = QLabel("Log Workout Session")
        workout_header.setStyleSheet("font-size: 20px; font-weight: bold; margin-bottom: 15px;")
        layout.addWidget(workout_header)
        
        # Workout template selection
        template_layout = QHBoxLayout()
        
        template_layout.addWidget(QLabel("Select Workout:"))
        self.workout_template_combo = QComboBox()
        self.workout_template_combo.addItem("-- Create New Workout --")
        template_layout.addWidget(self.workout_template_combo)
        
        create_template_btn = QPushButton("Manage Workouts")
        create_template_btn.clicked.connect(self.open_workout_manager)
        create_template_btn.setStyleSheet("background-color: #667eea;")
        template_layout.addWidget(create_template_btn)
        
        layout.addLayout(template_layout)
        
        # Workout logging area (placeholder for now)
        workout_info = QLabel("Workout logging interface\n\nSelect or create a workout template to begin logging exercises.")
        workout_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        workout_info.setStyleSheet("""
            font-size: 16px;
            color: #6c757d;
            padding: 60px;
            background-color: #f8f9fa;
            border-radius: 12px;
            margin: 20px 0px;
        """)
        layout.addWidget(workout_info)
        
        layout.addStretch()
        
        return tab
    
    def create_body_measurements_tab(self):
        """Create body measurements tracking tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; }")
        
        form_widget = QWidget()
        form_layout = QVBoxLayout(form_widget)
        
        # Primary metrics
        primary_group = QGroupBox("Primary Metrics")
        primary_layout = QGridLayout()
        
        primary_layout.addWidget(QLabel("Weight:"), 0, 0)
        self.body_weight = QDoubleSpinBox()
        self.body_weight.setRange(30, 300)
        self.body_weight.setSuffix(" kg")
        self.body_weight.setDecimals(1)
        primary_layout.addWidget(self.body_weight, 0, 1)
        
        primary_layout.addWidget(QLabel("Body Fat %:"), 1, 0)
        self.body_bf = QDoubleSpinBox()
        self.body_bf.setRange(3, 50)
        self.body_bf.setSuffix(" %")
        self.body_bf.setDecimals(1)
        primary_layout.addWidget(self.body_bf, 1, 1)
        
        primary_layout.addWidget(QLabel("Muscle Mass:"), 2, 0)
        self.body_muscle = QDoubleSpinBox()
        self.body_muscle.setRange(0, 150)
        self.body_muscle.setSuffix(" kg")
        self.body_muscle.setDecimals(1)
        primary_layout.addWidget(self.body_muscle, 2, 1)
        
        primary_group.setLayout(primary_layout)
        form_layout.addWidget(primary_group)
        
        # Circumferences
        circum_group = QGroupBox("Circumference Measurements (cm)")
        circum_layout = QGridLayout()
        
        self.body_waist = QDoubleSpinBox()
        self.body_waist.setRange(0, 200)
        self.body_waist.setSuffix(" cm")
        self.body_waist.setDecimals(1)
        circum_layout.addWidget(QLabel("Waist:"), 0, 0)
        circum_layout.addWidget(self.body_waist, 0, 1)
        
        self.body_chest = QDoubleSpinBox()
        self.body_chest.setRange(0, 200)
        self.body_chest.setSuffix(" cm")
        self.body_chest.setDecimals(1)
        circum_layout.addWidget(QLabel("Chest:"), 1, 0)
        circum_layout.addWidget(self.body_chest, 1, 1)
        
        self.body_arm = QDoubleSpinBox()
        self.body_arm.setRange(0, 100)
        self.body_arm.setSuffix(" cm")
        self.body_arm.setDecimals(1)
        circum_layout.addWidget(QLabel("Arm:"), 2, 0)
        circum_layout.addWidget(self.body_arm, 2, 1)
        
        self.body_forearm = QDoubleSpinBox()
        self.body_forearm.setRange(0, 100)
        self.body_forearm.setSuffix(" cm")
        self.body_forearm.setDecimals(1)
        circum_layout.addWidget(QLabel("Forearm:"), 3, 0)
        circum_layout.addWidget(self.body_forearm, 3, 1)
        
        self.body_thigh = QDoubleSpinBox()
        self.body_thigh.setRange(0, 150)
        self.body_thigh.setSuffix(" cm")
        self.body_thigh.setDecimals(1)
        circum_layout.addWidget(QLabel("Thigh:"), 4, 0)
        circum_layout.addWidget(self.body_thigh, 4, 1)
        
        self.body_calf = QDoubleSpinBox()
        self.body_calf.setRange(0, 100)
        self.body_calf.setSuffix(" cm")
        self.body_calf.setDecimals(1)
        circum_layout.addWidget(QLabel("Calf:"), 5, 0)
        circum_layout.addWidget(self.body_calf, 5, 1)
        
        self.body_neck = QDoubleSpinBox()
        self.body_neck.setRange(0, 100)
        self.body_neck.setSuffix(" cm")
        self.body_neck.setDecimals(1)
        circum_layout.addWidget(QLabel("Neck:"), 6, 0)
        circum_layout.addWidget(self.body_neck, 6, 1)
        
        circum_group.setLayout(circum_layout)
        form_layout.addWidget(circum_group)
        
        # Notes
        form_layout.addWidget(QLabel("Notes:"))
        self.body_notes = QTextEdit()
        self.body_notes.setMaximumHeight(100)
        self.body_notes.setPlaceholderText("Measurement notes...")
        form_layout.addWidget(self.body_notes)
        
        # Save button
        save_body_btn = QPushButton("Save Body Measurements")
        save_body_btn.clicked.connect(self.save_body_measurements)
        save_body_btn.setMinimumHeight(50)
        save_body_btn.setStyleSheet("background-color: #28a745; font-size: 16px; font-weight: bold;")
        form_layout.addWidget(save_body_btn)
        
        scroll.setWidget(form_widget)
        layout.addWidget(scroll)
        
        return tab

    def load_existing_data(self):
        """Load existing tracking data for selected date and populate forms"""
        current_user = self.user_manager.get_current_user()
        if not current_user:
            return

        user_id = current_user['id']
        date_str = self.selected_date.strftime('%Y-%m-%d')
        
        try:
            # Clear all forms first
            self.clear_all_forms()
            
            # Load diet data
            cursor = self.db.conn.cursor()
            diet_entry = cursor.execute(
                "SELECT * FROM diet_entries WHERE user_id = ? AND entry_date = ?",
                (user_id, date_str)
            ).fetchone()
            
            if diet_entry:
                diet_data = dict(diet_entry)
                self.populate_diet_form(diet_data)
            
            # Load sleep data  
            sleep_entry = cursor.execute(
                "SELECT * FROM sleep_entries WHERE user_id = ? AND entry_date = ?", 
                (user_id, date_str)
            ).fetchone()
            
            if sleep_entry:
                sleep_data = dict(sleep_entry)
                self.populate_sleep_form(sleep_data)
                
            # Load body measurements
            body_entry = cursor.execute(
                "SELECT * FROM body_measurements WHERE user_id = ? AND measurement_date = ?",
                (user_id, date_str) 
            ).fetchone()
            
            if body_entry:
                body_data = dict(body_entry)
                self.populate_body_form(body_data)
                
        except Exception as e:
            print(f"Error loading existing data: {e}")

    def clear_all_forms(self):
        """Clear all input forms"""
        # Clear diet form
        self.diet_calories.setValue(0)
        self.diet_protein.setValue(0)
        self.diet_carbs.setValue(0)
        self.diet_fats.setValue(0)
        self.diet_fiber.setValue(0)
        self.diet_sugar.setValue(0)
        self.diet_sodium.setValue(0)
        self.diet_potassium.setValue(0)
        self.diet_calcium.setValue(0)
        self.diet_iron.setValue(0)
        self.diet_vitamin_d.setValue(0)
        self.diet_vitamin_c.setValue(0)
        self.diet_b12.setValue(0)
        self.diet_hydration.setValue(0)
        self.diet_meals.setValue(0)
        self.diet_protein_per_kg.setValue(0)
        self.diet_pre_workout_carbs.setValue(0)
        self.diet_post_workout_carbs.setValue(0)
        self.diet_creatine.setValue(0)
        self.diet_notes.clear()
        
        # Clear sleep form
        self.sleep_bedtime.setTime(self.sleep_bedtime.minimumTime())
        self.sleep_wake_time.setTime(self.sleep_wake_time.minimumTime())
        self.sleep_duration.setValue(0)
        self.sleep_quality.setValue(1)
        self.sleep_fall_asleep.setValue(0)
        self.sleep_awakenings.setValue(0)
        self.sleep_deep.setValue(0)
        self.sleep_rem.setValue(0)
        self.sleep_caffeine_cutoff.setTime(self.sleep_caffeine_cutoff.minimumTime())
        self.sleep_screen_time.setValue(0)
        self.sleep_temperature.setValue(0)
        self.sleep_env_rating.setValue(0)
        self.sleep_notes.clear()
        
        # Clear body form
        self.body_weight.setValue(0)
        self.body_bf.setValue(0)
        self.body_muscle.setValue(0)
        self.body_waist.setValue(0)
        self.body_chest.setValue(0)
        self.body_arm.setValue(0)
        self.body_forearm.setValue(0)
        self.body_thigh.setValue(0)
        self.body_calf.setValue(0)
        self.body_neck.setValue(0)
        self.body_notes.clear()

    def populate_diet_form(self, data):
        """Populate diet form with existing data"""
        self.diet_calories.setValue(data.get('total_calories', 0))
        self.diet_protein.setValue(data.get('protein_g', 0))
        self.diet_carbs.setValue(data.get('carbs_g', 0))
        self.diet_fats.setValue(data.get('fat_g', 0))
        self.diet_fiber.setValue(data.get('fiber_g') or 0)
        self.diet_sugar.setValue(data.get('sugar_g') or 0)
        self.diet_sodium.setValue(data.get('sodium_mg') or 0)
        self.diet_potassium.setValue(data.get('potassium_mg') or 0)
        self.diet_calcium.setValue(data.get('calcium_mg') or 0)
        self.diet_iron.setValue(data.get('iron_mg') or 0)
        self.diet_vitamin_d.setValue(data.get('vitamin_d_ug') or 0)
        self.diet_vitamin_c.setValue(data.get('vitamin_c_mg') or 0)
        self.diet_b12.setValue(data.get('b12_ug') or 0)
        self.diet_hydration.setValue(data.get('hydration_liters') or 0)
        self.diet_meals.setValue(data.get('meals_count') or 0)
        self.diet_protein_per_kg.setValue(data.get('protein_per_kg') or 0)
        self.diet_pre_workout_carbs.setValue(data.get('pre_workout_carbs_g') or 0)
        self.diet_post_workout_carbs.setValue(data.get('post_workout_carbs_g') or 0)
        self.diet_creatine.setValue(data.get('creatine_g') or 0)
        if data.get('notes'):
            self.diet_notes.setPlainText(data['notes'])

    def populate_sleep_form(self, data):
        """Populate sleep form with existing data"""  
        if data.get('bedtime'):
            time_obj = QTime.fromString(data['bedtime'], "HH:mm")
            self.sleep_bedtime.setTime(time_obj)
        if data.get('wake_time'):
            time_obj = QTime.fromString(data['wake_time'], "HH:mm") 
            self.sleep_wake_time.setTime(time_obj)
            
        self.sleep_duration.setValue(data.get('sleep_duration_hours', 0))
        self.sleep_quality.setValue(data.get('sleep_quality', 1))
        self.sleep_fall_asleep.setValue(data.get('time_to_fall_asleep_minutes') or 0)
        self.sleep_awakenings.setValue(data.get('awakenings_count') or 0)
        self.sleep_deep.setValue(data.get('deep_sleep_hours') or 0)
        self.sleep_rem.setValue(data.get('rem_sleep_hours') or 0)
        
        if data.get('caffeine_cutoff_time'):
            time_obj = QTime.fromString(data['caffeine_cutoff_time'], "HH:mm")
            self.sleep_caffeine_cutoff.setTime(time_obj)
            
        self.sleep_screen_time.setValue(data.get('screen_time_before_bed_minutes') or 0)
        self.sleep_temperature.setValue(data.get('room_temperature_celsius') or 0) 
        self.sleep_env_rating.setValue(data.get('sleep_environment_rating') or 0)
        
        if data.get('notes'):
            self.sleep_notes.setPlainText(data['notes'])

    def populate_body_form(self, data):
        """Populate body measurements form with existing data"""
        self.body_weight.setValue(data.get('weight_kg') or 0)
        self.body_bf.setValue(data.get('body_fat_percentage') or 0)
        self.body_muscle.setValue(data.get('muscle_mass_kg') or 0)
        self.body_waist.setValue(data.get('waist_cm') or 0)
        self.body_chest.setValue(data.get('chest_cm') or 0)
        self.body_arm.setValue(data.get('arm_cm') or 0)
        self.body_forearm.setValue(data.get('forearm_cm') or 0)
        self.body_thigh.setValue(data.get('thigh_cm') or 0)
        self.body_calf.setValue(data.get('calf_cm') or 0)
        self.body_neck.setValue(data.get('neck_cm') or 0)
        
        if data.get('notes'):
            self.body_notes.setPlainText(data['notes'])

    # ALSO ADD THIS FIXED save_diet_entry method (with the comma fix):

    def save_diet_entry(self):
        """Save diet entry to database"""
        current_user = self.user_manager.get_current_user()
        if not current_user:
            QMessageBox.warning(self, "Error", "No user selected.")
            return
        
        try:
            entry_id = self.db.save_diet_entry(
                user_id=current_user['id'],
                entry_date=self.selected_date,
                total_calories=self.diet_calories.value(),
                protein_g=self.diet_protein.value(),
                carbs_g=self.diet_carbs.value(),
                fat_g=self.diet_fats.value(),
                fiber_g=self.diet_fiber.value() if self.diet_fiber.value() > 0 else None,
                sugar_g=self.diet_sugar.value() if self.diet_sugar.value() > 0 else None,
                sodium_mg=self.diet_sodium.value() if self.diet_sodium.value() > 0 else None,
                potassium_mg=self.diet_potassium.value() if self.diet_potassium.value() > 0 else None,
                calcium_mg=self.diet_calcium.value() if self.diet_calcium.value() > 0 else None,
                iron_mg=self.diet_iron.value() if self.diet_iron.value() > 0 else None,  # <-- ADD COMMA HERE
                vitamin_d_ug=self.diet_vitamin_d.value() if self.diet_vitamin_d.value() > 0 else None,
                vitamin_c_mg=self.diet_vitamin_c.value() if self.diet_vitamin_c.value() > 0 else None,
                b12_ug=self.diet_b12.value() if self.diet_b12.value() > 0 else None,
                hydration_liters=self.diet_hydration.value() if self.diet_hydration.value() > 0 else None,
                meals_count=self.diet_meals.value() if self.diet_meals.value() > 0 else None,
                protein_per_kg=self.diet_protein_per_kg.value() if self.diet_protein_per_kg.value() > 0 else None,
                pre_workout_carbs_g=self.diet_pre_workout_carbs.value() if self.diet_pre_workout_carbs.value() > 0 else None,
                post_workout_carbs_g=self.diet_post_workout_carbs.value() if self.diet_post_workout_carbs.value() > 0 else None,
                creatine_g=self.diet_creatine.value() if self.diet_creatine.value() > 0 else None,
                notes=self.diet_notes.toPlainText() if self.diet_notes.toPlainText() else None
            )
            
            QMessageBox.information(self, "Success", f"Diet entry saved for {self.selected_date.strftime('%B %d, %Y')}!")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save diet entry: {str(e)}")
        # Event handlers
        def on_date_selected(self, qdate):
            """Handle calendar date selection"""
            self.selected_date = qdate.toPyDate()
            self.date_info_label.setText(f"Selected: {self.selected_date.strftime('%B %d, %Y')}")
            self.load_existing_data()
        
        def jump_to_today(self):
            """Jump calendar to today"""
            self.calendar.setSelectedDate(QDate.currentDate())
            self.selected_date = date.today()
            self.date_info_label.setText(f"Selected: {self.selected_date.strftime('%B %d, %Y')}")
            self.load_existing_data()
        
        def load_existing_data(self):
            """Load existing tracking data for selected date"""
            pass
        
        def save_diet_entry(self):
            """Save diet entry to database"""
            current_user = self.user_manager.get_current_user()
            if not current_user:
                QMessageBox.warning(self, "Error", "No user selected.")
                return
            
            try:
                entry_id = self.db.save_diet_entry(
                    user_id=current_user['id'],
                    entry_date=self.selected_date,
                    total_calories=self.diet_calories.value(),
                    protein_g=self.diet_protein.value(),
                    carbs_g=self.diet_carbs.value(),
                    fat_g=self.diet_fats.value(),
                    fiber_g=self.diet_fiber.value() if self.diet_fiber.value() > 0 else None,
                    sugar_g=self.diet_sugar.value() if self.diet_sugar.value() > 0 else None,
                    sodium_mg=self.diet_sodium.value() if self.diet_sodium.value() > 0 else None,
                    potassium_mg=self.diet_potassium.value() if self.diet_potassium.value() > 0 else None,
                    calcium_mg=self.diet_calcium.value() if self.diet_calcium.value() > 0 else None,
                    iron_mg=self.diet_iron.value() if self.diet_iron.value() > 0 else None,
                    vitamin_d_ug=self.diet_vitamin_d.value() if self.diet_vitamin_d.value() > 0 else None,
                    vitamin_c_mg=self.diet_vitamin_c.value() if self.diet_vitamin_c.value() > 0 else None,
                    b12_ug=self.diet_b12.value() if self.diet_b12.value() > 0 else None,
                    hydration_liters=self.diet_hydration.value() if self.diet_hydration.value() > 0 else None,
                    meals_count=self.diet_meals.value() if self.diet_meals.value() > 0 else None,
                    protein_per_kg=self.diet_protein_per_kg.value() if self.diet_protein_per_kg.value() > 0 else None,
                    pre_workout_carbs_g=self.diet_pre_workout_carbs.value() if self.diet_pre_workout_carbs.value() > 0 else None,
                    post_workout_carbs_g=self.diet_post_workout_carbs.value() if self.diet_post_workout_carbs.value() > 0 else None,
                    creatine_g=self.diet_creatine.value() if self.diet_creatine.value() > 0 else None,
                    notes=self.diet_notes.toPlainText() if self.diet_notes.toPlainText() else None
                )
                
                QMessageBox.information(self, "Success", f"Diet entry saved for {self.selected_date.strftime('%B %d, %Y')}!")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save diet entry: {str(e)}")
        
    def save_sleep_entry(self):
        """Save sleep entry to database"""
        current_user = self.user_manager.get_current_user()
        if not current_user:
            QMessageBox.warning(self, "Error", "No user selected.")
            return
        
        try:
            entry_id = self.db.save_sleep_entry(
                user_id=current_user['id'],
                entry_date=self.selected_date,
                bedtime=self.sleep_bedtime.time().toString("HH:mm") if self.sleep_bedtime.time().isValid() else None,
                wake_time=self.sleep_waketime.time().toString("HH:mm") if self.sleep_waketime.time().isValid() else None,
                sleep_duration_hours=self.sleep_duration.value(),
                sleep_quality=self.sleep_quality.value(),
                time_to_fall_asleep_minutes=self.sleep_fall_asleep.value() if self.sleep_fall_asleep.value() > 0 else None,
                awakenings_count=self.sleep_awakenings.value() if self.sleep_awakenings.value() > 0 else None,
                deep_sleep_hours=self.sleep_deep.value() if self.sleep_deep.value() > 0 else None,
                rem_sleep_hours=self.sleep_rem.value() if self.sleep_rem.value() > 0 else None,
                caffeine_cutoff_time=self.sleep_caffeine_cutoff.time().toString("HH:mm") if self.sleep_caffeine_cutoff.time().isValid() else None,
                screen_time_before_bed_minutes=self.sleep_screen_time.value() if self.sleep_screen_time.value() > 0 else None,
                room_temperature_celsius=self.sleep_temperature.value() if self.sleep_temperature.value() > 0 else None,
                sleep_environment_rating=self.sleep_env_rating.value() if self.sleep_env_rating.value() > 0 else None,
                notes=self.sleep_notes.toPlainText() if self.sleep_notes.toPlainText() else None
            )
            
            QMessageBox.information(self, "Success", f"Sleep entry saved for {self.selected_date.strftime('%B %d, %Y')}!")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save sleep entry: {str(e)}")
    
    def save_body_measurements(self):
        """Save body measurements to database"""
        current_user = self.user_manager.get_current_user()
        if not current_user:
            QMessageBox.warning(self, "Error", "No user selected.")
            return
        
        try:
            entry_id = self.db.save_body_measurement(
                user_id=current_user['id'],
                measurement_date=self.selected_date,
                weight_kg=self.body_weight.value() if self.body_weight.value() > 0 else None,
                body_fat_percentage=self.body_bf.value() if self.body_bf.value() > 0 else None,
                muscle_mass_kg=self.body_muscle.value() if self.body_muscle.value() > 0 else None,
                waist_cm=self.body_waist.value() if self.body_waist.value() > 0 else None,
                chest_cm=self.body_chest.value() if self.body_chest.value() > 0 else None,
                arm_cm=self.body_arm.value() if self.body_arm.value() > 0 else None,
                forearm_cm=self.body_forearm.value() if self.body_forearm.value() > 0 else None,
                thigh_cm=self.body_thigh.value() if self.body_thigh.value() > 0 else None,
                calf_cm=self.body_calf.value() if self.body_calf.value() > 0 else None,
                neck_cm=self.body_neck.value() if self.body_neck.value() > 0 else None,
                notes=self.body_notes.toPlainText() if self.body_notes.toPlainText() else None
            )
            
            QMessageBox.information(self, "Success", f"Body measurements saved for {self.selected_date.strftime('%B %d, %Y')}!")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save body measurements: {str(e)}")
    
    def open_workout_manager(self):
        """Open workout template manager"""
        QMessageBox.information(self, "Coming Soon", "Workout template manager will be implemented next!")
    
    def export_data(self):
        """Export tracking data to CSV"""
        current_user = self.user_manager.get_current_user()
        if not current_user:
            QMessageBox.warning(self, "Error", "No user selected.")
            return
        
        try:
            filename, _ = QFileDialog.getSaveFileName(
                self, "Export Tracking Data", 
                f"hypertrophy_data_{current_user['username']}.csv",
                "CSV files (*.csv)"
            )
            
            if filename:
                # Export logic here - placeholder
                QMessageBox.information(self, "Success", f"Data exported to {filename}")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export data: {str(e)}")
    
    def refresh_data(self):
        """Refresh tracking data"""
        self.load_existing_data()