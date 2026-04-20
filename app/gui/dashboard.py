"""
Scientific Hypertrophy Trainer - Dashboard
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame, 
    QListWidget, QListWidgetItem
)
from PyQt6.QtCore import Qt, pyqtSignal
from datetime import datetime

class DashboardWidget(QWidget):
    start_assessment = pyqtSignal()
    open_tracking = pyqtSignal()
    
    def __init__(self, user_manager, tracking_system):
        super().__init__()
        self.user_manager = user_manager
        self.db = user_manager.db 
        self.init_ui()
    
    def init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(40, 40, 40, 40)
        main_layout.setSpacing(30)
        
        # --- HEADER ---
        self.header_frame = QFrame()
        self.header_frame.setStyleSheet("background: transparent; border: none;")
        header_layout = QVBoxLayout(self.header_frame)
        header_layout.setContentsMargins(0, 0, 0, 0)
        
        self.lbl_welcome = QLabel("Welcome back, Athlete")
        self.lbl_welcome.setStyleSheet("font-size: 36px; font-weight: 900; color: #cdd6f4; letter-spacing: 1px;")
        
        self.lbl_date = QLabel(datetime.now().strftime("%A, %B %d").upper())
        self.lbl_date.setStyleSheet("color: #89b4fa; font-size: 14px; font-weight: 900; letter-spacing: 2px;")
        
        header_layout.addWidget(self.lbl_welcome)
        header_layout.addWidget(self.lbl_date)
        main_layout.addWidget(self.header_frame)
        
        # --- STATS GRID ---
        stats_container = QWidget()
        stats_layout = QHBoxLayout(stats_container)
        stats_layout.setSpacing(20)
        stats_layout.setContentsMargins(0, 0, 0, 0)
        
        self.card_tier = self.create_stat_card("KNOWLEDGE TIER", "1", "Novice", "#fab387", "rgba(250, 179, 135, 0.1)")
        self.card_volume = self.create_stat_card("WEEKLY VOLUME", "0", "Sets Logged", "#a6e3a1", "rgba(166, 227, 161, 0.1)")
        self.card_sleep = self.create_stat_card("AVG RECOVERY", "-", "Sleep Score", "#cba6f7", "rgba(203, 166, 247, 0.1)")
        self.card_cals = self.create_stat_card("AVG INTAKE", "0", "kcal / day", "#f38ba8", "rgba(243, 139, 168, 0.1)")
        
        stats_layout.addWidget(self.card_tier)
        stats_layout.addWidget(self.card_volume)
        stats_layout.addWidget(self.card_sleep)
        stats_layout.addWidget(self.card_cals)
        main_layout.addWidget(stats_container)
        
        # --- LOWER SECTION ---
        lower_container = QHBoxLayout()
        lower_container.setSpacing(30)
        
        # LEFT: Quick Actions
        action_frame = QFrame()
        action_layout = QVBoxLayout(action_frame)
        action_layout.setContentsMargins(25, 25, 25, 25)
        action_layout.setSpacing(15)
        
        action_layout.addWidget(self.create_section_label("QUICK ACTIONS"))
        
        btn_log = QPushButton("🏋️  Log Training Session")
        btn_log.setStyleSheet("background-color: #89b4fa; color: #11111b; font-weight: 900; padding: 18px; font-size: 15px;")
        btn_log.clicked.connect(self.open_tracking.emit)
        
        btn_diet = QPushButton("🥑  Log Nutrition & Macros")
        btn_diet.setStyleSheet("background-color: #313244; color: #cdd6f4; font-weight: bold; padding: 18px; font-size: 15px;")
        btn_diet.clicked.connect(self.open_tracking.emit)
        
        btn_assess = QPushButton("🏆  Take Assessment Exam")
        btn_assess.setStyleSheet("background-color: #313244; color: #cdd6f4; font-weight: bold; padding: 18px; font-size: 15px;")
        btn_assess.clicked.connect(self.start_assessment.emit)
        
        action_layout.addWidget(btn_log)
        action_layout.addWidget(btn_diet)
        action_layout.addWidget(btn_assess)
        action_layout.addStretch()
        lower_container.addWidget(action_frame, 1)
        
        # RIGHT: Feed
        feed_frame = QFrame()
        feed_layout = QVBoxLayout(feed_frame)
        feed_layout.setContentsMargins(25, 25, 25, 25)
        
        feed_layout.addWidget(self.create_section_label("RECENT ACTIVITY LOG"))
        
        self.activity_list = QListWidget()
        self.activity_list.setStyleSheet("""
            QListWidget { background: transparent; border: none; outline: none; }
            QListWidget::item { 
                background-color: #1e1e2e; 
                margin-bottom: 12px; 
                padding: 18px; 
                border-radius: 12px; 
                color: #cdd6f4; 
                font-size: 14px;
                font-weight: bold;
            }
            QListWidget::item:hover { background-color: #262639; }
        """)
        feed_layout.addWidget(self.activity_list)
        lower_container.addWidget(feed_frame, 2)
        
        main_layout.addLayout(lower_container)

    def create_stat_card(self, title, value, unit, color, bg_glow):
        card = QFrame()
        card.setStyleSheet(f"""
            QFrame {{ background-color: #181825; border-radius: 16px; border-top: 4px solid {color}; }}
            QFrame:hover {{ background-color: {bg_glow}; }}
        """)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 20, 20, 20)
        
        lbl_title = QLabel(title)
        lbl_title.setStyleSheet(f"color: {color}; font-size: 11px; font-weight: 900; letter-spacing: 1.5px; border: none; background: transparent;")
        
        lbl_val = QLabel(value)
        lbl_val.setStyleSheet("color: white; font-size: 38px; font-weight: 900; border: none; background: transparent; margin-top: 10px;")
        
        lbl_unit = QLabel(unit)
        lbl_unit.setStyleSheet("color: #a6adc8; font-size: 13px; font-weight: bold; border: none; background: transparent;")
        
        layout.addWidget(lbl_title)
        layout.addWidget(lbl_val)
        layout.addWidget(lbl_unit)
        
        card.value_label = lbl_val
        card.unit_label = lbl_unit
        return card

    def create_section_label(self, text):
        lbl = QLabel(text)
        lbl.setStyleSheet("color: #6c7086; font-weight: 900; font-size: 12px; letter-spacing: 1.5px; margin-bottom: 15px; border: none; background: transparent;")
        return lbl

    def refresh_data(self):
        user = self.user_manager.get_current_user()
        if not user: return
        
        self.lbl_welcome.setText(f"Welcome back, {user['username']}!")
        
        tier = self.user_manager.get_current_tier()
        tier_names =["Novice", "Intermediate", "Advanced", "Elite"]
        self.card_tier.value_label.setText(str(tier + 1))
        self.card_tier.unit_label.setText(tier_names[tier] if tier < 4 else "Elite")
        
        cursor = self.db.conn.cursor()
        try:
            vol = cursor.execute("SELECT COUNT(*) FROM exercise_performances ep JOIN workout_sessions ws ON ep.workout_session_id = ws.id WHERE ws.user_id = ? AND ws.session_date >= date('now', '-7 days')", (user['id'],)).fetchone()[0]
            self.card_volume.value_label.setText(str(vol))
        except: self.card_volume.value_label.setText("0")

        try:
            sleep = cursor.execute("SELECT AVG(sleep_quality) FROM sleep_entries WHERE user_id = ? AND entry_date >= date('now', '-7 days')", (user['id'],)).fetchone()[0]
            self.card_sleep.value_label.setText(f"{sleep:.1f}" if sleep else "-")
        except: self.card_sleep.value_label.setText("-")

        try:
            cals = cursor.execute("SELECT AVG(total_calories) FROM diet_entries WHERE user_id = ? AND entry_date >= date('now', '-7 days')", (user['id'],)).fetchone()[0]
            self.card_cals.value_label.setText(str(int(cals)) if cals else "0")
        except: self.card_cals.value_label.setText("0")

        self.update_activity_feed(user['id'])

    def update_activity_feed(self, user_id):
        self.activity_list.clear()
        cursor = self.db.conn.cursor()
        workouts = cursor.execute("SELECT session_date, 'Workout', 'Logged a Training Session' FROM workout_sessions WHERE user_id=? ORDER BY session_date DESC LIMIT 3", (user_id,)).fetchall()
        diets = cursor.execute("SELECT entry_date, 'Nutrition', total_calories || ' kcal logged' FROM diet_entries WHERE user_id=? ORDER BY entry_date DESC LIMIT 3", (user_id,)).fetchall()
        
        activities = [(w[0], "💪", "Training Session Completed", "#a6e3a1") for w in workouts] + [(d[0], "🥑", f"Nutrition: {d[2]}", "#f38ba8") for d in diets]
        activities.sort(key=lambda x: x[0], reverse=True)
        
        if not activities:
            self.activity_list.addItem("No recent activity found.")
            return

        for date_str, icon, desc, color in activities[:6]:
            item = QListWidgetItem(f" {icon}    {date_str}   |   {desc}")
            self.activity_list.addItem(item)
