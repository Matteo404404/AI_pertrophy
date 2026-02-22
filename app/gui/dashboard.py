"""
Scientific Hypertrophy Trainer - Dashboard v3.0 (Pro HUD)
- Dark Theme "Command Center"
- Real-time Biometrics (Volume, Sleep, Calories)
- Recent Activity Feed
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame, 
    QGridLayout, QProgressBar, QScrollArea, QListWidget, QListWidgetItem
)
from PyQt6.QtCore import Qt, pyqtSignal, QSize
from datetime import datetime, timedelta

class DashboardWidget(QWidget):
    # Signals to switch tabs in MainWindow
    start_assessment = pyqtSignal()
    open_tracking = pyqtSignal()
    
    def __init__(self, user_manager, tracking_system):
        super().__init__()
        self.user_manager = user_manager
        self.tracking_system = tracking_system # Note: We might access DB directly for speed
        self.db = user_manager.db 
        
        self.init_ui()
    
    def init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(30, 30, 30, 30)
        main_layout.setSpacing(25)
        
        # --- HEADER ---
        self.header_frame = QFrame()
        header_layout = QVBoxLayout(self.header_frame)
        self.lbl_welcome = QLabel("Welcome back, Athlete")
        self.lbl_welcome.setStyleSheet("font-size: 32px; font-weight: 900; color: #89b4fa; letter-spacing: 1px;")
        
        self.lbl_date = QLabel(datetime.now().strftime("%A, %B %d"))
        self.lbl_date.setStyleSheet("color: #a6adc8; font-size: 16px; font-weight: bold;")
        
        header_layout.addWidget(self.lbl_welcome)
        header_layout.addWidget(self.lbl_date)
        main_layout.addWidget(self.header_frame)
        
        # --- STATS GRID (The HUD) ---
        stats_container = QFrame()
        stats_container.setStyleSheet("background-color: transparent;")
        stats_layout = QHBoxLayout(stats_container)
        stats_layout.setSpacing(20)
        stats_layout.setContentsMargins(0, 0, 0, 0)
        
        # 1. Knowledge Tier
        self.card_tier = self.create_stat_card("KNOWLEDGE TIER", "1", "Novice", "#fab387")
        stats_layout.addWidget(self.card_tier)
        
        # 2. Weekly Volume
        self.card_volume = self.create_stat_card("WEEKLY VOLUME", "0", "Sets", "#a6e3a1")
        stats_layout.addWidget(self.card_volume)
        
        # 3. Recovery Status
        self.card_sleep = self.create_stat_card("AVG RECOVERY", "-", "Sleep Score", "#cba6f7")
        stats_layout.addWidget(self.card_sleep)
        
        # 4. Nutrition
        self.card_cals = self.create_stat_card("AVG INTAKE", "0", "kcal/day", "#f38ba8")
        stats_layout.addWidget(self.card_cals)
        
        main_layout.addWidget(stats_container)
        
        # --- LOWER SECTION (Split) ---
        lower_container = QHBoxLayout()
        lower_container.setSpacing(20)
        
        # LEFT: Quick Actions
        action_frame = QFrame()
        action_frame.setStyleSheet("background-color: #181825; border-radius: 12px; border: 1px solid #313244;")
        action_layout = QVBoxLayout(action_frame)
        
        action_layout.addWidget(self.create_section_label("QUICK ACTIONS"))
        
        btn_log = QPushButton("📝  Log Workout")
        btn_log.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_log.setStyleSheet("background-color: #89b4fa; color: #1e1e2e; font-weight: bold; padding: 15px; border-radius: 8px; font-size: 14px;")
        btn_log.clicked.connect(self.open_tracking.emit)
        
        btn_diet = QPushButton("🥑  Track Nutrition")
        btn_diet.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_diet.setStyleSheet("background-color: #313244; color: #cdd6f4; font-weight: bold; padding: 15px; border-radius: 8px; font-size: 14px; border: 1px solid #45475a;")
        btn_diet.clicked.connect(self.open_tracking.emit)
        
        btn_assess = QPushButton("🏆  Take Assessment")
        btn_assess.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_assess.setStyleSheet("background-color: #313244; color: #cdd6f4; font-weight: bold; padding: 15px; border-radius: 8px; font-size: 14px; border: 1px solid #45475a;")
        btn_assess.clicked.connect(self.start_assessment.emit)
        
        action_layout.addWidget(btn_log)
        action_layout.addWidget(btn_diet)
        action_layout.addWidget(btn_assess)
        action_layout.addStretch()
        
        lower_container.addWidget(action_frame, 1)
        
        # RIGHT: Recent Activity Feed
        feed_frame = QFrame()
        feed_frame.setStyleSheet("background-color: #181825; border-radius: 12px; border: 1px solid #313244;")
        feed_layout = QVBoxLayout(feed_frame)
        
        feed_layout.addWidget(self.create_section_label("RECENT ACTIVITY"))
        
        self.activity_list = QListWidget()
        self.activity_list.setStyleSheet("""
            QListWidget { background: transparent; border: none; outline: none; }
            QListWidget::item { 
                background-color: #1e1e2e; 
                margin-bottom: 8px; 
                padding: 12px; 
                border-radius: 6px; 
                color: #cdd6f4; 
                border-left: 3px solid #45475a;
            }
        """)
        feed_layout.addWidget(self.activity_list)
        
        lower_container.addWidget(feed_frame, 2)
        main_layout.addLayout(lower_container)

    # --- HELPERS ---
    def create_stat_card(self, title, value, unit, color):
        card = QFrame()
        card.setStyleSheet(f"""
            QFrame {{
                background-color: #181825;
                border-radius: 12px;
                border: 1px solid #313244;
            }}
        """)
        layout = QVBoxLayout(card)
        
        lbl_title = QLabel(title)
        lbl_title.setStyleSheet(f"color: {color}; font-size: 11px; font-weight: bold; letter-spacing: 1px;")
        
        lbl_val = QLabel(value)
        lbl_val.setStyleSheet("color: white; font-size: 32px; font-weight: 900;")
        
        lbl_unit = QLabel(unit)
        lbl_unit.setStyleSheet("color: #a6adc8; font-size: 12px;")
        
        layout.addWidget(lbl_title)
        layout.addWidget(lbl_val)
        layout.addWidget(lbl_unit)
        
        # Store ref to value label for updating
        card.value_label = lbl_val
        card.unit_label = lbl_unit
        
        return card

    def create_section_label(self, text):
        lbl = QLabel(text)
        lbl.setStyleSheet("color: #a6adc8; font-weight: bold; font-size: 12px; margin-bottom: 10px;")
        return lbl

    # --- LOGIC ---
    def refresh_data(self):
        user = self.user_manager.get_current_user()
        if not user: return
        
        # 1. Update Header
        self.lbl_welcome.setText(f"Welcome back, {user['username']}!")
        
        # 2. Update Stats (Fetch real data)
        # Tier
        tier = self.user_manager.get_current_tier()
        tier_names = ["Novice", "Intermediate", "Advanced", "Elite"]
        self.card_tier.value_label.setText(str(tier + 1))
        self.card_tier.unit_label.setText(tier_names[tier] if tier < 4 else "Elite")
        
        # Volume (Last 7 days)
        # We need a quick DB query here. Using direct cursor for speed.
        cursor = self.db.conn.cursor()
        
        # Volume
        try:
            vol = cursor.execute("""
                SELECT COUNT(*) FROM exercise_performances ep
                JOIN workout_sessions ws ON ep.workout_session_id = ws.id
                WHERE ws.user_id = ? AND ws.session_date >= date('now', '-7 days')
            """, (user['id'],)).fetchone()[0]
            self.card_volume.value_label.setText(str(vol))
        except: self.card_volume.value_label.setText("0")

        # Sleep
        try:
            sleep = cursor.execute("""
                SELECT AVG(sleep_quality) FROM sleep_entries
                WHERE user_id = ? AND entry_date >= date('now', '-7 days')
            """, (user['id'],)).fetchone()[0]
            val = f"{sleep:.1f}" if sleep else "-"
            self.card_sleep.value_label.setText(val)
        except: self.card_sleep.value_label.setText("-")

        # Cals
        try:
            cals = cursor.execute("""
                SELECT AVG(total_calories) FROM diet_entries
                WHERE user_id = ? AND entry_date >= date('now', '-7 days')
            """, (user['id'],)).fetchone()[0]
            val = str(int(cals)) if cals else "0"
            self.card_cals.value_label.setText(val)
        except: self.card_cals.value_label.setText("0")

        # 3. Update Activity Feed
        self.update_activity_feed(user['id'])

    def update_activity_feed(self, user_id):
        self.activity_list.clear()
        
        # Fetch last 5 activities (Union of Workouts, Diet, Sleep)
        # Simplified: Just fetching workouts for now to avoid complex union query strings
        cursor = self.db.conn.cursor()
        
        # Workouts
        workouts = cursor.execute("""
            SELECT session_date, 'Workout', 'Logged a session' 
            FROM workout_sessions WHERE user_id=? ORDER BY session_date DESC LIMIT 3
        """, (user_id,)).fetchall()
        
        # Diet
        diets = cursor.execute("""
            SELECT entry_date, 'Nutrition', total_calories || ' kcal logged'
            FROM diet_entries WHERE user_id=? ORDER BY entry_date DESC LIMIT 3
        """, (user_id,)).fetchall()
        
        # Combine and Sort
        activities = []
        for w in workouts: activities.append((w[0], "🏋️", "Workout Logged"))
        for d in diets: activities.append((d[0], "🥑", f"Nutrition: {d[2]}"))
        
        # Sort by date (desc)
        activities.sort(key=lambda x: x[0], reverse=True)
        
        if not activities:
            self.activity_list.addItem("No recent activity.")
            return

        for date_str, icon, desc in activities[:5]:
            item = QListWidgetItem(f"{icon}  {date_str} — {desc}")
            self.activity_list.addItem(item)