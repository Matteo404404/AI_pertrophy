"""
Scientific Hypertrophy Trainer - Main Window
- Professional Dark Sidebar
- Navigation Logic
- Layout Management
"""
from gui.analytics import AnalyticsWidget
import sys
import os
from PyQt6.QtWidgets import (QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, 
                             QPushButton, QStackedWidget, QLabel, QFrame,
                             QApplication, QMessageBox, QButtonGroup)
from PyQt6.QtCore import Qt, pyqtSignal, QSize
from PyQt6.QtGui import QIcon, QFont

# Import Modules
from gui.user_selection import UserSelectionWidget
from gui.dashboard import DashboardWidget
from gui.assessment import AssessmentWidget
from gui.tracking import TrackingWidget
from gui.learning import LearningWidget

from core.user_manager import UserManager
from database.db_manager import DatabaseManager
from core.assessment_engine import AssessmentEngine
from core.tracking_system import TrackingSystem

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Core Components
        self.db = DatabaseManager()
        self.user_manager = UserManager(self.db)
        self.assessment_engine = AssessmentEngine()
        self.tracking_system = TrackingSystem(self.db)
        
        self.setWindowTitle("Scientific Hypertrophy Trainer")
        self.resize(1400, 900)
        
        # Apply Base Style
        self.setStyleSheet("background-color: #1e1e2e; color: #cdd6f4;")
        
        # Main Layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.main_layout = QHBoxLayout(central_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)
        
        # 1. Create Sidebar (Hidden initially)
        self.create_sidebar()
        self.main_layout.addWidget(self.sidebar)
        
        # 2. Content Area
        self.content_stack = QStackedWidget()
        self.main_layout.addWidget(self.content_stack)
        
        # 3. Initialize Screens
        self.create_screens()
        
        # Start at User Selection
        self.sidebar.hide()
        self.content_stack.setCurrentWidget(self.user_selection)

    def create_sidebar(self):
        """Creates a modern, flat, professional sidebar"""
        self.sidebar = QFrame()
        self.sidebar.setFixedWidth(260)
        self.sidebar.setStyleSheet("""
            QFrame {
                background-color: #11111b; /* Very Dark Navy */
                border-right: 1px solid #313244;
            }
            QLabel {
                color: #a6adc8;
                font-weight: bold;
                padding: 5px;
            }
        """)
        
        layout = QVBoxLayout(self.sidebar)
        layout.setContentsMargins(15, 30, 15, 30)
        layout.setSpacing(10)
        
        # --- App Branding ---
        brand_lbl = QLabel("HYPERTROPHY\nSCIENTIST")
        brand_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        brand_lbl.setStyleSheet("font-size: 20px; font-weight: 900; color: #89b4fa; letter-spacing: 2px; margin-bottom: 20px;")
        layout.addWidget(brand_lbl)
        
        # --- User Info Box ---
        self.user_card = QFrame()
        self.user_card.setStyleSheet("background-color: #181825; border-radius: 8px; padding: 10px;")
        uc_layout = QVBoxLayout(self.user_card)
        self.lbl_username = QLabel("Guest")
        self.lbl_username.setStyleSheet("color: white; font-size: 14px;")
        self.lbl_status = QLabel("Not Logged In")
        self.lbl_status.setStyleSheet("color: #fab387; font-size: 11px;")
        uc_layout.addWidget(self.lbl_username)
        uc_layout.addWidget(self.lbl_status)
        layout.addWidget(self.user_card)
        
        layout.addSpacing(20)
        layout.addWidget(QLabel("MAIN MENU"))
        
        # --- Navigation Buttons ---
        self.nav_group = QButtonGroup()
        self.nav_group.setExclusive(True)
        
        self.btn_dash = self.create_nav_btn("📊  Dashboard", self.show_dashboard)
        self.btn_track = self.create_nav_btn("📝  Tracking & Logs", self.show_tracking)
        self.btn_analytics = self.create_nav_btn("📈  Analytics", self.show_analytics)
        self.btn_learn = self.create_nav_btn("🎓  Knowledge Base", self.show_learning)
        self.btn_assess = self.create_nav_btn("🏆  Assessment", self.show_assessment)
        
        layout.addWidget(self.btn_dash)
        layout.addWidget(self.btn_track)
        layout.addWidget(self.btn_analytics)
        layout.addWidget(self.btn_learn)
        layout.addWidget(self.btn_assess)
        
        layout.addStretch()
        
        # --- Bottom Actions ---
        layout.addWidget(QLabel("SESSION"))
        btn_logout = self.create_nav_btn("👤  Switch User", self.logout_user, is_bottom=True)
        layout.addWidget(btn_logout)

    def create_nav_btn(self, text, callback, is_bottom=False):
        btn = QPushButton(text)
        btn.setCheckable(True)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.clicked.connect(callback)
        
        # Modern CSS for buttons
        base_color = "#f38ba8" if is_bottom else "#cdd6f4" # Red for logout, White for others
        btn.setStyleSheet(f"""
            QPushButton {{
                text-align: left;
                padding: 12px 15px;
                border: none;
                border-radius: 8px;
                color: {base_color};
                font-size: 14px;
                background-color: transparent;
            }}
            QPushButton:hover {{
                background-color: #313244;
            }}
            QPushButton:checked {{
                background-color: #313244;
                color: #89b4fa; /* Accent Blue */
                border-left: 3px solid #89b4fa;
                font-weight: bold;
            }}
        """)
        
        if not is_bottom:
            self.nav_group.addButton(btn)
            
        return btn

    def create_screens(self):
        self.user_selection = UserSelectionWidget(self.db, self.user_manager)
        self.user_selection.user_selected.connect(self.on_user_selected)
        self.content_stack.addWidget(self.user_selection)
        
        self.dashboard = DashboardWidget(self.user_manager, self.tracking_system)
        self.dashboard.start_assessment.connect(self.show_assessment)
        self.dashboard.open_tracking.connect(self.show_tracking)
        self.content_stack.addWidget(self.dashboard)
        
        self.assessment = AssessmentWidget(self.assessment_engine, self.user_manager)
        self.assessment.assessment_completed.connect(self.on_assessment_completed)
        self.content_stack.addWidget(self.assessment)
        
        self.tracking = TrackingWidget(self.db, self.tracking_system, self.user_manager)
        self.content_stack.addWidget(self.tracking)

        self.analytics = AnalyticsWidget(self.db, self.tracking_system, self.user_manager)
        self.content_stack.addWidget(self.analytics)

        self.learning = LearningWidget(self.db, self.user_manager)
        self.content_stack.addWidget(self.learning)

    # --- Navigation Logic ---
    def show_dashboard(self):
        self.content_stack.setCurrentWidget(self.dashboard)
        self.dashboard.refresh_data()
        self.btn_dash.setChecked(True)

    def show_tracking(self):
        self.content_stack.setCurrentWidget(self.tracking)
        self.tracking.refresh_data()
        self.btn_track.setChecked(True)

    def show_analytics(self):
        self.content_stack.setCurrentWidget(self.analytics)
        self.analytics.refresh_data()
        self.btn_analytics.setChecked(True)

    def show_learning(self):
        self.content_stack.setCurrentWidget(self.learning)
        self.learning.refresh_data()
        self.btn_learn.setChecked(True)

    def show_assessment(self):
        self.content_stack.setCurrentWidget(self.assessment)
        self.assessment.refresh_tiers()
        self.btn_assess.setChecked(True)

    def on_user_selected(self, user_id):
        if self.user_manager.set_current_user(user_id):
            user = self.user_manager.get_current_user()
            tier = self.user_manager.get_current_tier() + 1
            
            # Update Sidebar Info
            self.lbl_username.setText(user['username'])
            self.lbl_status.setText(f"Level: {user['experience_level'].capitalize()} • Tier {tier}")
            
            self.sidebar.show()
            self.show_dashboard()

    def logout_user(self):
        reply = QMessageBox.question(self, 'Logout', 'Switch user?', QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            self.sidebar.hide()
            self.content_stack.setCurrentWidget(self.user_selection)
            self.user_manager.current_user = None

    def on_assessment_completed(self, results):
        self.user_manager.save_assessment_result(results)
        self.show_dashboard()