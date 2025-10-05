"""
Scientific Hypertrophy Trainer - Main Window with Fixed Light Theme
PyQt6 main application window with properly visible labels
"""

import sys
from PyQt6.QtWidgets import (QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, 
                             QPushButton, QStackedWidget, QLabel, QFrame,
                             QApplication, QMessageBox)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QIcon, QPixmap, QPainter, QColor

from database.db_manager import DatabaseManager
from core.user_manager import UserManager
from core.assessment_engine import AssessmentEngine
from core.tracking_system import TrackingSystem

from .user_selection import UserSelectionWidget
from .dashboard import DashboardWidget
from .assessment import AssessmentWidget
from .tracking import TrackingWidget
from .learning import LearningWidget


class MainWindow(QMainWindow):
    """Main application window with sidebar navigation"""
    
    def __init__(self):
        super().__init__()
        
        # Initialize core components
        self.db = DatabaseManager()
        self.user_manager = UserManager(self.db)
        self.assessment_engine = AssessmentEngine()
        self.tracking_system = TrackingSystem(self.db)
        
        # Apply stylesheet FIRST
        self.apply_embedded_stylesheet()
        
        # Initialize UI
        self.init_ui()
        
        # Start with user selection
        self.show_user_selection()

    def apply_embedded_stylesheet(self):
        """Apply beautiful modern light theme with proper label visibility"""
        stylesheet = """
/* ========== BASE STYLES ========== */
QMainWindow {
    background-color: #f8f9fa;
}

QWidget {
    background-color: transparent;
    color: #1a1a1a;
}

/* ========== LABELS - ALWAYS VISIBLE ========== */
QLabel {
    color: #1a1a1a;
    background-color: transparent;
}

/* ========== FRAMES ========== */
QFrame {
    background-color: #ffffff;
    border: 1px solid #e9ecef;
    border-radius: 12px;
}

/* ========== SIDEBAR ========== */
#sidebar {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #667eea, stop:1 #764ba2);
    border-right: none;
}

#sidebar QLabel {
    color: #ffffff;
}

#sidebar QPushButton {
    background-color: rgba(255, 255, 255, 0.1);
    border: none;
    padding: 14px 16px;
    text-align: left;
    color: #ffffff;
    font-size: 15px;
    border-radius: 8px;
    margin: 4px 12px;
}

#sidebar QPushButton:hover {
    background-color: rgba(255, 255, 255, 0.2);
}

#sidebar QPushButton:checked {
    background-color: rgba(255, 255, 255, 0.95);
    color: #667eea;
    font-weight: 600;
}

/* ========== CONTENT AREA ========== */
#content_area {
    background-color: #f8f9fa;
}

/* ========== BUTTONS ========== */
QPushButton {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #667eea, stop:1 #764ba2);
    color: #ffffff;
    border: none;
    padding: 12px 24px;
    border-radius: 8px;
    font-size: 14px;
    font-weight: 600;
}

QPushButton:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #5568d3, stop:1 #6941a5);
}

QPushButton:pressed {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #4e5fc1, stop:1 #5d3a93);
}

QPushButton:disabled {
    background-color: #e9ecef;
    color: #adb5bd;
}

/* ========== INPUT FIELDS ========== */
QLineEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QComboBox, QTimeEdit {
    background-color: #f8f9fa;
    border: 2px solid #e9ecef;
    border-radius: 8px;
    padding: 10px 14px;
    color: #1a1a1a;
    font-size: 14px;
}

QLineEdit:focus, QTextEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus, QTimeEdit:focus {
    border-color: #667eea;
    background-color: #ffffff;
}

QLineEdit::placeholder {
    color: #adb5bd;
}

/* ========== SPINBOX BUTTONS ========== */
QSpinBox::up-button, QDoubleSpinBox::up-button {
    background-color: #e9ecef;
    border: none;
    border-radius: 0px 6px 0px 0px;
    width: 20px;
}

QSpinBox::down-button, QDoubleSpinBox::down-button {
    background-color: #e9ecef;
    border: none;
    border-radius: 0px 0px 6px 0px;
    width: 20px;
}

QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {
    background-color: #667eea;
}

QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-bottom: 5px solid #495057;
}

QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid #495057;
}

/* ========== COMBOBOX ========== */
QComboBox::drop-down {
    border: none;
    width: 30px;
    background-color: #e9ecef;
    border-left: 2px solid #dee2e6;
    border-radius: 0px 6px 6px 0px;
}

QComboBox::down-arrow {
    image: none;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 6px solid #495057;
    margin-right: 10px;
}

QComboBox QAbstractItemView {
    background-color: #ffffff;
    border: 2px solid #e9ecef;
    border-radius: 8px;
    selection-background-color: #667eea;
    selection-color: #ffffff;
    padding: 4px;
}

QComboBox QAbstractItemView::item {
    padding: 8px;
    border-radius: 6px;
}

/* ========== GROUP BOXES ========== */
QGroupBox {
    background-color: #f8f9fa;
    border: 2px solid #e9ecef;
    border-radius: 10px;
    padding: 20px;
    margin-top: 14px;
    font-weight: 600;
    font-size: 15px;
    color: #495057;
}

QGroupBox::title {
    color: #667eea;
    subcontrol-origin: margin;
    left: 16px;
    padding: 0px 10px;
    background-color: #f8f9fa;
}

/* ========== PROGRESS BARS ========== */
QProgressBar {
    background-color: #e9ecef;
    border: none;
    border-radius: 8px;
    height: 10px;
    text-align: center;
    color: #ffffff;
    font-weight: 600;
}

QProgressBar::chunk {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #667eea, stop:1 #764ba2);
    border-radius: 8px;
}

/* ========== TABS ========== */
QTabWidget::pane {
    background-color: #ffffff;
    border: 2px solid #e9ecef;
    border-radius: 10px;
    padding: 10px;
}

QTabBar::tab {
    background-color: #f8f9fa;
    color: #6c757d;
    padding: 12px 20px;
    margin-right: 6px;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
    font-size: 14px;
    border: 2px solid #e9ecef;
    border-bottom: none;
}

QTabBar::tab:selected {
    background-color: #ffffff;
    color: #667eea;
    font-weight: 600;
    border-bottom: 3px solid #667eea;
}

QTabBar::tab:hover {
    background-color: #e9ecef;
    color: #495057;
}

/* ========== LIST WIDGETS ========== */
QListWidget {
    background-color: #f8f9fa;
    border: 2px solid #e9ecef;
    border-radius: 10px;
    padding: 4px;
}

QListWidget::item {
    background-color: #ffffff;
    border: 2px solid #e9ecef;
    padding: 16px;
    border-radius: 10px;
    margin: 6px;
    color: #1a1a1a;
}

QListWidget::item:hover {
    background-color: #f8f9fa;
    border-color: #667eea;
}

QListWidget::item:selected {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #667eea, stop:1 #764ba2);
    border-color: #667eea;
    color: #ffffff;
}

/* ========== RADIO BUTTONS ========== */
QRadioButton {
    color: #1a1a1a;
    spacing: 10px;
    padding: 8px;
}

QRadioButton::indicator {
    width: 20px;
    height: 20px;
    border-radius: 10px;
    border: 3px solid #e9ecef;
    background-color: #ffffff;
}

QRadioButton::indicator:checked {
    border-color: #667eea;
    background-color: #667eea;
}

QRadioButton::indicator:hover {
    border-color: #667eea;
}

/* ========== SCROLL BARS ========== */
QScrollBar:vertical {
    background-color: #f8f9fa;
    width: 12px;
    border-radius: 6px;
}

QScrollBar::handle:vertical {
    background-color: #dee2e6;
    border-radius: 6px;
    min-height: 30px;
}

QScrollBar::handle:vertical:hover {
    background-color: #adb5bd;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}

QScrollBar:horizontal {
    background-color: #f8f9fa;
    height: 12px;
    border-radius: 6px;
}

QScrollBar::handle:horizontal {
    background-color: #dee2e6;
    border-radius: 6px;
    min-width: 30px;
}

QScrollBar::handle:horizontal:hover {
    background-color: #adb5bd;
}

/* ========== SCROLL AREA ========== */
QScrollArea {
    background-color: transparent;
    border: none;
}

/* ========== TABLE WIDGET ========== */
QTableWidget {
    background-color: #ffffff;
    border: 2px solid #e9ecef;
    border-radius: 10px;
    gridline-color: #e9ecef;
}

QTableWidget::item {
    padding: 10px;
    color: #1a1a1a;
}

QTableWidget::item:selected {
    background-color: #667eea;
    color: #ffffff;
}

QHeaderView::section {
    background-color: #f8f9fa;
    color: #495057;
    padding: 10px;
    border: none;
    border-bottom: 2px solid #e9ecef;
    font-weight: 600;
}

/* ========== TOOL TIPS ========== */
QToolTip {
    background-color: #495057;
    color: #ffffff;
    border: none;
    border-radius: 6px;
    padding: 8px 12px;
    font-size: 13px;
}
        """
        self.setStyleSheet(stylesheet)
        print("✅ Applied fixed modern light theme")
    
             
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("Scientific Hypertrophy Trainer")
        self.setMinimumSize(1280, 720)
        self.resize(1400, 900)
        
        # Central widget setup
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Create sidebar
        self.create_sidebar()
        main_layout.addWidget(self.sidebar)
        
        # Create content area
        self.content_stack = QStackedWidget()
        self.content_stack.setObjectName("content_area")
        main_layout.addWidget(self.content_stack)
        
        # Initialize screens
        self.create_screens()
        
        # Initially hide sidebar (show only during main app)
        self.sidebar.hide()
        
    def create_sidebar(self):
        """Create the sidebar navigation"""
        self.sidebar = QFrame()
        self.sidebar.setObjectName("sidebar")
        self.sidebar.setFixedWidth(220)
        
        sidebar_layout = QVBoxLayout(self.sidebar)
        sidebar_layout.setContentsMargins(0, 20, 0, 20)
        sidebar_layout.setSpacing(5)
        
        # App title
        title_label = QLabel("Scientific\nHypertrophy\nTrainer")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("""
            font-size: 18px;
            font-weight: bold;
            padding: 20px 10px;
        """)
        sidebar_layout.addWidget(title_label)
        
        # Current user info
        self.current_user_label = QLabel("No user selected")
        self.current_user_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.current_user_label.setStyleSheet("""
            font-size: 12px;
            padding: 10px;
            background-color: rgba(255, 255, 255, 0.2);
            border: 1px solid rgba(255, 255, 255, 0.3);
            border-radius: 6px;
            margin: 0px 10px 20px 10px;
        """)
        sidebar_layout.addWidget(self.current_user_label)
        
        # Navigation buttons
        nav_buttons = [
            ("🏠", "Dashboard", self.show_dashboard),
            ("📋", "Assessment", self.show_assessment),
            ("📊", "Tracking", self.show_tracking),
            ("🎓", "Learning", self.show_learning),
            ("👤", "Switch User", self.logout_user)
        ]
        
        self.nav_buttons = {}
        for icon, text, callback in nav_buttons:
            btn = QPushButton(f"{icon}  {text}")
            btn.setCheckable(True)
            btn.clicked.connect(callback)
            btn.setMinimumHeight(50)
            sidebar_layout.addWidget(btn)
            self.nav_buttons[text] = btn
        
        sidebar_layout.addStretch()
        
        # App info
        info_label = QLabel("Version 1.0\nEvidence-based\nHypertrophy Training")
        info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        info_label.setStyleSheet("""
            font-size: 10px;
            padding: 10px;
        """)
        sidebar_layout.addWidget(info_label)
        
    def create_screens(self):
        """Create all application screens"""
        # User Selection Screen
        self.user_selection = UserSelectionWidget(self.db, self.user_manager)
        self.user_selection.user_selected.connect(self.on_user_selected)
        self.content_stack.addWidget(self.user_selection)
        
        # Dashboard Screen
        self.dashboard = DashboardWidget(self.user_manager, self.tracking_system)
        self.dashboard.start_assessment.connect(self.show_assessment)
        self.dashboard.open_tracking.connect(self.show_tracking)
        self.content_stack.addWidget(self.dashboard)
        
        # Assessment Screen
        self.assessment = AssessmentWidget(self.assessment_engine, self.user_manager)
        self.assessment.assessment_completed.connect(self.on_assessment_completed)
        self.content_stack.addWidget(self.assessment)
        
        # Tracking Screen
        self.tracking = TrackingWidget(self.db, self.tracking_system, self.user_manager)
        self.content_stack.addWidget(self.tracking)
        
        # Learning Screen
        self.learning = LearningWidget(self.db, self.user_manager)
        self.content_stack.addWidget(self.learning)
    
    # ===== NAVIGATION METHODS =====
    
    def show_user_selection(self):
        """Show user selection screen"""
        self.content_stack.setCurrentWidget(self.user_selection)
        self.sidebar.hide()
        
    def show_dashboard(self):
        """Show dashboard screen"""
        if not self.user_manager.get_current_user():
            self.show_user_selection()
            return
            
        self.content_stack.setCurrentWidget(self.dashboard)
        self.dashboard.refresh_data()
        self.set_active_nav_button("Dashboard")
        
    def show_assessment(self):
        """Show assessment screen"""
        if not self.user_manager.get_current_user():
            self.show_user_selection()
            return
            
        self.content_stack.setCurrentWidget(self.assessment)
        self.assessment.refresh_tiers()
        self.set_active_nav_button("Assessment")
        
    def show_tracking(self):
        """Show tracking screen"""
        if not self.user_manager.get_current_user():
            self.show_user_selection()
            return
            
        self.content_stack.setCurrentWidget(self.tracking)
        self.tracking.refresh_data()
        self.set_active_nav_button("Tracking")
        
    def show_learning(self):
        """Show learning screen"""
        if not self.user_manager.get_current_user():
            self.show_user_selection()
            return
            
        self.content_stack.setCurrentWidget(self.learning)
        self.learning.refresh_data()
        self.set_active_nav_button("Learning")
    
    def set_active_nav_button(self, button_name):
        """Set active navigation button"""
        for name, btn in self.nav_buttons.items():
            btn.setChecked(name == button_name)
    
    def logout_user(self):
        """Logout current user"""
        reply = QMessageBox.question(
            self, 'Switch User', 
            'Are you sure you want to switch users?\nAll changes will be saved automatically.',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.user_manager.current_user = None
            self.show_user_selection()
    
    # ===== EVENT HANDLERS =====
    
    def on_user_selected(self, user_id):
        """Handle user selection"""
        if self.user_manager.set_current_user(user_id):
            user = self.user_manager.get_current_user()
            self.current_user_label.setText(f"{user['username']}\nTier {self.user_manager.get_current_tier() + 1}")
            self.sidebar.show()
            self.show_dashboard()
        else:
            QMessageBox.critical(self, "Error", "Failed to load user data.")
    
    def on_assessment_completed(self, results):
        """Handle assessment completion"""
        # Save results to database
        try:
            self.user_manager.save_assessment_result(results)
            
            # Show results message
            message = f"Assessment Complete!\n\n"
            message += f"Score: {results['score']}/{results['total_questions']} ({results['percentage']:.1f}%)\n"
            message += f"Status: {'PASSED ✅' if results['passed'] else 'FAILED ❌'}\n"
            
            if results['passed'] and results.get('next_tier_unlocked'):
                message += f"\n🎉 Congratulations! You've unlocked the next tier!"
            
            QMessageBox.information(self, "Assessment Results", message)
            
            # Refresh dashboard data
            self.dashboard.refresh_data()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save assessment results: {str(e)}")
    
    def closeEvent(self, event):
        """Handle application close"""
        self.db.close()
        event.accept()


def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    app.setApplicationName("Scientific Hypertrophy Trainer")
    
    window = MainWindow()
    window.show()
    
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())