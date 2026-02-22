"""
Scientific Hypertrophy Trainer - User Selection v2.0 (Pro Dark Theme)
- "Command Center" Login Screen
- Dark Cards with Hover Effects
- Integrated Create Profile Form
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame, 
    QLineEdit, QComboBox, QDoubleSpinBox, QMessageBox, QScrollArea, 
    QFileDialog, QGridLayout, QSizePolicy
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QCursor
import json

class UserSelectionWidget(QWidget):
    user_selected = pyqtSignal(int)
    
    def __init__(self, db_manager, user_manager):
        super().__init__()
        self.db = db_manager
        self.user_manager = user_manager
        
        self.init_ui()
        self.refresh_users()
    
    def init_ui(self):
        # Global Layout
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)
        
        # --- BACKGROUND & CONTAINER ---
        # We split the screen: Left (User List) 60%, Right (Create/Tools) 40%
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(40)
        
        # LEFT PANEL: User List
        self.create_left_panel(layout)
        
        # RIGHT PANEL: Create User & Tools
        self.create_right_panel(layout)
        
        self.main_layout.addWidget(container)

    def create_left_panel(self, parent_layout):
        left_frame = QFrame()
        left_layout = QVBoxLayout(left_frame)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        # Header
        header = QLabel("WHO IS TRAINING?")
        header.setStyleSheet("font-size: 32px; font-weight: 900; color: #89b4fa; letter-spacing: 2px;")
        left_layout.addWidget(header)
        
        sub = QLabel("Select a profile to load biometrics and history.")
        sub.setStyleSheet("font-size: 14px; color: #a6adc8; margin-bottom: 20px;")
        left_layout.addWidget(sub)
        
        # Scroll Area for Users
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("background: transparent; border: none;")
        
        self.users_container = QWidget()
        self.users_layout = QVBoxLayout(self.users_container)
        self.users_layout.setSpacing(15)
        self.users_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        scroll.setWidget(self.users_container)
        left_layout.addWidget(scroll)
        
        parent_layout.addWidget(left_frame, 2) # Takes 2/3rds width

    def create_right_panel(self, parent_layout):
        right_frame = QFrame()
        right_frame.setStyleSheet("""
            QFrame {
                background-color: #181825; 
                border-radius: 16px; 
                border: 1px solid #313244;
            }
        """)
        layout = QVBoxLayout(right_frame)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(20)
        
        # Title
        lbl = QLabel("NEW ATHLETE")
        lbl.setStyleSheet("font-size: 20px; font-weight: bold; color: #a6e3a1; border: none;")
        layout.addWidget(lbl)
        
        # Form
        self.username_input = QLineEdit()
        self.username_input.setPlaceholderText("Username")
        layout.addWidget(self.create_label("Identifier"))
        layout.addWidget(self.username_input)
        
        self.experience_combo = QComboBox()
        self.experience_combo.addItems(['Beginner', 'Intermediate', 'Advanced'])
        layout.addWidget(self.create_label("Experience Level"))
        layout.addWidget(self.experience_combo)
        
        # Stats Grid
        stats_grid = QGridLayout()
        self.weight_input = self.create_spinbox(30, 300, 75.0, " kg")
        stats_grid.addWidget(self.create_label("Weight"), 0, 0)
        stats_grid.addWidget(self.weight_input, 1, 0)
        
        self.height_input = self.create_spinbox(120, 250, 175, " cm")
        stats_grid.addWidget(self.create_label("Height"), 0, 1)
        stats_grid.addWidget(self.height_input, 1, 1)
        
        self.bodyfat_input = self.create_spinbox(3, 60, 15.0, " %")
        stats_grid.addWidget(self.create_label("Body Fat"), 2, 0)
        stats_grid.addWidget(self.bodyfat_input, 3, 0)
        
        layout.addLayout(stats_grid)
        
        # Create Button
        create_btn = QPushButton("Initialize Profile")
        create_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        create_btn.clicked.connect(self.create_user)
        create_btn.setStyleSheet("""
            QPushButton {
                background-color: #a6e3a1; color: #1e1e2e; 
                font-weight: 800; padding: 15px; border-radius: 8px; font-size: 14px;
            }
            QPushButton:hover { background-color: #94e2d5; }
        """)
        layout.addWidget(create_btn)
        
        layout.addStretch()
        
        # Footer Tools
        tools_lbl = QLabel("DATA MANAGEMENT")
        tools_lbl.setStyleSheet("font-weight: bold; color: #6c757d; font-size: 11px; margin-top: 20px; border: none;")
        layout.addWidget(tools_lbl)
        
        tool_layout = QHBoxLayout()
        btn_import = QPushButton("Import")
        btn_import.clicked.connect(self.import_users)
        btn_import.setStyleSheet("background-color: #313244; color: #cdd6f4; padding: 8px; border-radius: 6px;")
        
        btn_export = QPushButton("Export")
        btn_export.clicked.connect(self.export_users)
        btn_export.setStyleSheet("background-color: #313244; color: #cdd6f4; padding: 8px; border-radius: 6px;")
        
        tool_layout.addWidget(btn_import)
        tool_layout.addWidget(btn_export)
        layout.addLayout(tool_layout)
        
        parent_layout.addWidget(right_frame, 1) # Takes 1/3rd width

    # --- CARD GENERATION ---
    def create_user_card(self, user):
        card = QFrame()
        card.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        # Important: The card click needs to trigger selection
        # We wrap the whole frame in a transparent button overlay or just use mousePressEvent
        
        card.setStyleSheet("""
            QFrame {
                background-color: #262639;
                border: 1px solid #313244;
                border-radius: 12px;
            }
            QFrame:hover {
                border: 1px solid #89b4fa;
                background-color: #313244;
            }
        """)
        card.setFixedHeight(100)
        
        layout = QHBoxLayout(card)
        layout.setContentsMargins(20, 15, 20, 15)
        
        # Avatar / Initials
        initial = user['username'][0].upper() if user['username'] else "?"
        avatar = QLabel(initial)
        avatar.setAlignment(Qt.AlignmentFlag.AlignCenter)
        avatar.setFixedSize(50, 50)
        avatar.setStyleSheet("""
            background-color: #89b4fa; color: #1e1e2e; 
            font-size: 24px; font-weight: bold; border-radius: 25px; border: none;
        """)
        layout.addWidget(avatar)
        
        # Info
        info_layout = QVBoxLayout()
        name = QLabel(user['username'])
        name.setStyleSheet("font-size: 18px; font-weight: bold; color: white; border: none; background: transparent;")
        
        # Badges
        tier = user.get('current_tier', 0) + 1
        details = QLabel(f"TIER {tier} • {user['experience_level'].upper()}")
        details.setStyleSheet("color: #a6adc8; font-size: 12px; font-weight: bold; border: none; background: transparent;")
        
        info_layout.addWidget(name)
        info_layout.addWidget(details)
        layout.addLayout(info_layout)
        
        layout.addStretch()
        
        # Select Button
        sel_btn = QPushButton("Select")
        sel_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        sel_btn.clicked.connect(lambda: self.user_selected.emit(user['id']))
        sel_btn.setStyleSheet("""
            QPushButton {
                background-color: #89b4fa; color: #1e1e2e;
                font-weight: bold; padding: 8px 16px; border-radius: 6px; border: none;
            }
            QPushButton:hover { background-color: #b4befe; }
        """)
        layout.addWidget(sel_btn)
        
        # Delete Button
        del_btn = QPushButton("✕")
        del_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        del_btn.setFixedSize(30, 30)
        del_btn.clicked.connect(lambda: self.delete_user(user['id']))
        del_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent; color: #45475a;
                font-weight: bold; border-radius: 15px; font-size: 14px;
            }
            QPushButton:hover { background-color: #f38ba8; color: #1e1e2e; }
        """)
        layout.addWidget(del_btn)
        
        return card

    # --- LOGIC ---
    def refresh_users(self):
        # Clear list
        while self.users_layout.count():
            child = self.users_layout.takeAt(0)
            if child.widget(): child.widget().deleteLater()
            
        users = self.db.get_all_users()
        
        if not users:
            empty = QLabel("No active athletes.\nCreate a profile to begin.")
            empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
            empty.setStyleSheet("color: #45475a; font-size: 16px; font-style: italic; margin-top: 50px;")
            self.users_layout.addWidget(empty)
        else:
            for user in users:
                self.users_layout.addWidget(self.create_user_card(user))

    def create_user(self):
        name = self.username_input.text().strip()
        if not name:
            QMessageBox.warning(self, "Input Error", "Username required")
            return
            
        try:
            self.db.create_user(
                username=name,
                experience_level=self.experience_combo.currentText().lower(),
                weight_kg=self.weight_input.value(),
                height_cm=self.height_input.value(),
                body_fat_percentage=self.bodyfat_input.value()
            )
            # Reset and refresh
            self.username_input.clear()
            self.refresh_users()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to create user: {e}")

    def delete_user(self, user_id):
        confirm = QMessageBox.question(self, "Confirm Delete", 
            "Are you sure? This will wipe all training logs permanently.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        
        if confirm == QMessageBox.StandardButton.Yes:
            # Need DB method for this, assume it exists or wrap in try
            try:
                # Direct SQL if method missing in DB manager interface
                cursor = self.db.conn.cursor()
                cursor.execute("DELETE FROM users WHERE id=?", (user_id,))
                self.db.conn.commit()
                self.refresh_users()
            except Exception as e:
                print(e)

    # Helpers
    def create_label(self, text):
        lbl = QLabel(text)
        lbl.setStyleSheet("color: #a6adc8; font-weight: bold; font-size: 12px; border: none;")
        return lbl
        
    def create_spinbox(self, min_val, max_val, default, suffix):
        sb = QDoubleSpinBox()
        sb.setRange(min_val, max_val)
        sb.setValue(default)
        sb.setSuffix(suffix)
        return sb

    def import_users(self):
        pass # Placeholder for future implementation

    def export_users(self):
        pass # Placeholder