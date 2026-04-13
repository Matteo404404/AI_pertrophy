"""
Scientific Hypertrophy Trainer - User Selection v3.0 (God-Tier UI)
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame, 
    QLineEdit, QComboBox, QDoubleSpinBox, QMessageBox, QScrollArea, 
    QGridLayout
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QCursor

class UserSelectionWidget(QWidget):
    user_selected = pyqtSignal(int)
    
    def __init__(self, db_manager, user_manager):
        super().__init__()
        self.db = db_manager
        self.user_manager = user_manager
        self.init_ui()
        self.refresh_users()
    
    def init_ui(self):
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)
        
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(40)
        
        self.create_left_panel(layout)
        self.create_right_panel(layout)
        self.main_layout.addWidget(container)

    def create_left_panel(self, parent_layout):
        left_frame = QFrame()
        left_frame.setStyleSheet("background: transparent; border: none;")
        left_layout = QVBoxLayout(left_frame)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        header = QLabel("SELECT ATHLETE")
        header.setStyleSheet("font-size: 36px; font-weight: 900; color: #89b4fa; letter-spacing: 2px;")
        left_layout.addWidget(header)
        
        sub = QLabel("Load your biometrics, AI models, and training history.")
        sub.setStyleSheet("font-size: 15px; color: #a6adc8; margin-bottom: 20px;")
        left_layout.addWidget(sub)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("background: transparent; border: none;")
        
        self.users_container = QWidget()
        self.users_layout = QVBoxLayout(self.users_container)
        self.users_layout.setSpacing(20)
        self.users_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        scroll.setWidget(self.users_container)
        left_layout.addWidget(scroll)
        parent_layout.addWidget(left_frame, 5) 

    def create_right_panel(self, parent_layout):
        right_frame = QFrame()
        # The global theme handles the base QFrame, but we force specific padding here
        layout = QVBoxLayout(right_frame)
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(25)
        
        lbl = QLabel("NEW ATHLETE PROFILE")
        lbl.setStyleSheet("font-size: 20px; font-weight: 900; color: #a6e3a1; border: none; letter-spacing: 1px;")
        layout.addWidget(lbl)
        
        self.username_input = QLineEdit()
        self.username_input.setPlaceholderText("Enter Username...")
        layout.addWidget(self.create_label("IDENTIFIER"))
        layout.addWidget(self.username_input)
        
        self.experience_combo = QComboBox()
        self.experience_combo.addItems(['Beginner', 'Intermediate', 'Advanced'])
        layout.addWidget(self.create_label("TRAINING EXPERIENCE"))
        layout.addWidget(self.experience_combo)
        
        stats_grid = QGridLayout()
        stats_grid.setSpacing(15)
        
        self.weight_input = self.create_spinbox(30, 300, 75.0, " kg")
        stats_grid.addWidget(self.create_label("BODY WEIGHT"), 0, 0)
        stats_grid.addWidget(self.weight_input, 1, 0)
        
        self.height_input = self.create_spinbox(120, 250, 175, " cm")
        stats_grid.addWidget(self.create_label("HEIGHT"), 0, 1)
        stats_grid.addWidget(self.height_input, 1, 1)
        
        self.bodyfat_input = self.create_spinbox(3, 60, 15.0, " %")
        stats_grid.addWidget(self.create_label("EST. BODY FAT"), 2, 0)
        stats_grid.addWidget(self.bodyfat_input, 3, 0)
        
        layout.addLayout(stats_grid)
        layout.addStretch()
        
        create_btn = QPushButton("INITIALIZE AI PROFILE")
        create_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        create_btn.clicked.connect(self.create_user)
        create_btn.setStyleSheet("background-color: #a6e3a1; color: #11111b; font-weight: 900; font-size: 14px; padding: 18px;")
        layout.addWidget(create_btn)
        
        parent_layout.addWidget(right_frame, 4)

    def create_user_card(self, user):
        card = QFrame()
        card.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        # Card style overrides global slightly for hover effects
        card.setStyleSheet("""
            QFrame { background: #181825; border: 1px solid #2a2b3c; border-radius: 16px; }
            QFrame:hover { border: 1px solid #89b4fa; background: #1e1e2e; }
        """)
        card.setFixedHeight(110)
        
        layout = QHBoxLayout(card)
        layout.setContentsMargins(25, 20, 25, 20)
        layout.setSpacing(20)
        
        initial = user['username'][0].upper() if user['username'] else "?"
        avatar = QLabel(initial)
        avatar.setAlignment(Qt.AlignmentFlag.AlignCenter)
        avatar.setFixedSize(60, 60)
        avatar.setStyleSheet("""
            background-color: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #89b4fa, stop:1 #cba6f7);
            color: #11111b; font-size: 28px; font-weight: 900; border-radius: 30px; border: none;
        """)
        layout.addWidget(avatar)
        
        info_layout = QVBoxLayout()
        info_layout.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        name = QLabel(user['username'])
        name.setStyleSheet("font-size: 20px; font-weight: 900; color: white; border: none; background: transparent;")
        
        tier = user.get('current_tier', 0) + 1
        details = QLabel(f"TIER {tier} CLEARANCE  •  {user['experience_level'].upper()}")
        details.setStyleSheet("color: #a6adc8; font-size: 12px; font-weight: bold; border: none; background: transparent; letter-spacing: 1px;")
        
        info_layout.addWidget(name)
        info_layout.addWidget(details)
        layout.addLayout(info_layout)
        layout.addStretch()
        
        sel_btn = QPushButton("Load Profile")
        sel_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        sel_btn.clicked.connect(lambda: self.user_selected.emit(user['id']))
        sel_btn.setStyleSheet("background-color: #89b4fa; color: #11111b; font-weight: bold; padding: 12px 24px;")
        layout.addWidget(sel_btn)
        
        del_btn = QPushButton("✕")
        del_btn.setFixedSize(40, 40)
        del_btn.clicked.connect(lambda: self.delete_user(user['id']))
        del_btn.setStyleSheet("background: transparent; color: #6c7086; font-weight: bold; font-size: 16px;")
        layout.addWidget(del_btn)
        
        return card

    def refresh_users(self):
        while self.users_layout.count():
            child = self.users_layout.takeAt(0)
            if child.widget(): child.widget().deleteLater()
            
        users = self.db.get_all_users()
        if not users:
            empty = QLabel("No active athletes.\nCreate a profile to begin.")
            empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
            empty.setStyleSheet("color: #45475a; font-size: 18px; font-style: italic; margin-top: 50px; border: none;")
            self.users_layout.addWidget(empty)
        else:
            for user in users:
                self.users_layout.addWidget(self.create_user_card(user))

    def create_user(self):
        name = self.username_input.text().strip()
        if not name: return
        try:
            self.db.create_user(
                username=name, experience_level=self.experience_combo.currentText().lower(),
                weight_kg=self.weight_input.value(), height_cm=self.height_input.value(),
                body_fat_percentage=self.bodyfat_input.value()
            )
            self.username_input.clear()
            self.refresh_users()
        except Exception as e: QMessageBox.critical(self, "Error", f"Failed: {e}")

    def delete_user(self, user_id):
        confirm = QMessageBox.question(self, "Wipe Data", "Are you sure? This deletes all AI history.", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if confirm == QMessageBox.StandardButton.Yes:
            try:
                cursor = self.db.conn.cursor()
                cursor.execute("DELETE FROM users WHERE id=?", (user_id,))
                self.db.conn.commit()
                self.refresh_users()
            except Exception as e: print(e)

    def create_label(self, text):
        lbl = QLabel(text)
        lbl.setStyleSheet("color: #6c7086; font-weight: 900; font-size: 11px; letter-spacing: 1px; border: none;")
        return lbl
        
    def create_spinbox(self, min_val, max_val, default, suffix):
        sb = QDoubleSpinBox()
        sb.setRange(min_val, max_val)
        sb.setValue(default)
        sb.setSuffix(suffix)
        return sb
