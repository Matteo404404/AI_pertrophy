"""
Scientific Hypertrophy Trainer - User Selection Screen
Fixed version with proper label visibility
"""

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QFrame, QLineEdit, QComboBox, 
                             QDoubleSpinBox, QMessageBox, QScrollArea, QFileDialog)
from PyQt6.QtCore import Qt, pyqtSignal
import json


class UserSelectionWidget(QWidget):
    """Beautiful user selection screen with fixed styling"""
    
    user_selected = pyqtSignal(int)
    
    def __init__(self, db_manager, user_manager):
        super().__init__()
        self.db = db_manager
        self.user_manager = user_manager
        self.init_ui()
        self.refresh_users()
    
    def init_ui(self):
        """Initialize user interface"""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(40, 40, 40, 40)
        main_layout.setSpacing(30)
        
        # Header
        header = QLabel("Scientific Hypertrophy Trainer")
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header.setStyleSheet("font-size: 32px; font-weight: bold; color: #667eea; margin-bottom: 5px;")
        main_layout.addWidget(header)
        
        subtitle = QLabel("Evidence-based muscle building through progressive knowledge assessment")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet("font-size: 15px; color: #6c757d; margin-bottom: 30px;")
        main_layout.addWidget(subtitle)
        
        # Content layout
        content_layout = QHBoxLayout()
        content_layout.setSpacing(30)
        
        self.create_user_list_section(content_layout)
        self.create_new_user_section(content_layout)
        
        main_layout.addLayout(content_layout)
        
        # Bottom buttons
        bottom_layout = QHBoxLayout()
        
        import_btn = QPushButton("📥 Import Users")
        import_btn.clicked.connect(self.import_users)
        import_btn.setStyleSheet("""
            background-color: #6c757d;
            padding: 10px 20px;
            border-radius: 6px;
            font-weight: 600;
        """)
        
        export_btn = QPushButton("💾 Export Users")
        export_btn.clicked.connect(self.export_users)
        export_btn.setStyleSheet("""
            background-color: #6c757d;
            padding: 10px 20px;
            border-radius: 6px;
            font-weight: 600;
        """)
        
        bottom_layout.addWidget(import_btn)
        bottom_layout.addWidget(export_btn)
        bottom_layout.addStretch()
        
        main_layout.addLayout(bottom_layout)
    
    def create_user_list_section(self, parent_layout):
        """Create user list section"""
        list_frame = QFrame()
        
        list_layout = QVBoxLayout(list_frame)
        list_layout.setContentsMargins(20, 20, 20, 20)
        
        title = QLabel("Select User")
        title.setStyleSheet("font-size: 22px; font-weight: bold; margin-bottom: 20px;")
        list_layout.addWidget(title)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("border: none; background-color: transparent;")
        
        scroll_widget = QWidget()
        self.users_layout = QVBoxLayout(scroll_widget)
        self.users_layout.setSpacing(15)
        
        scroll.setWidget(scroll_widget)
        list_layout.addWidget(scroll)
        
        parent_layout.addWidget(list_frame, stretch=2)
    
    def create_new_user_section(self, parent_layout):
        """Create new user section with proper label visibility"""
        form_frame = QFrame()
        form_frame.setFixedWidth(420)
        
        form_layout = QVBoxLayout(form_frame)
        form_layout.setContentsMargins(30, 30, 30, 30)
        form_layout.setSpacing(12)
        
        # Title
        title = QLabel("Create New User")
        title.setStyleSheet("font-size: 24px; font-weight: bold; margin-bottom: 5px;")
        form_layout.addWidget(title)
        
        # Subtitle
        subtitle = QLabel("Set up your training profile")
        subtitle.setStyleSheet("font-size: 13px; color: #6c757d; margin-bottom: 20px;")
        form_layout.addWidget(subtitle)
        
        # Helper function to add labeled fields
        def add_field(label_text, widget):
            lbl = QLabel(label_text)
            lbl.setStyleSheet("font-size: 14px; font-weight: 600; margin-top: 10px; margin-bottom: 5px;")
            form_layout.addWidget(lbl)
            form_layout.addWidget(widget)
        
        # USERNAME
        self.username_input = QLineEdit()
        self.username_input.setPlaceholderText("Enter your username")
        add_field("Username", self.username_input)
        
        # TRAINING EXPERIENCE
        self.experience_combo = QComboBox()
        self.experience_combo.addItems(['Beginner', 'Intermediate', 'Advanced'])
        add_field("Training Experience", self.experience_combo)
        
        # WEIGHT
        self.weight_input = QDoubleSpinBox()
        self.weight_input.setRange(30, 300)
        self.weight_input.setValue(75.0)
        self.weight_input.setDecimals(1)
        self.weight_input.setSuffix(" kg")
        add_field("Weight (kg)", self.weight_input)
        
        # HEIGHT
        self.height_input = QDoubleSpinBox()
        self.height_input.setRange(120, 250)
        self.height_input.setValue(175.0)
        self.height_input.setDecimals(0)
        self.height_input.setSuffix(" cm")
        add_field("Height (cm)", self.height_input)
        
        # BODY FAT
        self.bodyfat_input = QDoubleSpinBox()
        self.bodyfat_input.setRange(3, 50)
        self.bodyfat_input.setValue(15.0)
        self.bodyfat_input.setDecimals(1)
        self.bodyfat_input.setSuffix(" %")
        add_field("Body Fat (%)", self.bodyfat_input)
        
        # Spacer
        form_layout.addSpacing(15)
        
        # CREATE BUTTON
        create_btn = QPushButton("✨ Create Profile")
        create_btn.clicked.connect(self.create_user)
        create_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        create_btn.setStyleSheet("""
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #2ecc71, stop:1 #27ae60);
            padding: 16px;
            font-size: 16px;
            font-weight: bold;
        """)
        form_layout.addWidget(create_btn)
        
        parent_layout.addWidget(form_frame)
    
    def refresh_users(self):
        """Refresh user list"""
        # Clear existing users
        while self.users_layout.count():
            child = self.users_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        
        users = self.db.get_all_users()
        
        if not users:
            no_users = QLabel("No users yet. Create your first profile!")
            no_users.setAlignment(Qt.AlignmentFlag.AlignCenter)
            no_users.setStyleSheet("font-size: 14px; color: #6c757d; padding: 40px;")
            self.users_layout.addWidget(no_users)
        else:
            for user in users:
                self.users_layout.addWidget(self.create_user_card(user))
        
        self.users_layout.addStretch()
    
    def create_user_card(self, user):
        """Create user card"""
        card = QFrame()
        card.setStyleSheet("""
            QFrame {
                background-color: #f8f9fa;
                border: 2px solid #e9ecef;
                border-radius: 10px;
                padding: 16px;
            }
        """)
        
        card_layout = QHBoxLayout(card)
        info_layout = QVBoxLayout()
        
        # Username
        username = QLabel(user['username'])
        username.setStyleSheet("font-size: 16px; font-weight: bold;")
        info_layout.addWidget(username)
        
        # Badges
        stats_layout = QHBoxLayout()
        
        tier = user.get('current_tier', 0)
        tier_colors = ['#2ecc71', '#f39c12', '#e74c3c']
        tier_badge = QLabel(f"Tier {tier + 1}")
        tier_badge.setStyleSheet(f"""
            background-color: {tier_colors[tier]};
            color: #ffffff;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 600;
        """)
        stats_layout.addWidget(tier_badge)
        
        exp = user.get('experience_level', 'beginner')
        exp_badge = QLabel(exp.capitalize())
        exp_badge.setStyleSheet("""
            background-color: #667eea;
            color: #ffffff;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 600;
        """)
        stats_layout.addWidget(exp_badge)
        stats_layout.addStretch()
        info_layout.addLayout(stats_layout)
        
        # Stats
        stats_text = f"Weight: {user.get('weight_kg', 0):.1f}kg • Height: {user.get('height_cm', 0):.0f}cm • BF: {user.get('body_fat_percentage', 0):.1f}%"
        stats_label = QLabel(stats_text)
        stats_label.setStyleSheet("font-size: 12px; color: #6c757d; margin-top: 8px;")
        info_layout.addWidget(stats_label)
        
        card_layout.addLayout(info_layout)
        card_layout.addStretch()
        
        # Buttons
        buttons_layout = QVBoxLayout()
        
        select_btn = QPushButton("Select")
        select_btn.clicked.connect(lambda: self.user_selected.emit(user['id']))
        select_btn.setStyleSheet("""
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #667eea, stop:1 #764ba2);
            padding: 8px 20px;
            border-radius: 6px;
            font-weight: 600;
        """)
        buttons_layout.addWidget(select_btn)
        
        delete_btn = QPushButton("🗑️")
        delete_btn.clicked.connect(lambda: self.delete_user(user['id']))
        delete_btn.setFixedSize(40, 40)
        delete_btn.setStyleSheet("""
            background-color: #e74c3c;
            border-radius: 20px;
            font-size: 16px;
        """)
        buttons_layout.addWidget(delete_btn)
        
        card_layout.addLayout(buttons_layout)
        return card
    
    def create_user(self):
        """Create new user"""
        username = self.username_input.text().strip()
        
        if not username:
            QMessageBox.warning(self, "Error", "Please enter a username")
            return
        
        try:
            user_id = self.db.create_user(
                username=username,
                experience_level=self.experience_combo.currentText().lower(),
                weight_kg=self.weight_input.value(),
                height_cm=self.height_input.value(),
                body_fat_percentage=self.bodyfat_input.value()
            )
            
            QMessageBox.information(self, "Success", f"User '{username}' created successfully!")
            
            # Reset form
            self.username_input.clear()
            self.experience_combo.setCurrentIndex(0)
            self.weight_input.setValue(75.0)
            self.height_input.setValue(175.0)
            self.bodyfat_input.setValue(15.0)
            
            self.refresh_users()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to create user: {str(e)}")
    
    def delete_user(self, user_id):
        """Delete user"""
        reply = QMessageBox.question(
            self, 'Delete User',
            'Are you sure you want to delete this user?\nAll data will be lost.',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            try:
                self.db.delete_user(user_id)
                self.refresh_users()
                QMessageBox.information(self, "Success", "User deleted successfully!")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to delete user: {str(e)}")
    
    def import_users(self):
        """Import users from JSON"""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Import Users", "", "JSON files (*.json)"
        )
        
        if filename:
            try:
                with open(filename, 'r') as f:
                    users_data = json.load(f)
                
                QMessageBox.information(self, "Success", f"Imported {len(users_data)} users!")
                self.refresh_users()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to import users: {str(e)}")
    
    def export_users(self):
        """Export users to JSON"""
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Users", "users_export.json", "JSON files (*.json)"
        )
        
        if filename:
            try:
                users = self.db.get_all_users()
                with open(filename, 'w') as f:
                    json.dump(users, f, indent=2)
                
                QMessageBox.information(self, "Success", f"Exported {len(users)} users!")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export users: {str(e)}")