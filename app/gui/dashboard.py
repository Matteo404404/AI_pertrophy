"""
Scientific Hypertrophy Trainer - Dashboard Screen
Main user dashboard with stats, quick actions, and progress overview
"""

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QFrame, QGridLayout, QProgressBar,
                             QScrollArea, QGroupBox)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QPixmap, QPainter, QColor
from datetime import datetime, timedelta


class DashboardWidget(QWidget):
    """Main dashboard showing user stats and quick actions"""
    
    start_assessment = pyqtSignal()
    open_tracking = pyqtSignal()
    
    def __init__(self, user_manager, tracking_system):
        super().__init__()
        self.user_manager = user_manager
        self.tracking_system = tracking_system
        self.init_ui()
    
    def init_ui(self):
        """Initialize dashboard interface"""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(40, 40, 40, 40)
        
        # Scrollable content
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("QScrollArea { border: none; }")
        
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        
        # Header
        self.create_header(content_layout)
        
        # Stats overview
        self.create_stats_overview(content_layout)
        
        # Quick actions
        self.create_quick_actions(content_layout)
        
        # Progress overview
        self.create_progress_overview(content_layout)
        
        # Recent activity
        self.create_recent_activity(content_layout)
        
        content_layout.addStretch()
        
        scroll_area.setWidget(content_widget)
        main_layout.addWidget(scroll_area)
    
    def create_header(self, parent_layout):
        """Create dashboard header"""
        header_frame = QFrame()
        header_layout = QVBoxLayout(header_frame)
        
        # Welcome message
        self.welcome_label = QLabel("Welcome back!")
        self.welcome_label.setStyleSheet("font-size: 32px; font-weight: bold; color: #667eea; margin-bottom: 10px;")
        
        self.status_label = QLabel("Ready to optimize your hypertrophy journey")
        self.status_label.setStyleSheet("font-size: 16px; color: #6c757d; margin-bottom: 30px;")
        
        header_layout.addWidget(self.welcome_label)
        header_layout.addWidget(self.status_label)
        
        parent_layout.addWidget(header_frame)
    
    def create_stats_overview(self, parent_layout):
        """Create statistics overview cards"""
        stats_frame = QFrame()
        stats_layout = QVBoxLayout(stats_frame)
        
        # Section title
        title = QLabel("Progress Overview")
        title.setStyleSheet("font-size: 24px; font-weight: bold; margin-bottom: 20px;")
        stats_layout.addWidget(title)
        
        # Stats grid
        stats_grid = QGridLayout()
        
        # Create stat cards
        self.stat_cards = {}
        stat_names = [
            ("current_tier", "Current Tier", "#667eea"),
            ("questions_answered", "Questions Answered", "#28a745"),
            ("accuracy", "Accuracy Rate", "#ffc107"),
            ("assessments_passed", "Assessments Passed", "#dc3545")
        ]
        
        for i, (key, label, color) in enumerate(stat_names):
            card = self.create_stat_card("0", label, color)
            self.stat_cards[key] = card
            stats_grid.addWidget(card, i // 2, i % 2)
        
        stats_layout.addLayout(stats_grid)
        parent_layout.addWidget(stats_frame)
    
    def create_stat_card(self, value, label, color):
        """Create individual stat card"""
        card_frame = QFrame()
        card_frame.setStyleSheet(f"""
            QFrame {{
                background-color: #ffffff;
                border: 2px solid {color};
                border-radius: 12px;
                padding: 20px;
                margin: 10px;
                min-height: 100px;
            }}
        """)
        
        card_layout = QVBoxLayout(card_frame)
        card_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        value_label = QLabel(value)
        value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        value_label.setStyleSheet(f"font-size: 36px; font-weight: bold; color: {color}; margin-bottom: 10px;")
        
        desc_label = QLabel(label)
        desc_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        desc_label.setStyleSheet("font-size: 14px; color: #6c757d;")
        
        card_layout.addWidget(value_label)
        card_layout.addWidget(desc_label)
        
        # Store references for updates
        card_frame.value_label = value_label
        card_frame.desc_label = desc_label
        
        return card_frame
    
    def create_quick_actions(self, parent_layout):
        """Create quick action buttons"""
        actions_frame = QFrame()
        actions_layout = QVBoxLayout(actions_frame)
        
        # Section title
        title = QLabel("Quick Actions")
        title.setStyleSheet("font-size: 24px; font-weight: bold; margin-bottom: 20px;")
        actions_layout.addWidget(title)
        
        # Action buttons grid
        actions_grid = QGridLayout()
        
        # Assessment button
        assess_btn = self.create_action_button(
            "📋", "Take Assessment", 
            "Test your hypertrophy knowledge and unlock new tiers",
            "#667eea"
        )
        assess_btn.clicked.connect(self.start_assessment.emit)
        actions_grid.addWidget(assess_btn, 0, 0)
        
        # Track diet button
        diet_btn = self.create_action_button(
            "🍽️", "Log Diet", 
            "Track your nutrition for optimal muscle growth",
            "#28a745"
        )
        diet_btn.clicked.connect(lambda: self.open_tracking.emit())
        actions_grid.addWidget(diet_btn, 0, 1)
        
        # Track sleep button
        sleep_btn = self.create_action_button(
            "😴", "Log Sleep", 
            "Monitor your recovery and sleep quality",
            "#6f42c1"
        )
        sleep_btn.clicked.connect(lambda: self.open_tracking.emit())
        actions_grid.addWidget(sleep_btn, 1, 0)
        
        # Log workout button
        workout_btn = self.create_action_button(
            "💪", "Log Workout", 
            "Record your training sessions and progress",
            "#fd7e14"
        )
        workout_btn.clicked.connect(lambda: self.open_tracking.emit())
        actions_grid.addWidget(workout_btn, 1, 1)
        
        actions_layout.addLayout(actions_grid)
        parent_layout.addWidget(actions_frame)
    
    def create_action_button(self, icon, title, description, color):
        """Create quick action button"""
        btn_frame = QFrame()
        btn_frame.setStyleSheet(f"""
            QFrame {{
                background-color: #ffffff;
                border: 2px solid {color};
                border-radius: 12px;
                padding: 20px;
                margin: 10px;
                min-height: 120px;
            }}
            QFrame:hover {{
                border-color: {color};
                background-color: #f8f9fa;
            }}
        """)
        
        btn_layout = QVBoxLayout(btn_frame)
        btn_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Icon
        icon_label = QLabel(icon)
        icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon_label.setStyleSheet(f"font-size: 32px; color: {color}; margin-bottom: 10px;")
        
        # Title
        title_label = QLabel(title)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; margin-bottom: 5px;")
        
        # Description
        desc_label = QLabel(description)
        desc_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("font-size: 12px; color: #6c757d;")
        
        btn_layout.addWidget(icon_label)
        btn_layout.addWidget(title_label)
        btn_layout.addWidget(desc_label)
        
        # Convert frame to button-like behavior
        button = QPushButton()
        button.setStyleSheet("QPushButton { background: transparent; border: none; }")
        button.setLayout(btn_layout)
        
        return button
    
    def create_progress_overview(self, parent_layout):
        """Create tier progress overview"""
        progress_frame = QFrame()
        progress_layout = QVBoxLayout(progress_frame)
        
        # Section title
        title = QLabel("Knowledge Progression")
        title.setStyleSheet("font-size: 24px; font-weight: bold; margin-bottom: 20px;")
        progress_layout.addWidget(title)
        
        # Tier progress bars
        self.tier_progress_bars = {}
        tier_names = ["Tier 1: Fundamentals", "Tier 2: Intermediate", "Tier 3: Advanced"]
        tier_colors = ["#28a745", "#ffc107", "#dc3545"]
        
        for i, (name, color) in enumerate(zip(tier_names, tier_colors)):
            tier_layout = QVBoxLayout()
            
            # Tier label
            tier_label = QLabel(name)
            tier_label.setStyleSheet(f"font-size: 16px; font-weight: bold; color: {color}; margin-bottom: 5px;")
            tier_layout.addWidget(tier_label)
            
            # Progress bar
            progress_bar = QProgressBar()
            progress_bar.setStyleSheet(f"""
                QProgressBar {{
                    background-color: #e9ecef;
                    border: none;
                    border-radius: 8px;
                    height: 20px;
                    text-align: center;
                    font-weight: bold;
                }}
                QProgressBar::chunk {{
                    background-color: {color};
                    border-radius: 8px;
                }}
            """)
            progress_bar.setRange(0, 100)
            progress_bar.setValue(0)
            tier_layout.addWidget(progress_bar)
            
            progress_layout.addLayout(tier_layout)
            self.tier_progress_bars[i] = progress_bar
        
        parent_layout.addWidget(progress_frame)
    
    def create_recent_activity(self, parent_layout):
        """Create recent activity section"""
        activity_frame = QFrame()
        activity_layout = QVBoxLayout(activity_frame)
        
        # Section title
        title = QLabel("Recent Activity")
        title.setStyleSheet("font-size: 24px; font-weight: bold; margin-bottom: 20px;")
        activity_layout.addWidget(title)
        
        # Activity list
        self.activity_list = QVBoxLayout()
        activity_layout.addLayout(self.activity_list)
        
        parent_layout.addWidget(activity_frame)
    
    def refresh_data(self):
        """Refresh dashboard data"""
        current_user = self.user_manager.get_current_user()
        if not current_user:
            return
        
        # Update welcome message
        self.welcome_label.setText(f"Welcome back, {current_user['username']}!")
        
        # Get user stats
        dashboard_data = self.user_manager.get_dashboard_data()
        stats = dashboard_data.get('stats', {})
        tier_status = dashboard_data.get('tier_status', {})
        
        # Update stat cards
        self.stat_cards['current_tier'].value_label.setText(str(stats.get('current_tier', 0) + 1))
        self.stat_cards['questions_answered'].value_label.setText(str(stats.get('total_questions_answered', 0)))
        self.stat_cards['accuracy'].value_label.setText(f"{stats.get('accuracy_percentage', 0):.1f}%")
        self.stat_cards['assessments_passed'].value_label.setText(str(stats.get('passed_assessments', 0)))
        
        # Update tier progress
        tiers = tier_status.get('tiers', [])
        for i, tier in enumerate(tiers):
            if i < len(self.tier_progress_bars):
                progress = 100 if tier.get('completed') else (50 if tier.get('accessible') else 0)
                self.tier_progress_bars[i].setValue(progress)
                
                status_text = "✅ Completed" if tier.get('completed') else ("🔓 Available" if tier.get('accessible') else "🔒 Locked")
                self.tier_progress_bars[i].setFormat(f"{tier.get('name', f'Tier {i+1}')} - {status_text}")
        
        # Update recent activity
        self.update_recent_activity(dashboard_data.get('recent_activity', {}))
    
    def update_recent_activity(self, activity_data):
        """Update recent activity list"""
        # Clear existing activity items
        while self.activity_list.count():
            child = self.activity_list.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        
        # Add activity items
        activities = []
        
        # Diet entries
        diet_count = activity_data.get('diet_entries', 0)
        if diet_count > 0:
            activities.append((f"📊 Logged {diet_count} diet entries this week", "#28a745"))
        
        # Sleep entries
        sleep_count = activity_data.get('sleep_entries', 0)
        if sleep_count > 0:
            activities.append((f"😴 Tracked {sleep_count} sleep sessions this week", "#6f42c1"))
        
        # Training entries
        training_count = activity_data.get('training_entries', 0)
        if training_count > 0:
            activities.append((f"💪 Completed {training_count} workouts this week", "#fd7e14"))
        
        if not activities:
            activities.append(("No recent activity - start tracking your progress!", "#6c757d"))
        
        for text, color in activities:
            activity_item = QLabel(text)
            activity_item.setStyleSheet(f"""
                color: {color};
                font-size: 14px;
                padding: 8px;
                background-color: #f8f9fa;
                border-radius: 6px;
                margin: 2px;
            """)
            self.activity_list.addWidget(activity_item)