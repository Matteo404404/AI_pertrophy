"""
Scientific Hypertrophy Trainer - Assessment Interface v2.1 (FIXED)
- Fixed: Added missing 'clear_content' method
- Features: Dark Theme, Detailed Results Review
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame, 
    QProgressBar, QButtonGroup, QMessageBox, QTextEdit, QScrollArea,
    QGridLayout, QSizePolicy
)
from PyQt6.QtCore import Qt, pyqtSignal, QSize
from PyQt6.QtGui import QFont, QCursor

class AssessmentWidget(QWidget):
    assessment_completed = pyqtSignal(dict)
    
    def __init__(self, assessment_engine, user_manager):
        super().__init__()
        self.assessment_engine = assessment_engine
        self.user_manager = user_manager
        self.current_assessment = None
        
        self.init_ui()
        self.refresh_tiers()
    
    def init_ui(self):
        # Global Layout
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(30, 30, 30, 30)
        self.main_layout.setSpacing(20)
        
        # Header (Fixed at top)
        self.header_frame = QFrame()
        self.header_layout = QVBoxLayout(self.header_frame)
        self.lbl_title = QLabel("KNOWLEDGE ASSESSMENT")
        self.lbl_title.setStyleSheet("font-size: 24px; font-weight: 900; color: #89b4fa; letter-spacing: 2px;")
        self.lbl_subtitle = QLabel("Validate your understanding of hypertrophy principles to unlock advanced tracking features.")
        self.lbl_subtitle.setStyleSheet("color: #a6adc8; font-size: 14px;")
        self.header_layout.addWidget(self.lbl_title)
        self.header_layout.addWidget(self.lbl_subtitle)
        self.main_layout.addWidget(self.header_frame)
        
        # Dynamic Content Area
        self.content_area = QFrame()
        self.content_layout = QVBoxLayout(self.content_area)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.addWidget(self.content_area)
        
        # Initialize with Tier Selection
        self.show_tier_selection()

    # ==========================================
    # VIEW 1: TIER SELECTION
    # ==========================================
    def show_tier_selection(self):
        self.clear_content()
        self.lbl_title.setText("SELECT CLEARANCE LEVEL")
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("border: none; background: transparent;")
        
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(20)
        
        # Get User Status
        tier_status = self.user_manager.get_tier_status()
        tiers = tier_status.get('tiers', [])
        
        descriptions = [
            "Fundamentals: Volume, Frequency, and Progressive Overload basics.",
            "Intermediate: Effective Reps, Length-Tension Relationships, and Fatigue Management.",
            "Advanced: Mesocycles, MRV/MEV, and Resensitization Phases."
        ]
        
        for i, tier in enumerate(tiers):
            desc = descriptions[i] if i < len(descriptions) else ""
            card = self.create_tier_card(i, tier, desc)
            layout.addWidget(card)
            
        layout.addStretch()
        scroll.setWidget(container)
        self.content_layout.addWidget(scroll)

    def create_tier_card(self, level, data, desc):
        card = QFrame()
        
        if data['completed']:
            border_col = "#a6e3a1"
            status_txt = "✅ COMPLETED"
            bg_col = "rgba(166, 227, 161, 0.05)"
            btn_txt = "Review Concepts"
            btn_style = "background-color: #313244; color: #a6e3a1; border: 1px solid #a6e3a1;"
        elif data['accessible']:
            border_col = "#89b4fa"
            status_txt = "🔓 AVAILABLE"
            bg_col = "rgba(137, 180, 250, 0.1)"
            btn_txt = "Start Assessment"
            btn_style = "background-color: #89b4fa; color: #1e1e2e; font-weight: bold;"
        else:
            border_col = "#45475a"
            status_txt = "🔒 LOCKED"
            bg_col = "rgba(69, 71, 90, 0.3)"
            btn_txt = "Locked"
            btn_style = "background-color: transparent; color: #45475a; border: 1px solid #45475a;"

        card.setStyleSheet(f"""
            QFrame {{
                background-color: {bg_col};
                border: 1px solid {border_col};
                border-radius: 12px;
            }}
        """)
        
        layout = QHBoxLayout(card)
        layout.setContentsMargins(20, 20, 20, 20)
        
        info_layout = QVBoxLayout()
        title = QLabel(f"TIER {level + 1}: {data['name'].upper()}")
        title.setStyleSheet(f"font-size: 18px; font-weight: bold; color: {border_col};")
        description = QLabel(desc)
        description.setWordWrap(True)
        description.setStyleSheet("color: #cdd6f4; margin-top: 5px;")
        
        info_layout.addWidget(title)
        info_layout.addWidget(description)
        layout.addLayout(info_layout, stretch=1)
        
        action_layout = QVBoxLayout()
        status_lbl = QLabel(status_txt)
        status_lbl.setAlignment(Qt.AlignmentFlag.AlignRight)
        status_lbl.setStyleSheet(f"font-weight: bold; font-size: 12px; color: {border_col}; margin-bottom: 10px;")
        
        btn = QPushButton(btn_txt)
        btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        btn.setStyleSheet(f"QPushButton {{ {btn_style} padding: 8px 20px; border-radius: 6px; }}")
        
        if data['accessible']:
            btn.clicked.connect(lambda checked, l=level: self.start_assessment(l))
        else:
            btn.setEnabled(False)
            
        action_layout.addWidget(status_lbl)
        action_layout.addWidget(btn)
        layout.addLayout(action_layout)
        
        return card

    # ==========================================
    # VIEW 2: ACTIVE QUESTION
    # ==========================================
    def start_assessment(self, tier_level):
        try:
            info = self.assessment_engine.start_assessment(tier_level)
            self.current_assessment = info
            self.load_question_view()
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def load_question_view(self):
        self.clear_content()
        question = self.assessment_engine.get_current_question()
        
        if not question:
            self.finish_assessment()
            return

        self.lbl_title.setText(f"TIER {self.current_assessment['tier_level'] + 1} EXAM")
        
        progress = QProgressBar()
        progress.setRange(0, question['total_questions'])
        progress.setValue(question['question_number'] - 1)
        progress.setStyleSheet("QProgressBar { background-color: #313244; border-radius: 4px; height: 8px; } QProgressBar::chunk { background-color: #89b4fa; border-radius: 4px; }")
        self.content_layout.addWidget(progress)
        
        progress_lbl = QLabel(f"Question {question['question_number']} / {question['total_questions']}")
        progress_lbl.setAlignment(Qt.AlignmentFlag.AlignRight)
        progress_lbl.setStyleSheet("color: #a6adc8; font-weight: bold; margin-bottom: 20px;")
        self.content_layout.addWidget(progress_lbl)
        
        q_card = QFrame()
        q_card.setStyleSheet("background-color: #262639; border-radius: 16px; padding: 20px;")
        q_layout = QVBoxLayout(q_card)
        
        q_text = QLabel(question['question'])
        q_text.setWordWrap(True)
        q_text.setStyleSheet("font-size: 22px; font-weight: bold; color: white; line-height: 1.4;")
        q_layout.addWidget(q_text)
        self.content_layout.addWidget(q_card)
        self.content_layout.addSpacing(20)
        
        self.btn_group = QButtonGroup()
        self.btn_group.setExclusive(True)
        
        for i, option in enumerate(question['options']):
            btn = QPushButton(option['text'])
            btn.setCheckable(True)
            btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
            btn.setStyleSheet("""
                QPushButton { background-color: #1e1e2e; border: 2px solid #313244; border-radius: 12px; padding: 20px; text-align: left; font-size: 16px; color: #cdd6f4; }
                QPushButton:hover { border-color: #89b4fa; background-color: #262639; }
                QPushButton:checked { background-color: #89b4fa; color: #1e1e2e; border-color: #89b4fa; font-weight: bold; }
            """)
            self.btn_group.addButton(btn)
            self.content_layout.addWidget(btn)
            
        self.content_layout.addStretch()
        
        submit_btn = QPushButton("Submit Answer")
        submit_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        submit_btn.clicked.connect(self.submit_answer)
        submit_btn.setStyleSheet("background-color: #a6e3a1; color: #1e1e2e; font-size: 16px; font-weight: bold; padding: 15px; border-radius: 8px;")
        self.content_layout.addWidget(submit_btn)

    def submit_answer(self):
        btn = self.btn_group.checkedButton()
        if not btn: return
        
        result = self.assessment_engine.submit_answer(btn.text())
        if result['status'] == 'complete':
            self.show_results(result['results'])
        else:
            self.load_question_view()

    # ==========================================
    # VIEW 3: RESULTS (With Detailed Review)
    # ==========================================
    def show_results(self, results):
        self.clear_content()
        self.lbl_title.setText("ASSESSMENT DEBRIEF")
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("border: none; background: transparent;")
        
        content_container = QWidget()
        layout = QVBoxLayout(content_container)
        layout.setSpacing(20)
        
        score_frame = QFrame()
        score_frame.setStyleSheet("background-color: #262639; border-radius: 16px; border: 1px solid #313244;")
        score_layout = QVBoxLayout(score_frame)
        
        score_lbl = QLabel(f"{results['score']} / {results['total_questions']}")
        score_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        score_lbl.setStyleSheet("font-size: 48px; font-weight: 900; color: #fab387;")
        score_layout.addWidget(score_lbl)
        
        pass_fail = "PASSED" if results['passed'] else "FAILED"
        col = "#a6e3a1" if results['passed'] else "#f38ba8"
        status_lbl = QLabel(pass_fail)
        status_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        status_lbl.setStyleSheet(f"font-size: 24px; font-weight: bold; color: {col}; letter-spacing: 4px;")
        score_layout.addWidget(status_lbl)
        
        layout.addWidget(score_frame)
        
        # Breakdown
        layout.addWidget(QLabel("QUESTION BREAKDOWN:"))
        
        for ans in results['answers']:
            row = QFrame()
            border_col = "#a6e3a1" if ans['is_correct'] else "#f38ba8"
            row.setStyleSheet(f"QFrame {{ background-color: #1e1e2e; border-left: 4px solid {border_col}; border-radius: 4px; padding: 10px; }}")
            r_layout = QVBoxLayout(row)
            
            q_lbl = QLabel(ans.get('question_text', 'Question'))
            q_lbl.setWordWrap(True)
            q_lbl.setStyleSheet("font-weight: bold; color: white; font-size: 14px;")
            r_layout.addWidget(q_lbl)
            
            if ans['is_correct']:
                res_lbl = QLabel(f"✅ Correct: {ans['selected_answer']}")
                res_lbl.setStyleSheet("color: #a6e3a1;")
                r_layout.addWidget(res_lbl)
            else:
                user_lbl = QLabel(f"❌ You said: {ans['selected_answer']}")
                user_lbl.setStyleSheet("color: #f38ba8;")
                corr_lbl = QLabel(f"✅ Correct: {ans['correct_answer']}")
                corr_lbl.setStyleSheet("color: #a6e3a1;")
                
                exp_lbl = QLabel(f"💡 {ans.get('explanation', '')}")
                exp_lbl.setWordWrap(True)
                exp_lbl.setStyleSheet("color: #a6adc8; font-style: italic; margin-top: 5px;")
                
                r_layout.addWidget(user_lbl)
                r_layout.addWidget(corr_lbl)
                r_layout.addWidget(exp_lbl)
                
            layout.addWidget(row)

        btn_layout = QHBoxLayout()
        retry_btn = QPushButton("Retry / Back")
        retry_btn.clicked.connect(self.show_tier_selection)
        retry_btn.setStyleSheet("background-color: #313244; color: white; padding: 15px; border-radius: 8px;")
        
        btn_layout.addWidget(retry_btn)
        layout.addLayout(btn_layout)
        layout.addStretch()
        
        scroll.setWidget(content_container)
        self.content_layout.addWidget(scroll)
        
        self.assessment_completed.emit(results)

    # --- THE MISSING METHOD ---
    def clear_content(self):
        while self.content_layout.count():
            child = self.content_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
    
    def refresh_tiers(self):
        self.show_tier_selection()
    
    def finish_assessment(self):
        results = self.assessment_engine.finish_assessment()
        self.show_results(results)