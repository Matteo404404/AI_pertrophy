"""
Scientific Hypertrophy Trainer - Assessment Interface
- Modernized UI with smooth cards and clean spacing.
- Instant feedback (AI Coaching moved to Learning tab to prevent CPU lockups).
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame, 
    QProgressBar, QButtonGroup, QMessageBox, QScrollArea
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QCursor

class AssessmentWidget(QWidget):
    assessment_completed = pyqtSignal(dict)
    
    def __init__(self, assessment_engine, user_manager):
        super().__init__()
        self.assessment_engine = assessment_engine
        self.user_manager = user_manager
        self.current_assessment = None
        self.init_ui()
    
    def init_ui(self):
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(40, 40, 40, 40)
        self.main_layout.setSpacing(20)
        
        self.header_frame = QFrame()
        self.header_layout = QVBoxLayout(self.header_frame)
        self.lbl_title = QLabel("KNOWLEDGE ASSESSMENT")
        self.lbl_title.setStyleSheet("font-size: 28px; font-weight: 900; color: #89b4fa; letter-spacing: 1px;")
        self.lbl_subtitle = QLabel("Validate your understanding to unlock advanced ML tracking.")
        self.lbl_subtitle.setStyleSheet("color: #a6adc8; font-size: 15px;")
        self.header_layout.addWidget(self.lbl_title)
        self.header_layout.addWidget(self.lbl_subtitle)
        self.main_layout.addWidget(self.header_frame)
        
        self.content_area = QFrame()
        self.content_layout = QVBoxLayout(self.content_area)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.addWidget(self.content_area)
        
        self.refresh_tiers()

    def clear_content(self):
        while self.content_layout.count():
            child = self.content_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

    def refresh_tiers(self):
        self.show_tier_selection()

    def show_tier_selection(self):
        self.clear_content()
        self.lbl_title.setText("SELECT CLEARANCE LEVEL")
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("border: none; background: transparent;")
        
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(20)
        
        tier_status = self.user_manager.get_tier_status()
        tiers = tier_status.get('tiers',[])
        
        descriptions =[
            "Fundamentals: Volume, Frequency, and Progressive Overload basics.",
            "Intermediate: Effective Reps, Force Production, and Fatigue Management.",
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
            b_col, s_txt, bg_col, b_txt = "#a6e3a1", "✅ COMPLETED", "rgba(166, 227, 161, 0.05)", "Review Concepts"
        elif data['unlocked']:
            b_col, s_txt, bg_col, b_txt = "#89b4fa", "🔓 AVAILABLE", "rgba(137, 180, 250, 0.1)", "Start Assessment"
        else:
            b_col, s_txt, bg_col, b_txt = "#45475a", "🔒 LOCKED", "rgba(69, 71, 90, 0.2)", "Locked"

        card.setStyleSheet(f"background-color: {bg_col}; border: 1px solid {b_col}; border-radius: 12px;")
        layout = QHBoxLayout(card)
        layout.setContentsMargins(25, 25, 25, 25)
        
        i_layout = QVBoxLayout()
        title = QLabel(f"TIER {level + 1}: {data['name'].upper()}")
        title.setStyleSheet(f"font-size: 18px; font-weight: bold; color: {b_col}; border: none;")
        desc_lbl = QLabel(desc)
        desc_lbl.setStyleSheet("color: #cdd6f4; margin-top: 5px; border: none; background: transparent;")
        i_layout.addWidget(title)
        i_layout.addWidget(desc_lbl)
        layout.addLayout(i_layout, stretch=1)
        
        a_layout = QVBoxLayout()
        s_lbl = QLabel(s_txt)
        s_lbl.setAlignment(Qt.AlignmentFlag.AlignRight)
        s_lbl.setStyleSheet(f"font-weight: bold; font-size: 12px; color: {b_col}; border: none; background: transparent;")
        
        btn = QPushButton(b_txt)
        btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        if data['unlocked']:
            btn.setStyleSheet(f"background-color: {b_col}; color: #1e1e2e; font-weight: bold; padding: 10px 20px; border-radius: 6px;")
            btn.clicked.connect(lambda c, l=level: self.start_assessment(l))
        else:
            btn.setStyleSheet("background-color: transparent; color: #45475a; border: 1px solid #45475a; padding: 10px 20px; border-radius: 6px;")
            btn.setEnabled(False)
            
        a_layout.addWidget(s_lbl)
        a_layout.addWidget(btn)
        layout.addLayout(a_layout)
        return card

    def start_assessment(self, tier_level):
        try:
            self.current_assessment = self.assessment_engine.start_assessment(tier_level)
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
        progress.setStyleSheet("QProgressBar { background: #313244; border-radius: 4px; height: 6px; } QProgressBar::chunk { background: #89b4fa; border-radius: 4px; }")
        self.content_layout.addWidget(progress)
        
        q_card = QFrame()
        q_card.setStyleSheet("background: #181825; border-radius: 12px; border: 1px solid #313244; padding: 30px;")
        q_layout = QVBoxLayout(q_card)
        q_text = QLabel(question['question'])
        q_text.setWordWrap(True)
        q_text.setStyleSheet("font-size: 20px; font-weight: bold; color: white; line-height: 1.4; border: none;")
        q_layout.addWidget(q_text)
        self.content_layout.addWidget(q_card)
        
        self.btn_group = QButtonGroup()
        self.btn_group.setExclusive(True)
        for option in question['options']:
            btn = QPushButton(option['text'])
            btn.setCheckable(True)
            btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
            btn.setStyleSheet("""
                QPushButton { background: #262639; border: 1px solid #313244; border-radius: 8px; padding: 18px; text-align: left; font-size: 15px; color: #cdd6f4; margin-top: 10px;}
                QPushButton:hover { border-color: #89b4fa; }
                QPushButton:checked { background: #89b4fa; color: #1e1e2e; font-weight: bold; }
            """)
            self.btn_group.addButton(btn)
            self.content_layout.addWidget(btn)
            
        self.content_layout.addStretch()
        
        submit_btn = QPushButton("Confirm Answer")
        submit_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        submit_btn.clicked.connect(self.submit_answer)
        submit_btn.setStyleSheet("background-color: #a6e3a1; color: #1e1e2e; font-size: 16px; font-weight: 900; padding: 15px; border-radius: 8px;")
        self.content_layout.addWidget(submit_btn)

    def submit_answer(self):
        btn = self.btn_group.checkedButton()
        if not btn: return
        
        result = self.assessment_engine.submit_answer(btn.text())
        if result['status'] == 'complete':
            self.show_results(result['results'])
        else:
            self.load_question_view()

    def finish_assessment(self):
        results = self.assessment_engine.finish_assessment()
        self.show_results(results)

    def show_results(self, results):
        self.clear_content()
        self.lbl_title.setText("ASSESSMENT DEBRIEF")
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("border: none; background: transparent;")
        
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(20)
        
        score_frame = QFrame()
        score_frame.setStyleSheet("background: #181825; border-radius: 12px; border: 1px solid #313244; padding: 30px;")
        s_layout = QVBoxLayout(score_frame)
        score_lbl = QLabel(f"{results['score']} / {results['total_questions']}")
        score_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        score_lbl.setStyleSheet("font-size: 50px; font-weight: 900; color: #fab387; border: none;")
        s_layout.addWidget(score_lbl)
        
        pass_fail = "✅ PASSED - NEW TIER UNLOCKED" if results['passed'] else "❌ FAILED - REVIEW CONCEPTS"
        col = "#a6e3a1" if results['passed'] else "#f38ba8"
        status = QLabel(pass_fail)
        status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        status.setStyleSheet(f"font-size: 20px; font-weight: bold; color: {col}; border: none;")
        s_layout.addWidget(status)
        layout.addWidget(score_frame)
        
        for ans in results['answers']:
            row = QFrame()
            b_col = "#a6e3a1" if ans['is_correct'] else "#f38ba8"
            row.setStyleSheet(f"QFrame {{ background: #1e1e2e; border-left: 5px solid {b_col}; border-radius: 6px; padding: 15px; }}")
            r_layout = QVBoxLayout(row)
            
            q_lbl = QLabel(ans.get('question_text', ''))
            q_lbl.setWordWrap(True)
            q_lbl.setStyleSheet("font-weight: bold; color: white; font-size: 15px; border: none;")
            r_layout.addWidget(q_lbl)
            
            if not ans['is_correct']:
                user_lbl = QLabel(f"❌ You said: {ans['selected_answer']}")
                user_lbl.setStyleSheet("color: #f38ba8; border: none; font-size: 14px; margin-top: 5px;")
                corr_lbl = QLabel(f"✅ Correct: {ans['correct_answer']}")
                corr_lbl.setStyleSheet("color: #a6e3a1; border: none; font-size: 14px;")
                r_layout.addWidget(user_lbl)
                r_layout.addWidget(corr_lbl)
                
                exp = QLabel("<i>Go to the Knowledge Base -> Mistake Review tab to get AI Coaching on this topic.</i>")
                exp.setStyleSheet("color: #a6adc8; margin-top: 10px; border: none;")
                r_layout.addWidget(exp)
            else:
                res_lbl = QLabel(f"✅ Correct: {ans['selected_answer']}")
                res_lbl.setStyleSheet("color: #a6e3a1; border: none;")
                r_layout.addWidget(res_lbl)
                
            layout.addWidget(row)

        btn_layout = QHBoxLayout()
        retry_btn = QPushButton("Return to Menu")
        retry_btn.clicked.connect(self.show_tier_selection)
        retry_btn.setStyleSheet("background-color: #313244; color: white; padding: 15px; border-radius: 8px; font-weight: bold;")
        
        btn_layout.addWidget(retry_btn)
        layout.addLayout(btn_layout)
        layout.addStretch()
        
        scroll.setWidget(container)
        self.content_layout.addWidget(scroll)
        
        self.assessment_completed.emit(results)
