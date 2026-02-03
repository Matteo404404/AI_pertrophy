"""
Scientific Hypertrophy Trainer - Assessment Interface
Complete assessment system with tier selection, questions, and results
"""

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QFrame, QRadioButton, QProgressBar,
                             QScrollArea, QButtonGroup, QMessageBox, QTextEdit)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QFont
from datetime import datetime


class AssessmentWidget(QWidget):
    """Assessment interface with tier progression"""
    
    assessment_completed = pyqtSignal(dict)
    
    def __init__(self, assessment_engine, user_manager):
        super().__init__()
        self.assessment_engine = assessment_engine
        self.user_manager = user_manager
        self.current_assessment = None
        self.current_question = None
        self.start_time = None
        self.init_ui()
        self.refresh_tiers()
    
    def init_ui(self):
        """Initialize assessment interface"""
        # FORCE LIGHT THEME FOR ALL ELEMENTS
        self.setStyleSheet("""
            QWidget {
                background-color: #f8f9fa;
                color: #1a1a1a;
            }
            QLabel {
                color: #1a1a1a !important;
                background-color: transparent !important;
            }
            QFrame {
                background-color: #ffffff;
                border: 2px solid #e9ecef;
                border-radius: 12px;
            }
            QRadioButton {
                color: #1a1a1a !important;
            }
            QTextEdit {
                background-color: #ffffff;
                color: #1a1a1a;
                border: 2px solid #e9ecef;
            }
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
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #667eea, stop:1 #764ba2);
                border-radius: 8px;
            }
        """)
        
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(40, 40, 40, 40)
        
        # Header
        self.create_header(main_layout)
        
        # Content stack
        self.content_stack = QVBoxLayout()
        main_layout.addLayout(self.content_stack)
        
        # Create different views
        self.create_tier_selection_view()
        self.create_question_view()
        self.create_results_view()
        
        # Show tier selection by default
        self.show_tier_selection()
    
    def create_header(self, parent_layout):
        """Create assessment header"""
        header_frame = QFrame()
        header_frame.setStyleSheet("background-color: transparent; border: none;")
        header_layout = QVBoxLayout(header_frame)
        
        title = QLabel("Knowledge Assessment")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 32px; font-weight: bold; color: #667eea !important; margin-bottom: 10px;")
        
        subtitle = QLabel("Test your hypertrophy knowledge and unlock new tiers")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet("font-size: 16px; color: #6c757d !important; margin-bottom: 30px;")
        
        header_layout.addWidget(title)
        header_layout.addWidget(subtitle)
        parent_layout.addWidget(header_frame)
    
    def create_tier_selection_view(self):
        """Create tier selection interface"""
        self.tier_selection_widget = QWidget()
        tier_layout = QVBoxLayout(self.tier_selection_widget)
        
        # Section title
        section_title = QLabel("Select Assessment Tier")
        section_title.setStyleSheet("font-size: 24px; font-weight: bold; color: #1a1a1a !important; margin-bottom: 30px;")
        tier_layout.addWidget(section_title)
        
        # Tier cards container
        self.tier_cards_layout = QVBoxLayout()
        tier_layout.addLayout(self.tier_cards_layout)
        tier_layout.addStretch()
    
    def create_question_view(self):
        """Create question interface"""
        self.question_widget = QWidget()
        question_layout = QVBoxLayout(self.question_widget)
        
        # Progress section
        progress_frame = QFrame()
        progress_layout = QVBoxLayout(progress_frame)
        
        # Assessment title
        self.assessment_title = QLabel("Tier 1: Fundamentals")
        self.assessment_title.setStyleSheet("font-size: 24px; font-weight: bold; color: #667eea !important; margin-bottom: 15px;")
        progress_layout.addWidget(self.assessment_title)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        progress_layout.addWidget(self.progress_bar)
        
        # Progress text
        self.progress_text = QLabel("Question 1 of 20")
        self.progress_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.progress_text.setStyleSheet("font-size: 14px; color: #6c757d !important; margin-top: 10px;")
        progress_layout.addWidget(self.progress_text)
        
        question_layout.addWidget(progress_frame)
        
        # Question card
        question_card = QFrame()
        self.question_layout = QVBoxLayout(question_card)
        
        # Question text
        self.question_text = QLabel()
        self.question_text.setWordWrap(True)
        self.question_text.setStyleSheet("font-size: 18px; font-weight: bold; color: #1a1a1a !important; margin-bottom: 15px; line-height: 1.5;")
        self.question_layout.addWidget(self.question_text)
        
        # Question explanation
        self.question_explanation = QLabel()
        self.question_explanation.setWordWrap(True)
        self.question_explanation.setStyleSheet("font-size: 14px; color: #6c757d !important; font-style: italic; margin-bottom: 25px;")
        self.question_layout.addWidget(self.question_explanation)
        
        # Answer options
        self.answer_group = QButtonGroup()
        self.answer_buttons = []
        self.answers_layout = QVBoxLayout()
        self.question_layout.addLayout(self.answers_layout)
        
        question_layout.addWidget(question_card)
        
        # Navigation buttons
        nav_layout = QHBoxLayout()
        
        self.pause_btn = QPushButton("⏸️ Pause")
        self.pause_btn.clicked.connect(self.pause_assessment)
        self.pause_btn.setStyleSheet("background-color: #6c757d; color: #ffffff; padding: 12px 24px; border-radius: 8px; font-size: 14px; font-weight: 600;")
        
        self.quit_btn = QPushButton("❌ Quit Assessment")
        self.quit_btn.clicked.connect(self.quit_assessment)
        self.quit_btn.setStyleSheet("background-color: #dc3545; color: #ffffff; padding: 12px 24px; border-radius: 8px; font-size: 14px; font-weight: 600;")
        
        nav_layout.addWidget(self.pause_btn)
        nav_layout.addWidget(self.quit_btn)
        nav_layout.addStretch()
        
        self.submit_btn = QPushButton("Submit Answer")
        self.submit_btn.clicked.connect(self.submit_answer)
        self.submit_btn.setMinimumHeight(50)
        self.submit_btn.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: #ffffff;
                font-size: 16px;
                font-weight: bold;
                padding: 16px;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #218838;
            }
            QPushButton:disabled {
                background-color: #e9ecef;
                color: #adb5bd;
            }
        """)
        self.submit_btn.setEnabled(False)
        
        nav_layout.addWidget(self.submit_btn)
        question_layout.addLayout(nav_layout)
    
    def create_results_view(self):
        """Create results interface"""
        self.results_widget = QWidget()
        results_layout = QVBoxLayout(self.results_widget)
        
        results_card = QFrame()
        self.results_layout = QVBoxLayout(results_card)
        
        # Results title
        self.results_title = QLabel("Assessment Complete!")
        self.results_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.results_title.setStyleSheet("font-size: 32px; font-weight: bold; color: #28a745 !important; margin-bottom: 30px;")
        self.results_layout.addWidget(self.results_title)
        
        # Score display
        self.score_label = QLabel()
        self.score_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.score_label.setStyleSheet("font-size: 48px; font-weight: bold; color: #667eea !important; margin-bottom: 20px;")
        self.results_layout.addWidget(self.score_label)
        
        # Status message
        self.status_message = QLabel()
        self.status_message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_message.setWordWrap(True)
        self.status_message.setStyleSheet("font-size: 18px; color: #1a1a1a !important; margin-bottom: 30px; line-height: 1.5;")
        self.results_layout.addWidget(self.status_message)
        
        # Detailed breakdown
        self.breakdown_text = QTextEdit()
        self.breakdown_text.setReadOnly(True)
        self.breakdown_text.setMaximumHeight(200)
        self.results_layout.addWidget(self.breakdown_text)
        
        # Action buttons
        actions_layout = QHBoxLayout()
        
        self.back_to_tiers_btn = QPushButton("← Back to Tiers")
        self.back_to_tiers_btn.clicked.connect(self.show_tier_selection)
        self.back_to_tiers_btn.setStyleSheet("background-color: #6c757d; color: #ffffff; padding: 12px 24px; border-radius: 8px; font-size: 14px; font-weight: 600;")
        
        self.view_learning_btn = QPushButton("📚 View Learning Center")
        self.view_learning_btn.setStyleSheet("background-color: #667eea; color: #ffffff; padding: 12px 24px; border-radius: 8px; font-size: 14px; font-weight: 600;")
        
        actions_layout.addWidget(self.back_to_tiers_btn)
        actions_layout.addStretch()
        actions_layout.addWidget(self.view_learning_btn)
        
        self.results_layout.addLayout(actions_layout)
        results_layout.addWidget(results_card)
        results_layout.addStretch()
    
    def create_tier_card(self, tier_level, tier_data, is_accessible, is_completed):
        """Create individual tier selection card"""
        card = QFrame()
        
        # Determine card styling based on status
        if is_completed:
            border_color = "#28a745"
            bg_color = "#e8f5e9"
            status_text = "✅ Completed"
            status_color = "#28a745"
        elif is_accessible:
            border_color = "#667eea"
            bg_color = "#ede7f6"
            status_text = "🔓 Available"
            status_color = "#667eea"
        else:
            border_color = "#e9ecef"
            bg_color = "#f8f9fa"
            status_text = "🔒 Locked"
            status_color = "#6c757d"
        
        card.setStyleSheet(f"""
            QFrame {{
                background-color: {bg_color};
                border: 2px solid {border_color};
                border-radius: 12px;
                padding: 25px;
                margin: 10px 0px;
                min-height: 120px;
            }}
            QFrame QLabel {{
                color: #1a1a1a !important;
            }}
        """)
        
        card_layout = QHBoxLayout(card)
        
        # Left side - Tier info
        info_layout = QVBoxLayout()
        
        tier_title = QLabel(f"Tier {tier_level + 1}: {tier_data['title']}")
        tier_title.setStyleSheet(f"font-size: 20px; font-weight: bold; color: {status_color} !important; margin-bottom: 8px;")
        
        tier_desc = QLabel(tier_data['description'])
        tier_desc.setWordWrap(True)
        tier_desc.setStyleSheet("font-size: 14px; color: #495057 !important; margin-bottom: 15px;")
        
        tier_details = QLabel(f"{tier_data['questions']} questions • {int(tier_data['passing_percentage'] * 100)}% to pass")
        tier_details.setStyleSheet("font-size: 12px; color: #6c757d !important;")
        
        info_layout.addWidget(tier_title)
        info_layout.addWidget(tier_desc)
        info_layout.addWidget(tier_details)
        info_layout.addStretch()
        
        card_layout.addLayout(info_layout)
        card_layout.addStretch()
        
        # Right side - Status and action
        action_layout = QVBoxLayout()
        action_layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignRight)
        
        status_label = QLabel(status_text)
        status_label.setStyleSheet(f"""
            background-color: {status_color};
            color: #ffffff;
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 12px;
        """)
        status_label.setMaximumWidth(120)
        action_layout.addWidget(status_label)
        
        if is_accessible and not is_completed:
            start_btn = QPushButton("Start Assessment")
            start_btn.clicked.connect(lambda: self.start_assessment(tier_level))
            start_btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {status_color};
                    color: #ffffff;
                    padding: 12px 20px;
                    font-weight: bold;
                    margin-top: 15px;
                    border-radius: 6px;
                }}
            """)
            action_layout.addWidget(start_btn)
        elif is_completed:
            retake_btn = QPushButton("View Results")
            retake_btn.setStyleSheet("""
                background-color: #6c757d;
                color: #ffffff;
                padding: 12px 20px;
                font-weight: bold;
                margin-top: 15px;
                border-radius: 6px;
            """)
            action_layout.addWidget(retake_btn)
        
        card_layout.addLayout(action_layout)
        return card
    
    def refresh_tiers(self):
        """Refresh tier selection display"""
        while self.tier_cards_layout.count():
            child = self.tier_cards_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        
        current_user = self.user_manager.get_current_user()
        if not current_user:
            return
        
        tier_status = self.user_manager.get_tier_status()
        
        tier_data_map = {
            0: {
                'title': 'Fundamentals',
                'description': 'Essential hypertrophy principles - mechanical tension, volume, frequency, and nutrition basics',
                'questions': 20,
                'passing_percentage': 0.8
            },
            1: {
                'title': 'Intermediate',
                'description': 'Advanced concepts - effective reps, volume distribution, muscle length relationships',
                'questions': 20,
                'passing_percentage': 0.8
            },
            2: {
                'title': 'Advanced',
                'description': 'Expert optimization - MRV, autoregulation, periodization, and regional hypertrophy',
                'questions': 20,
                'passing_percentage': 0.8
            }
        }
        
        tiers = tier_status.get('tiers', [])
        for i, tier in enumerate(tiers):
            tier_data = tier_data_map.get(i, {})
            tier_card = self.create_tier_card(
                i,
                tier_data,
                tier.get('accessible', False),
                tier.get('completed', False)
            )
            self.tier_cards_layout.addWidget(tier_card)
    
    def start_assessment(self, tier_level):
        """Start assessment for specified tier"""
        try:
            if not self.user_manager.can_access_tier(tier_level):
                QMessageBox.warning(self, "Access Denied",
                                    f"You must complete previous tiers to access Tier {tier_level + 1}.")
                return
            
            assessment_info = self.assessment_engine.start_assessment(tier_level)
            self.current_assessment = assessment_info
            self.start_time = datetime.now()
            
            tier_names = ["Fundamentals", "Intermediate", "Advanced"]
            self.assessment_title.setText(f"Tier {tier_level + 1}: {tier_names[tier_level]}")
            
            self.load_next_question()
            self.show_question_view()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start assessment: {str(e)}")
    
    def load_next_question(self):
        """Load the next question"""
        question = self.assessment_engine.get_current_question()
        if not question:
            self.finish_assessment()
            return
        
        self.current_question = question
        
        self.progress_bar.setMaximum(question['total_questions'])
        self.progress_bar.setValue(question['question_number'])
        self.progress_text.setText(f"Question {question['question_number']} of {question['total_questions']}")
        
        self.question_text.setText(question['question'])
        self.question_explanation.setText(question.get('explanation', ''))
        
        # Clear previous answers
        for btn in self.answer_buttons:
            self.answer_group.removeButton(btn)
            btn.setParent(None)
        self.answer_buttons.clear()
        
        # Create new answer options
        for i, option in enumerate(question['options']):
            radio_btn = QRadioButton(option['text'])
            radio_btn.setStyleSheet("""
                QRadioButton {
                    font-size: 16px;
                    padding: 15px;
                    margin: 8px 0px;
                    background-color: #f8f9fa;
                    border-radius: 8px;
                    color: #1a1a1a !important;
                }
                QRadioButton:hover {
                    background-color: #e9ecef;
                }
                QRadioButton::indicator {
                    width: 20px;
                    height: 20px;
                    border-radius: 10px;
                    border: 2px solid #dee2e6;
                    background-color: #ffffff;
                    margin-right: 15px;
                }
                QRadioButton::indicator:checked {
                    border-color: #667eea;
                    background-color: #667eea;
                }
            """)
            radio_btn.toggled.connect(self.on_answer_selected)
            self.answer_group.addButton(radio_btn, i)
            self.answer_buttons.append(radio_btn)
            self.answers_layout.addWidget(radio_btn)
        
        self.submit_btn.setEnabled(False)
    
    def on_answer_selected(self):
        """Handle answer selection"""
        self.submit_btn.setEnabled(True)
    
    def submit_answer(self):
        """Submit the current answer"""
        selected_btn = self.answer_group.checkedButton()
        if not selected_btn:
            return
        
        selected_text = selected_btn.text()
        
        try:
            result = self.assessment_engine.submit_answer(selected_text)
            if result['status'] == 'complete':
                self.show_results(result['results'])
            else:
                self.load_next_question()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to submit answer: {str(e)}")
    
    def pause_assessment(self):
        """Pause the current assessment"""
        reply = QMessageBox.question(
            self, 'Pause Assessment',
            'Do you want to pause this assessment?\nYou can resume later from where you left off.',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.show_tier_selection()
    
    def quit_assessment(self):
        """Quit the current assessment"""
        reply = QMessageBox.question(
            self, 'Quit Assessment',
            'Are you sure you want to quit this assessment?\nAll progress will be lost.',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.assessment_engine.reset()
            self.show_tier_selection()
    
    def show_results(self, results):
        """Display assessment results"""
        score_text = f"{results['score']}/{results['total_questions']}"
        percentage = results['percentage']
        
        self.score_label.setText(score_text)
        
        if results['passed']:
            self.results_title.setText("🎉 Assessment Passed!")
            self.results_title.setStyleSheet("font-size: 32px; font-weight: bold; color: #28a745 !important; margin-bottom: 30px;")
            status_msg = f"Excellent! You scored {percentage:.1f}% ({score_text})\n"
            if results.get('next_tier_unlocked'):
                status_msg += "🚀 You've unlocked the next tier!"
            else:
                status_msg += "You've mastered this tier's concepts."
        else:
            self.results_title.setText("📚 Keep Learning!")
            self.results_title.setStyleSheet("font-size: 32px; font-weight: bold; color: #ffc107 !important; margin-bottom: 30px;")
            passing_score = results['passing_score']
            status_msg = f"You scored {percentage:.1f}% ({score_text})\n"
            status_msg += f"You need {passing_score} correct answers ({results['passing_percentage']:.0f}%) to pass.\n"
            status_msg += "Review the concepts and try again!"
        
        self.status_message.setText(status_msg)
        
        wrong_answers = [ans for ans in results['answers'] if not ans['is_correct']]
        breakdown_text = f"Assessment Summary:\n"
        breakdown_text += f"• Correct: {results['score']} questions\n"
        breakdown_text += f"• Incorrect: {len(wrong_answers)} questions\n"
        breakdown_text += f"• Accuracy: {percentage:.1f}%\n\n"
        
        if wrong_answers:
            breakdown_text += "Areas for improvement:\n"
            concepts = {}
            for ans in wrong_answers:
                concept = self.categorize_question(ans['question_text'])
                if concept not in concepts:
                    concepts[concept] = []
                concepts[concept].append(ans['question_text'][:60] + "...")
            
            for concept, questions in concepts.items():
                breakdown_text += f"• {concept}: {len(questions)} questions\n"
        
        self.breakdown_text.setPlainText(breakdown_text)
        self.assessment_completed.emit(results)
        self.show_results_view()
    
    def categorize_question(self, question_text):
        """Simple question categorization"""
        text_lower = question_text.lower()
        
        if any(kw in text_lower for kw in ['volume', 'sets', 'mrv']):
            return 'Training Volume'
        elif any(kw in text_lower for kw in ['intensity', 'failure', 'rir']):
            return 'Training Intensity'
        elif any(kw in text_lower for kw in ['protein', 'nutrition', 'diet']):
            return 'Nutrition'
        elif any(kw in text_lower for kw in ['sleep', 'recovery']):
            return 'Recovery & Sleep'
        elif any(kw in text_lower for kw in ['frequency']):
            return 'Training Frequency'
        else:
            return 'General Concepts'
    
    def finish_assessment(self):
        """Finish the assessment and show results"""
        results = self.assessment_engine.finish_assessment()
        self.show_results(results)
    
    def show_tier_selection(self):
        """Show tier selection view"""
        self.clear_content_stack()
        self.content_stack.addWidget(self.tier_selection_widget)
        self.refresh_tiers()
    
    def show_question_view(self):
        """Show question view"""
        self.clear_content_stack()
        self.content_stack.addWidget(self.question_widget)
    
    def show_results_view(self):
        """Show results view"""
        self.clear_content_stack()
        self.content_stack.addWidget(self.results_widget)
    
    def clear_content_stack(self):
        """Clear content stack"""
        while self.content_stack.count():
            child = self.content_stack.takeAt(0)
            if child.widget():
                child.widget().setParent(None)
