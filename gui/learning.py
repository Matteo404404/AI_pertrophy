"""
Scientific Hypertrophy Trainer - Learning Center
Study wrong answers grouped by concept categories
"""

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QFrame, QScrollArea, QGroupBox,
                             QTextEdit, QListWidget, QListWidgetItem,
                             QStackedWidget)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont


class LearningWidget(QWidget):
    """Learning center for reviewing missed questions and concepts"""
    
    def __init__(self, db_manager, user_manager):
        super().__init__()
        self.db = db_manager
        self.user_manager = user_manager
        self.init_ui()
    
    def init_ui(self):
        """Initialize learning center interface"""
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
            QListWidget {
                background-color: #ffffff;
                border: 2px solid #e9ecef;
                border-radius: 8px;
                color: #1a1a1a;
            }
            QListWidget::item {
                padding: 15px;
                border-radius: 6px;
                margin: 5px 0px;
                background-color: #f8f9fa;
                color: #1a1a1a;
            }
            QListWidget::item:hover {
                background-color: #e9ecef;
            }
            QListWidget::item:selected {
                background-color: #667eea;
                color: #ffffff;
            }
        """)
        
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(40, 40, 40, 40)
        
        # Header
        header = QLabel("Learning Center")
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header.setStyleSheet("font-size: 32px; font-weight: bold; color: #667eea !important; margin-bottom: 10px;")
        main_layout.addWidget(header)
        
        subtitle = QLabel("Review missed questions and master hypertrophy concepts")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet("font-size: 16px; color: #6c757d !important; margin-bottom: 30px;")
        main_layout.addWidget(subtitle)
        
        # Content area
        content_layout = QHBoxLayout()
        
        # Left side - Concept categories
        self.create_category_list(content_layout)
        
        # Right side - Questions and explanations
        self.create_content_area(content_layout)
        
        main_layout.addLayout(content_layout)
    
    def create_category_list(self, parent_layout):
        """Create concept category list"""
        category_frame = QFrame()
        category_frame.setStyleSheet("""
            QFrame {
                background-color: #ffffff;
                border: 2px solid #e9ecef;
                border-radius: 12px;
                padding: 20px;
            }
            QFrame QLabel {
                color: #1a1a1a !important;
            }
        """)
        category_frame.setMaximumWidth(350)
        
        category_layout = QVBoxLayout(category_frame)
        
        # Section title
        category_title = QLabel("Study Priority")
        category_title.setStyleSheet("font-size: 20px; font-weight: bold; color: #1a1a1a !important; margin-bottom: 15px;")
        category_layout.addWidget(category_title)
        
        # Category list
        self.category_list = QListWidget()
        self.category_list.currentItemChanged.connect(self.on_category_selected)
        category_layout.addWidget(self.category_list)
        
        # Study progress info
        self.progress_label = QLabel("Select a category to begin studying")
        self.progress_label.setWordWrap(True)
        self.progress_label.setStyleSheet("""
            font-size: 12px;
            color: #6c757d !important;
            padding: 10px;
            background-color: #f8f9fa;
            border-radius: 6px;
            margin-top: 15px;
        """)
        category_layout.addWidget(self.progress_label)
        
        parent_layout.addWidget(category_frame)
    
    def create_content_area(self, parent_layout):
        """Create content display area"""
        content_frame = QFrame()
        content_frame.setStyleSheet("""
            QFrame {
                background-color: #ffffff;
                border: 2px solid #e9ecef;
                border-radius: 12px;
                padding: 20px;
            }
        """)
        
        content_layout = QVBoxLayout(content_frame)
        
        # Content area with scroll
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background-color: transparent; }")
        
        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout(self.content_widget)
        
        # Initial empty state
        empty_state = QLabel("📚\n\nSelect a concept category\nto view learning materials")
        empty_state.setAlignment(Qt.AlignmentFlag.AlignCenter)
        empty_state.setStyleSheet("font-size: 18px; color: #adb5bd !important; padding: 100px;")
        self.content_layout.addWidget(empty_state)
        
        scroll.setWidget(self.content_widget)
        content_layout.addWidget(scroll)
        
        parent_layout.addWidget(content_frame)
    
    def refresh_data(self):
        """Refresh learning center data"""
        current_user = self.user_manager.get_current_user()
        if not current_user:
            return
        
        # Get all assessments for user
        assessments = self.db.get_user_assessments(current_user['id'])
        
        # Collect all wrong answers
        wrong_answers_by_concept = self.categorize_wrong_answers(assessments)
        
        # Populate category list
        self.category_list.clear()
        
        if not wrong_answers_by_concept:
            item = QListWidgetItem("✅ No mistakes yet!")
            item.setData(Qt.ItemDataRole.UserRole, None)
            self.category_list.addItem(item)
            self.progress_label.setText("You haven't made any mistakes yet. Keep learning!")
            return
        
        # Priority order
        priority_order = [
            'Training Volume',
            'Training Intensity',
            'Training Frequency',
            'Exercise Selection',
            'Nutrition',
            'Recovery & Sleep',
            'General Concepts'
        ]
        
        total_mistakes = sum(len(questions) for questions in wrong_answers_by_concept.values())
        
        for concept in priority_order:
            if concept in wrong_answers_by_concept:
                questions = wrong_answers_by_concept[concept]
                item_text = f"{concept} ({len(questions)} mistakes)"
                item = QListWidgetItem(item_text)
                item.setData(Qt.ItemDataRole.UserRole, {'concept': concept, 'questions': questions})
                self.category_list.addItem(item)
        
        # Add any other concepts
        for concept, questions in wrong_answers_by_concept.items():
            if concept not in priority_order:
                item_text = f"{concept} ({len(questions)} mistakes)"
                item = QListWidgetItem(item_text)
                item.setData(Qt.ItemDataRole.UserRole, {'concept': concept, 'questions': questions})
                self.category_list.addItem(item)
        
        self.progress_label.setText(f"Total mistakes to review: {total_mistakes}")
    
    def categorize_wrong_answers(self, assessments):
        """Categorize wrong answers by concept"""
        wrong_by_concept = {}
        
        for assessment in assessments:
            answers = self.db.get_assessment_answers(assessment['id'])
            for answer in answers:
                if not answer['is_correct']:
                    concept = self.get_question_concept(answer['question_text'])
                    if concept not in wrong_by_concept:
                        wrong_by_concept[concept] = []
                    wrong_by_concept[concept].append({
                        'question': answer['question_text'],
                        'selected': answer['selected_answer'],
                        'correct': answer['correct_answer'],
                        'question_id': answer['question_id']
                    })
        
        return wrong_by_concept
    
    def get_question_concept(self, question_text):
        """Determine concept category from question text"""
        text_lower = question_text.lower()
        
        if any(kw in text_lower for kw in ['volume', 'sets', 'mrv', 'mev', 'total sets']):
            return 'Training Volume'
        elif any(kw in text_lower for kw in ['intensity', 'failure', 'rir', 'rpe', 'effort', 'load']):
            return 'Training Intensity'
        elif any(kw in text_lower for kw in ['frequency', 'per week', 'training days', 'sessions per']):
            return 'Training Frequency'
        elif any(kw in text_lower for kw in ['exercise', 'movement', 'range of motion', 'length', 'stretch']):
            return 'Exercise Selection'
        elif any(kw in text_lower for kw in ['protein', 'nutrition', 'diet', 'calories', 'carb', 'fat', 'meal']):
            return 'Nutrition'
        elif any(kw in text_lower for kw in ['sleep', 'recovery', 'rest', 'deload']):
            return 'Recovery & Sleep'
        else:
            return 'General Concepts'
    
    def on_category_selected(self, current, previous):
        """Handle category selection"""
        if not current:
            return
        
        data = current.data(Qt.ItemDataRole.UserRole)
        if not data:
            return
        
        concept = data['concept']
        questions = data['questions']
        
        # Clear existing content
        while self.content_layout.count():
            child = self.content_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        
        # Concept header
        concept_header = QLabel(f"📖 {concept}")
        concept_header.setStyleSheet("font-size: 24px; font-weight: bold; color: #667eea !important; margin-bottom: 20px;")
        self.content_layout.addWidget(concept_header)
        
        # Progress info
        progress_info = QLabel(f"{len(questions)} questions to review in this category")
        progress_info.setStyleSheet("font-size: 14px; color: #6c757d !important; margin-bottom: 30px;")
        self.content_layout.addWidget(progress_info)
        
        # Display each question
        for i, q_data in enumerate(questions, 1):
            question_card = self.create_question_card(i, q_data)
            self.content_layout.addWidget(question_card)
        
        self.content_layout.addStretch()
    
    def create_question_card(self, number, question_data):
        """Create individual question review card"""
        card = QFrame()
        card.setStyleSheet("""
            QFrame {
                background-color: #f8f9fa;
                border: 2px solid #e9ecef;
                border-radius: 12px;
                padding: 25px;
                margin: 10px 0px;
            }
            QFrame QLabel {
                color: #1a1a1a !important;
            }
        """)
        
        card_layout = QVBoxLayout(card)
        
        # Question number
        q_number = QLabel(f"Question {number}")
        q_number.setStyleSheet("font-size: 14px; font-weight: bold; color: #6c757d !important; margin-bottom: 10px;")
        card_layout.addWidget(q_number)
        
        # Question text
        question = QLabel(question_data['question'])
        question.setWordWrap(True)
        question.setStyleSheet("font-size: 16px; font-weight: bold; color: #1a1a1a !important; margin-bottom: 20px; line-height: 1.5;")
        card_layout.addWidget(question)
        
        # Your answer (wrong)
        your_answer_label = QLabel("❌ Your Answer:")
        your_answer_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #dc3545 !important; margin-bottom: 5px;")
        card_layout.addWidget(your_answer_label)
        
        your_answer = QLabel(question_data['selected'])
        your_answer.setWordWrap(True)
        your_answer.setStyleSheet("""
            font-size: 14px;
            color: #721c24 !important;
            background-color: #f8d7da;
            padding: 10px;
            border-radius: 6px;
            border-left: 4px solid #dc3545;
            margin-bottom: 15px;
        """)
        card_layout.addWidget(your_answer)
        
        # Correct answer
        correct_answer_label = QLabel("✅ Correct Answer:")
        correct_answer_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #28a745 !important; margin-bottom: 5px;")
        card_layout.addWidget(correct_answer_label)
        
        correct_answer = QLabel(question_data['correct'])
        correct_answer.setWordWrap(True)
        correct_answer.setStyleSheet("""
            font-size: 14px;
            color: #155724 !important;
            background-color: #d4edda;
            padding: 10px;
            border-radius: 6px;
            border-left: 4px solid #28a745;
            margin-bottom: 15px;
        """)
        card_layout.addWidget(correct_answer)
        
        # Explanation
        explanation_label = QLabel("💡 Key Learning Point:")
        explanation_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #ffc107 !important; margin-bottom: 5px;")
        card_layout.addWidget(explanation_label)
        
        explanation = QLabel(self.get_explanation_for_question(question_data['question']))
        explanation.setWordWrap(True)
        explanation.setStyleSheet("""
            font-size: 14px;
            color: #856404 !important;
            background-color: #fff3cd;
            padding: 10px;
            border-radius: 6px;
            border-left: 4px solid #ffc107;
            line-height: 1.5;
        """)
        card_layout.addWidget(explanation)
        
        return card
    
    def get_explanation_for_question(self, question_text):
        """Get explanation for a question"""
        text_lower = question_text.lower()
        
        if 'volume' in text_lower or 'sets' in text_lower:
            return "Training volume is the total amount of work performed. Research shows progressive overload through volume increases is a primary driver of hypertrophy. Start with MEV (Minimum Effective Volume) and gradually increase towards MRV (Maximum Recoverable Volume)."
        elif 'rir' in text_lower or 'failure' in text_lower:
            return "RIR (Reps In Reserve) is a measure of proximity to failure. Research suggests training 0-3 RIR produces similar hypertrophy, but higher RIR allows for more volume accumulation with less fatigue. Most sets should be 1-3 RIR."
        elif 'frequency' in text_lower:
            return "Training frequency refers to how often you train a muscle per week. Research shows 2-3x per week frequency allows for optimal volume distribution and recovery, leading to superior hypertrophy compared to once-weekly training."
        elif 'protein' in text_lower:
            return "Protein intake of 1.6-2.2g per kg bodyweight is optimal for hypertrophy. Distribute protein evenly across 3-5 meals for maximal muscle protein synthesis stimulation throughout the day."
        elif 'exercise' in text_lower or 'stretch' in text_lower:
            return "Exercise selection should prioritize movements that train muscles in their lengthened position for superior hypertrophy. Tension at long muscle lengths creates more mechanical stimulus and muscle damage."
        elif 'sleep' in text_lower:
            return "Sleep is critical for recovery and hypertrophy. Aim for 7-9 hours of quality sleep per night. Poor sleep impairs protein synthesis, increases cortisol, and reduces training performance."
        else:
            return "Review the correct answer and understand the underlying principle. Evidence-based training requires understanding the science behind each decision."
