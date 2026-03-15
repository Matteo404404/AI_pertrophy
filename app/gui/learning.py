"""
Scientific Hypertrophy Trainer - Learning Center v3.1 (FIXED)
- Fixed: Indentation errors
- Fixed: Database content loading
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QListWidget, QFrame,
    QPushButton, QTextEdit, QScrollArea, QTabWidget, QGridLayout, QProgressBar
)
from PyQt6.QtCore import Qt 

# --- THE CUTTING-EDGE SCIENTIFIC KNOWLEDGE BASE ---
KNOWLEDGE_DB = {
    "Hypertrophy Mechanisms": {
        "Mechanical Tension": "The primary driver of hypertrophy. High tension + slow shortening speeds (involuntary slowing due to load/fatigue) recruits high-threshold motor units.",
        "Metabolic Stress": "Largely an artifact of training. Cell swelling and metabolite accumulation (the pump) do not independently drive significant growth without mechanical tension.",
        "Muscle Damage": "A byproduct, not a goal. Excessive damage (severe DOMS) competes with recovery resources and delays the protein synthesis needed for actual tissue accretion."
    },
    "Volume & Intensity": {
        "The Volume Myth": "More is not better. 1-3 working sets taken to true failure (0-1 RIR) maximizes motor unit recruitment. Doing 5+ sets per session accumulates CNS fatigue without adding effective reps.",
        "RIR (Reps In Reserve)": "Hypertrophy occurs largely in the last ~5 reps before failure. If a set stops at >2 RIR, mechanical tension on high-threshold motor units is insufficient.",
        "Rep Ranges": "4-10 reps is the biomechanical sweet spot. >12 reps generates disproportionate cardiovascular and central fatigue before mechanical tension can fully load the muscle."
    },
    "Advanced Biomechanics": {
        "Alignment vs. Matching": "Naive 'Neuromechanical Matching' assumes resistance profiles must perfectly match strength curves. In reality, aligning the line of force with muscle fiber orientation (Alignment) and overloading the stretched position is far more critical.",
        "Stability & Output": "High external stability (e.g., a chest-supported row or Hack Squat) removes balancing demands from the CNS. This allows 100% of neural drive to be directed to the prime movers.",
        "Active Insufficiency": "A two-joint muscle cannot generate maximal tension when fully shortened across both joints simultaneously (e.g., hamstrings during a lying leg curl with hips extended)."
    },
    "Execution & Recovery": {
        "Concentric Intent": "The concentric phase must be explosive (maximal intended velocity). The bar should only slow down involuntarily due to fatigue or heavy load to satisfy the force-velocity relationship.",
        "Systemic vs. Local Fatigue": "Compound free-weight lifts (Deadlifts) create massive Systemic/CNS fatigue. Machine isolations create Local fatigue. Programs must manage the systemic 'budget'.",
        "Sleep Architecture": "Deep sleep releases Growth Hormone. REM sleep restores neural drive. <7 hours of sleep drastically reduces fat loss and muscle retention."
    }
}

class LearningWidget(QWidget):
    def __init__(self, db_manager, user_manager):
        super().__init__()
        self.db = db_manager
        self.user_manager = user_manager
        self.init_ui()
    
    def init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(30, 30, 30, 30)
        
        header = QLabel("Knowledge Base & Education")
        header.setStyleSheet("font-size: 28px; font-weight: 800; color: #89b4fa;")
        main_layout.addWidget(header)
        
        self.tabs = QTabWidget()
        self.tabs.addTab(self.create_skill_tab(), "🧠 My Skill Profile")
        self.tabs.addTab(self.create_library_tab(), "📚 Concept Library")
        self.tabs.addTab(self.create_mistakes_tab(), "⚠️ Mistake Review")
        
        main_layout.addWidget(self.tabs)

    def create_skill_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(20, 20, 20, 20)
        
        info = QLabel("AI Confidence Scores based on your assessment history.")
        info.setStyleSheet("color: #a6adc8; font-style: italic; margin-bottom: 20px;")
        layout.addWidget(info)
        
        self.skill_container = QFrame()
        self.skill_container.setStyleSheet("background-color: #262639; border-radius: 12px; padding: 20px;")
        self.skill_layout = QVBoxLayout(self.skill_container)
        self.skill_layout.setSpacing(20)
        
        layout.addWidget(self.skill_container)
        layout.addStretch()
        return tab

    def create_library_tab(self):
        tab = QWidget()
        layout = QHBoxLayout(tab)
        layout.setContentsMargins(0, 20, 0, 0)
        
        topic_frame = QFrame()
        topic_frame.setFixedWidth(250)
        topic_frame.setStyleSheet("background: #181825; border-radius: 8px;")
        t_layout = QVBoxLayout(topic_frame)
        
        self.topic_list = QListWidget()
        self.topic_list.setStyleSheet("border: none; background: transparent; color: #cdd6f4;")
        for category in KNOWLEDGE_DB.keys():
            self.topic_list.addItem(category)
        self.topic_list.currentItemChanged.connect(self.load_topic_content)
        
        t_layout.addWidget(self.topic_list)
        layout.addWidget(topic_frame)
        
        content_frame = QFrame()
        content_frame.setStyleSheet("background: #1e1e2e;")
        c_layout = QVBoxLayout(content_frame)
        
        self.lbl_topic = QLabel("Select Topic")
        self.lbl_topic.setStyleSheet("font-size: 24px; color: #fab387; font-weight: bold;")
        self.txt_content = QTextEdit()
        self.txt_content.setReadOnly(True)
        self.txt_content.setStyleSheet("background: #262639; border-radius: 8px; padding: 15px; font-size: 14px; color: #cdd6f4;")
        
        c_layout.addWidget(self.lbl_topic)
        c_layout.addWidget(self.txt_content)
        layout.addWidget(content_frame)
        return tab

    def create_mistakes_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        self.mistakes_area = QScrollArea()
        self.mistakes_area.setWidgetResizable(True)
        self.mistakes_area.setStyleSheet("border: none; background: transparent;")
        
        self.mistakes_container = QWidget()
        self.mistakes_layout = QVBoxLayout(self.mistakes_container)
        self.mistakes_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        self.mistakes_area.setWidget(self.mistakes_container)
        layout.addWidget(self.mistakes_area)
        return tab

    def refresh_data(self):
        self.load_skills()
        self.load_mistakes()

    def load_skills(self):
        while self.skill_layout.count():
            child = self.skill_layout.takeAt(0)
            if child.widget(): child.widget().deleteLater()
            
        user = self.user_manager.get_current_user()
        if not user: return
        
        profile = self.db.get_user_ml_profile(user['id'])
        skills = [
            ("Training Literacy", profile.get('training_literacy_index', 0.5)),
            ("Recovery IQ", profile.get('recovery_knowledge', 0.5)),
            ("Technique", profile.get('technique_score', 0.5))
        ]
        
        for name, score in skills:
            self.add_skill_bar(name, score)

    def add_skill_bar(self, name, score):
        pct = int(score * 100)
        lbl = QLabel(f"{name}: {pct}%")
        lbl.setStyleSheet("color: white; font-weight: bold;")
        bar = QProgressBar()
        bar.setValue(pct)
        bar.setStyleSheet(f"QProgressBar::chunk {{ background-color: {'#a6e3a1' if pct > 70 else '#f9e2af'}; }}")
        self.skill_layout.addWidget(lbl)
        self.skill_layout.addWidget(bar)

    def load_topic_content(self, item):
        if not item: return
        cat = item.text()
        self.lbl_topic.setText(cat)
        html = ""
        for term, desc in KNOWLEDGE_DB.get(cat, {}).items():
            html += f"<h3 style='color: #89b4fa;'>{term}</h3><p>{desc}</p><br>"
        self.txt_content.setHtml(html)

    def load_mistakes(self):
        while self.mistakes_layout.count():
            child = self.mistakes_layout.takeAt(0)
            if child.widget(): child.widget().deleteLater()
            
        user = self.user_manager.get_current_user()
        if not user: return
        
        assessments = self.db.get_user_assessments(user['id'])
        found = False
        for a in assessments:
            answers = self.db.get_assessment_answers(a['id'])
            for ans in answers:
                if not ans['is_correct']:
                    self.add_mistake_card(ans)
                    found = True
        
        if not found:
            self.mistakes_layout.addWidget(QLabel("No mistakes found. Good job!"))

    def add_mistake_card(self, answer_data):
        card = QFrame()
        card.setStyleSheet("background: #262639; border-radius: 8px; border-left: 4px solid #f38ba8;")
        layout = QVBoxLayout(card)
        
        q_text = answer_data.get('question_text') or f"Question ID: {answer_data.get('question_id')}"
        layout.addWidget(QLabel(f"<b>Question:</b> {q_text}"))
        
        hbox = QHBoxLayout()
        wrong = QLabel(f"❌ You: {answer_data['user_answer']}")
        wrong.setStyleSheet("color: #f38ba8;")
        right = QLabel(f"✅ Correct: {answer_data['correct_answer']}")
        right.setStyleSheet("color: #a6e3a1;")
        
        hbox.addWidget(wrong)
        hbox.addWidget(right)
        layout.addLayout(hbox)
        
        self.mistakes_layout.addWidget(card)