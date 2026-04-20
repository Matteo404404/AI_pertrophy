"""
Scientific Hypertrophy Trainer - Learning Center v5.0 (RAG CHATBOT ENABLED)
- Integrates On-Demand Ollama AI Coaching for mistake reviews.
- NEW: Full Retrieval-Augmented Generation (RAG) Chatbot interface.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QListWidget, QFrame,
    QPushButton, QTextEdit, QLineEdit, QScrollArea, QTabWidget, QProgressBar
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QCursor
import requests
import json

# --- The Scientific Knowledge Base ---
KNOWLEDGE_DB = {
    "Mechanical Tension (Primary Driver)": {
        "Mechanical Tension": "Mechanical tension is force produced by cross-bridge cycling under load. The INTENT to move the weight explosively is what maximally recruits motor units. Grinder reps at RIR 0 produce dramatically more tension.",
        "Motor Units": "Per Henneman's Size Principle, the largest motor units (Type II) are only fully recruited under high force demands: heavy loads or lighter loads taken very close to failure.",
        "Muscle Damage": "Damas et al. (2015) showed muscle damage is a byproduct of novelty, not a cause of growth. Chasing soreness wastes recovery resources.",
    },
    "Volume & Intensity": {
        "Effective Reps": "Only the last ~5 reps before muscular failure fully recruit high-threshold motor units. These are 'effective reps'.",
        "Volume": "Schoenfeld (2017) showed volume is logarithmic. 2-3 hard sets per exercise per session captures the majority of the stimulus. Doing 5+ sets accumulates Junk Volume.",
        "RIR": "0-2 RIR for isolation exercises. 1-3 RIR for compounds.",
    },
    "Biomechanics": {
        "Resistance Profiles": "Choose exercises where the resistance is highest WHERE THE MUSCLE IS STRONGEST. Cables provide constant tension.",
        "Stretch-Mediated Hype": "Training at extreme stretch adds disproportionate muscle DAMAGE. Full ROM is optimal. Do not chase extreme stretches at the cost of force production.",
        "Active Insufficiency": "A biarticular muscle cannot produce maximal force when shortened across both joints. Seated leg curls allow hamstrings to produce MORE FORCE than lying leg curls.",
    }
}

# --- AI Threads ---
class AICoachWorker(QThread):
    finished = pyqtSignal(str)
    def __init__(self, question, user_ans, correct_ans):
        super().__init__()
        self.question, self.user_ans, self.correct_ans = question, user_ans, correct_ans

    def run(self):
        prompt = (f"Act as an expert hypertrophy coach. The user was asked: '{self.question}'. "
                  f"They answered: '{self.user_ans}', but the correct answer is '{self.correct_ans}'. "
                  "In exactly 2 supportive, educational sentences, explain why their answer is wrong "
                  "and clarify the correct concept. Keep it concise.")
        try:
            resp = requests.post("http://localhost:11434/api/generate", json={"model": "qwen3:1.7b", "prompt": prompt, "stream": False}, timeout=45)
            self.finished.emit(resp.json().get("response", "").strip() if resp.status_code == 200 else "")
        except: self.finished.emit("")

class RAGChatWorker(QThread):
    finished = pyqtSignal(str)
    def __init__(self, user_message):
        super().__init__()
        self.user_message = user_message

    def run(self):
        # Build RAG Context
        context = " ".join([f"{k}: {v}" for cat in KNOWLEDGE_DB.values() for k, v in cat.items()])
        prompt = (
            "You are an elite, science-based hypertrophy AI Coach. "
            "Use the following scientific principles to answer the user. Do not invent bro-science. "
            f"PRINCIPLES: {context}\n\n"
            f"USER QUESTION: {self.user_message}\n"
            "COACH ANSWER:"
        )
        try:
            resp = requests.post("http://localhost:11434/api/generate", json={"model": "qwen3:1.7b", "prompt": prompt, "stream": False}, timeout=60)
            self.finished.emit(resp.json().get("response", "").strip() if resp.status_code == 200 else "Connection Error.")
        except: self.finished.emit("Failed to connect to AI Engine.")

class LearningWidget(QWidget):
    def __init__(self, db_manager, user_manager):
        super().__init__()
        self.db = db_manager
        self.user_manager = user_manager
        self.workers =[]
        self.init_ui()
    
    def init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(30, 30, 30, 30)
        
        header = QLabel("Knowledge Base & AI Coach")
        header.setStyleSheet("font-size: 28px; font-weight: 900; color: #89b4fa;")
        main_layout.addWidget(header)
        
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabWidget::pane { border: none; background: transparent; }
            QTabBar::tab { background: #181825; padding: 12px 24px; border-radius: 6px; color: #a6adc8; font-weight: bold; margin-right: 5px; border: 1px solid #313244;}
            QTabBar::tab:selected { background: #313244; color: #a6e3a1; border-bottom: 3px solid #a6e3a1;}
        """)
        self.tabs.addTab(self.create_skill_tab(), "🧠 My Skill Profile")
        self.tabs.addTab(self.create_library_tab(), "📚 Concept Library")
        self.tabs.addTab(self.create_mistakes_tab(), "⚠️ Mistake Review")
        self.tabs.addTab(self.create_chat_tab(), "💬 Chat with AI Coach")
        
        main_layout.addWidget(self.tabs)

    # --- TAB 1: SKILLS ---
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

    # --- TAB 2: LIBRARY ---
    def create_library_tab(self):
        tab = QWidget()
        layout = QHBoxLayout(tab)
        layout.setContentsMargins(0, 20, 0, 0)
        
        topic_frame = QFrame()
        topic_frame.setFixedWidth(280)
        topic_frame.setStyleSheet("background: #181825; border-radius: 8px; border: 1px solid #313244;")
        t_layout = QVBoxLayout(topic_frame)
        self.topic_list = QListWidget()
        self.topic_list.setStyleSheet("""
            QListWidget { background: transparent; border: none; outline: none; }
            QListWidget::item { padding: 12px; color: #a6adc8; border-bottom: 1px solid #313244; border-radius: 4px; margin-bottom: 2px;}
            QListWidget::item:selected { background: #313244; color: #89b4fa; font-weight: bold; border-left: 3px solid #89b4fa;}
        """)
        for category in KNOWLEDGE_DB.keys(): self.topic_list.addItem(category)
        self.topic_list.currentItemChanged.connect(self.load_topic_content)
        t_layout.addWidget(self.topic_list)
        layout.addWidget(topic_frame)
        
        content_frame = QFrame()
        content_frame.setStyleSheet("background: #1e1e2e; border-radius: 8px; border: 1px solid #313244;")
        c_layout = QVBoxLayout(content_frame)
        self.lbl_topic = QLabel("Select Topic")
        self.lbl_topic.setStyleSheet("font-size: 24px; color: #fab387; font-weight: bold; border: none;")
        self.txt_content = QTextEdit()
        self.txt_content.setReadOnly(True)
        self.txt_content.setStyleSheet("background: #262639; border-radius: 8px; padding: 20px; font-size: 15px; color: #cdd6f4; line-height: 1.6; border: none;")
        c_layout.addWidget(self.lbl_topic)
        c_layout.addWidget(self.txt_content)
        layout.addWidget(content_frame)
        return tab

    # --- TAB 3: MISTAKES ---
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

    # --- TAB 4: RAG CHATBOT ---
    def create_chat_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setStyleSheet("background: #181825; border: 1px solid #313244; border-radius: 8px; padding: 15px; font-size: 15px; color: #cdd6f4;")
        self.chat_display.append("<b style='color:#89b4fa;'>🤖 AI Coach:</b> What's on your mind today, athlete?")
        layout.addWidget(self.chat_display, 1)
        
        input_row = QHBoxLayout()
        self.chat_input = QLineEdit()
        self.chat_input.setPlaceholderText("Ask a question about hypertrophy, volume, or exercise selection...")
        self.chat_input.setStyleSheet("background: #262639; padding: 15px; border: 1px solid #313244; border-radius: 8px; font-size: 15px; color: white;")
        self.chat_input.returnPressed.connect(self.send_chat_message)
        input_row.addWidget(self.chat_input, 1)
        
        btn_send = QPushButton("Send")
        btn_send.setStyleSheet("background-color: #a6e3a1; color: #1e1e2e; font-weight: bold; padding: 15px 30px; border-radius: 8px; font-size: 14px;")
        btn_send.clicked.connect(self.send_chat_message)
        input_row.addWidget(btn_send)
        
        layout.addLayout(input_row)
        return tab

    def send_chat_message(self):
        msg = self.chat_input.text().strip()
        if not msg: return
        self.chat_input.clear()
        
        self.chat_display.append(f"<br><b style='color:#a6e3a1;'>👤 You:</b> {msg}")
        self.chat_display.append("<i style='color:#6c7086;'>🤖 Coach is thinking...</i>")
        
        worker = RAGChatWorker(msg)
        def on_reply(text):
            cursor = self.chat_display.textCursor()
            self.chat_display.setTextCursor(cursor)
            html = self.chat_display.toHtml()
            thinking_marker = "Coach is thinking..."
            if thinking_marker in html:
                idx = html.rfind(thinking_marker)
                start = html.rfind("<i", 0, idx)
                end = html.find("</i>", idx) + 4
                if start != -1 and end > 4:
                    html = html[:start] + html[end:]
                    self.chat_display.setHtml(html)
            self.chat_display.append(f"<b style='color:#89b4fa;'>🤖 AI Coach:</b> {text}")
        
        worker.finished.connect(on_reply)
        self.workers.append(worker)
        worker.start()

    # --- LOGIC & REFRESH ---
    def refresh_data(self):
        self.load_skills()
        self.load_mistakes()

    def load_skills(self):
        while self.skill_layout.count():
            child = self.skill_layout.takeAt(0)
            if child.widget(): child.widget().deleteLater()
            
        user = self.user_manager.get_current_user()
        if not user: return
        
        profile = self.user_manager.get_user_ml_profile(user['id'])
        skills =[("Training Literacy", profile.get('training_literacy_index', 0.5)), ("Recovery IQ", profile.get('recovery_knowledge', 0.5)), ("Load Management", profile.get('load_management_score', 0.5)), ("Technique", profile.get('technique_score', 0.5))]
        
        for name, score in skills:
            pct = int(score * 100)
            lbl = QLabel(f"{name}: {pct}%")
            lbl.setStyleSheet("color: white; font-weight: bold; border: none;")
            bar = QProgressBar()
            bar.setValue(pct)
            color = '#a6e3a1' if pct > 70 else '#f9e2af' if pct > 40 else '#f38ba8'
            bar.setStyleSheet(f"QProgressBar {{ background: #313244; height: 8px; border-radius: 4px; border: none; }} QProgressBar::chunk {{ background-color: {color}; border-radius: 4px; }}")
            self.skill_layout.addWidget(lbl)
            self.skill_layout.addWidget(bar)

    def load_topic_content(self, item):
        if not item: return
        cat = item.text()
        self.lbl_topic.setText(cat)
        html = ""
        for term, desc in KNOWLEDGE_DB.get(cat, {}).items():
            html += f"<h3 style='color: #89b4fa; margin-top: 15px;'>{term}</h3><p style='margin-bottom: 20px;'>{desc}</p>"
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
            for ans in self.db.get_assessment_answers(a['id']):
                if not ans['is_correct']:
                    self.add_mistake_card(ans)
                    found = True
        if not found: self.mistakes_layout.addWidget(QLabel("No mistakes found in your history. Excellent job!"))

    def add_mistake_card(self, answer_data):
        card = QFrame()
        card.setStyleSheet("background: #181825; border-radius: 12px; border: 1px solid #313244; border-left: 5px solid #f38ba8;")
        layout = QVBoxLayout(card)
        
        q_text = answer_data.get('question_text') or f"Question ID: {answer_data.get('question_id')}"
        lbl_q = QLabel(f"<b>Question:</b> {q_text}")
        lbl_q.setWordWrap(True)
        lbl_q.setStyleSheet("color: white; font-size: 15px; border: none;")
        layout.addWidget(lbl_q)
        
        hbox = QHBoxLayout()
        wrong = QLabel(f"❌ Your answer: {answer_data['user_answer']}")
        wrong.setStyleSheet("color: #f38ba8; border: none; font-size: 14px;")
        right = QLabel(f"✅ Correct: {answer_data['correct_answer']}")
        right.setStyleSheet("color: #a6e3a1; border: none; font-size: 14px;")
        hbox.addWidget(wrong)
        hbox.addWidget(right)
        layout.addLayout(hbox)
        
        ai_container = QWidget()
        ai_layout = QVBoxLayout(ai_container)
        ai_layout.setContentsMargins(0, 10, 0, 0)
        btn_ask_ai = QPushButton("🤖 Ask AI Coach to Explain")
        btn_ask_ai.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        btn_ask_ai.setStyleSheet("background-color: #313244; color: #89b4fa; font-weight: bold; padding: 10px; border-radius: 6px;")
        ai_layout.addWidget(btn_ask_ai)
        
        def on_ask_ai():
            btn_ask_ai.setText("⏳ AI Coach is thinking...")
            btn_ask_ai.setEnabled(False)
            worker = AICoachWorker(q_text, answer_data['user_answer'], answer_data['correct_answer'])
            def handle_result(text):
                try:
                    btn_ask_ai.hide()
                    result_lbl = QLabel(f"🤖 <b>AI Coach:</b> {text}" if text else f"💡 {answer_data.get('explanation','')}")
                    result_lbl.setWordWrap(True)
                    result_lbl.setStyleSheet("color: #89b4fa; font-style: italic; padding: 15px; background: #262639; border-radius: 8px; border: 1px solid #313244;")
                    ai_layout.addWidget(result_lbl)
                except RuntimeError: pass
            worker.finished.connect(handle_result)
            self.workers.append(worker)
            worker.start()
            
        btn_ask_ai.clicked.connect(on_ask_ai)
        layout.addWidget(ai_container)
        self.mistakes_layout.addWidget(card)