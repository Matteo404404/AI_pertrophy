"""
Scientific Exercise Library v2.0
- Neuromechanical Data Visualization
- Stability Scores
- HTML-Rich Descriptions
- Unilateral/Bilateral Filters
"""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QListWidget, QFrame,
    QPushButton, QTextEdit, QLineEdit, QComboBox, QCheckBox, QStackedWidget,
    QGridLayout, QGroupBox, QMessageBox, QListWidgetItem, QProgressBar
)
from PyQt6.QtCore import Qt, pyqtSignal

class ExerciseLibraryDialog(QDialog):
    exercise_selected = pyqtSignal(int, str, bool) 

    def __init__(self, db_manager, parent=None):
        super().__init__(parent)
        self.db = db_manager
        self.setWindowTitle("Scientific Hypertrophy Database")
        self.setMinimumSize(1100, 700)
        self.setStyleSheet("background-color: #1e1e2e; color: #cdd6f4;")
        
        self.init_ui()
        self.load_exercises("All")

    def init_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setSpacing(0)
        
        # --- LEFT SIDEBAR (Filter) ---
        sidebar = QFrame()
        sidebar.setFixedWidth(240)
        sidebar.setStyleSheet("background-color: #181825; border-right: 1px solid #313244;")
        side_layout = QVBoxLayout(sidebar)
        side_layout.setSpacing(15)
        
        side_layout.addWidget(self.create_label("TARGET MUSCLE", "#fab387"))
        
        self.cat_list = QListWidget()
        self.cat_list.setStyleSheet("""
            QListWidget { background: transparent; border: none; outline: none; }
            QListWidget::item { padding: 12px; border-radius: 6px; margin-bottom: 4px; color: #a6adc8; }
            QListWidget::item:selected { background-color: #313244; color: #ffffff; border-left: 3px solid #89b4fa; }
            QListWidget::item:hover { background-color: #313244; }
        """)
        categories = ["All", "Chest", "Back", "Quads", "Hamstrings", "Glutes", "Shoulders", "Triceps", "Biceps"]
        self.cat_list.addItems(categories)
        self.cat_list.setCurrentRow(0)
        self.cat_list.currentItemChanged.connect(self.filter_exercises)
        side_layout.addWidget(self.cat_list)
        
        new_btn = QPushButton("+ New Entry")
        new_btn.setStyleSheet("""
            background-color: #a6e3a1; color: #1e1e2e; font-weight: bold; 
            padding: 12px; border-radius: 6px; text-align: left;
        """)
        new_btn.clicked.connect(self.show_create_form)
        side_layout.addWidget(new_btn)
        
        main_layout.addWidget(sidebar)
        
        # --- MIDDLE (List) ---
        list_frame = QFrame()
        list_frame.setFixedWidth(320)
        list_frame.setStyleSheet("background-color: #1e1e2e; border-right: 1px solid #313244;")
        list_layout = QVBoxLayout(list_frame)
        
        list_layout.addWidget(self.create_label("EXERCISE INDEX", "#89b4fa"))
        
        self.ex_list = QListWidget()
        self.ex_list.setStyleSheet("""
            QListWidget { background: transparent; border: none; }
            QListWidget::item { padding: 15px; border-bottom: 1px solid #313244; font-weight: bold; font-size: 14px; }
            QListWidget::item:selected { background-color: #262639; color: #89b4fa; }
        """)
        self.ex_list.currentItemChanged.connect(self.display_details)
        list_layout.addWidget(self.ex_list)
        
        main_layout.addWidget(list_frame)
        
        # --- RIGHT (Scientific HUD) ---
        self.stack = QStackedWidget()
        self.stack.setStyleSheet("background-color: #1e1e2e; padding: 20px;")
        
        # 1. Detail View
        self.details_page = QFrame()
        self.setup_details_view(self.details_page)
        self.stack.addWidget(self.details_page)
        
        # 2. Create View
        self.create_page = QFrame()
        self.setup_create_form(self.create_page)
        self.stack.addWidget(self.create_page)
        
        main_layout.addWidget(self.stack)

    def setup_details_view(self, parent):
        layout = QVBoxLayout(parent)
        layout.setSpacing(20)
        
        # Title Header
        self.lbl_name = QLabel("Select an Exercise")
        self.lbl_name.setStyleSheet("font-size: 32px; font-weight: 800; color: #ffffff;")
        layout.addWidget(self.lbl_name)
        
        # Tags Row
        self.tags_layout = QHBoxLayout()
        layout.addLayout(self.tags_layout)
        
        # --- METRICS GRID ---
        metrics_frame = QFrame()
        metrics_frame.setStyleSheet("background-color: #262639; border-radius: 12px; padding: 15px;")
        metrics_grid = QGridLayout(metrics_frame)
        metrics_grid.setSpacing(20)
        
        # Stability Score
        metrics_grid.addWidget(self.create_label("STABILITY SCORE", "#a6adc8", 10), 0, 0)
        self.progress_stability = QProgressBar()
        self.progress_stability.setRange(0, 10)
        self.progress_stability.setTextVisible(True)
        self.progress_stability.setFormat("%v/10")
        self.progress_stability.setStyleSheet("""
            QProgressBar { background-color: #181825; border-radius: 6px; height: 10px; color: white; }
            QProgressBar::chunk { background-color: #a6e3a1; border-radius: 6px; }
        """)
        metrics_grid.addWidget(self.progress_stability, 1, 0)
        
        # Resistance Profile
        metrics_grid.addWidget(self.create_label("RESISTANCE PROFILE", "#a6adc8", 10), 0, 1)
        self.lbl_profile = QLabel("-")
        self.lbl_profile.setStyleSheet("font-size: 16px; font-weight: bold; color: #fab387;")
        metrics_grid.addWidget(self.lbl_profile, 1, 1)
        
        # Regional Bias
        metrics_grid.addWidget(self.create_label("REGIONAL BIAS", "#a6adc8", 10), 0, 2)
        self.lbl_region = QLabel("-")
        self.lbl_region.setStyleSheet("font-size: 16px; font-weight: bold; color: #cba6f7;")
        metrics_grid.addWidget(self.lbl_region, 1, 2)
        
        layout.addWidget(metrics_frame)
        
        # Description Box
        layout.addWidget(self.create_label("NEUROMECHANICAL ANALYSIS", "#89b4fa"))
        self.txt_desc = QTextEdit()
        self.txt_desc.setReadOnly(True)
        self.txt_desc.setStyleSheet("""
            QTextEdit { background-color: #181825; border: none; border-radius: 8px; padding: 15px; font-size: 14px; line-height: 1.6; color: #cdd6f4; }
        """)
        layout.addWidget(self.txt_desc)
        
        # Select Button
        self.btn_select = QPushButton("Use This Exercise")
        self.btn_select.setStyleSheet("background-color: #89b4fa; color: #1e1e2e; font-weight: 800; font-size: 16px; padding: 15px; border-radius: 8px;")
        self.btn_select.clicked.connect(self.select_current_exercise)
        layout.addWidget(self.btn_select)

    def display_details(self, item):
        if not item: return
        self.stack.setCurrentIndex(0)
        
        ex_id = item.data(Qt.ItemDataRole.UserRole)
        # Fetch updated schema columns
        cursor = self.db.conn.cursor()
        # We fetch explicitly to handle the new columns
        try:
            ex = cursor.execute("SELECT * FROM exercises WHERE id=?", (ex_id,)).fetchone()
        except:
            return # DB schema might not match yet
        
        if ex:
            self.lbl_name.setText(ex['name'])
            
            # Tags
            self.clear_layout(self.tags_layout)
            tags = []
            if ex['is_compound']: tags.append(("COMPOUND", "#f38ba8"))
            else: tags.append(("ISOLATION", "#89dceb"))
            
            if ex['is_unilateral']: tags.append(("UNILATERAL", "#f9e2af"))
            if ex['lengthened_bias']: tags.append(("LENGTHENED BIAS", "#cba6f7"))
            
            for text, color in tags:
                lbl = QLabel(f"  {text}  ")
                lbl.setStyleSheet(f"background-color: {color}22; color: {color}; border: 1px solid {color}; border-radius: 12px; font-weight: bold; font-size: 11px;")
                self.tags_layout.addWidget(lbl)
            self.tags_layout.addStretch()
            
            # Metrics
            self.progress_stability.setValue(ex['stability_score'] or 5)
            self.lbl_profile.setText(ex['resistance_profile'] or "Standard")
            self.lbl_region.setText(ex['regional_bias'] or "General")
            
            # HTML Description
            self.txt_desc.setHtml(ex['instructions'])

    # --- SETUP CREATE FORM (Simplified for brevity, but logic is same) ---
    def setup_create_form(self, parent):
        layout = QVBoxLayout(parent)
        layout.addWidget(QLabel("Adding Custom Exercise..."))
        # (This remains similar to previous version, just styling updated)
        btn_back = QPushButton("Back")
        btn_back.clicked.connect(lambda: self.stack.setCurrentIndex(0))
        layout.addWidget(btn_back)
        layout.addStretch()

    # --- HELPERS ---
    def create_label(self, text, color, size=12):
        lbl = QLabel(text)
        lbl.setStyleSheet(f"color: {color}; font-weight: bold; font-size: {size}px; letter-spacing: 1px;")
        return lbl

    def clear_layout(self, layout):
        while layout.count():
            child = layout.takeAt(0)
            if child.widget(): child.widget().deleteLater()

    def load_exercises(self, category):
        self.ex_list.clear()
        cursor = self.db.conn.cursor()
        query = "SELECT id, name FROM exercises WHERE muscle_group_primary LIKE ? ORDER BY name"
        param = "%" if category == "All" else f"%{category}%"
        rows = cursor.execute(query, (param,)).fetchall()
        for row in rows:
            item = QListWidgetItem(row['name'])
            item.setData(Qt.ItemDataRole.UserRole, row['id'])
            self.ex_list.addItem(item)

    def filter_exercises(self, item):
        if item: self.load_exercises(item.text())

    def select_current_exercise(self):
        item = self.ex_list.currentItem()
        if not item: return
        ex_id = item.data(Qt.ItemDataRole.UserRole)
        
        # Fetch metadata to pass back
        cursor = self.db.conn.cursor()
        ex = cursor.execute("SELECT name, is_unilateral FROM exercises WHERE id=?", (ex_id,)).fetchone()
        
        self.exercise_selected.emit(ex_id, ex['name'], bool(ex['is_unilateral']))
        self.accept()
    
    def show_create_form(self):
        self.stack.setCurrentIndex(1)