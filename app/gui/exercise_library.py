"""
Scientific Exercise Library v3.3 (STABLE FIX)
- Fix: Converts Database Row to Dict to prevent 'AttributeError: get' crash
- Fix: High Contrast Text
- Layout: Stat Cards
"""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QListWidget, QFrame,
    QPushButton, QTextEdit, QLineEdit, QComboBox, QCheckBox, QStackedWidget,
    QGridLayout, QGroupBox, QMessageBox, QListWidgetItem, QProgressBar,
    QSizePolicy
)
from PyQt6.QtCore import Qt, pyqtSignal

class ExerciseLibraryDialog(QDialog):
    exercise_selected = pyqtSignal(int, str, bool) 

    def __init__(self, db_manager, parent=None):
        super().__init__(parent)
        self.db = db_manager
        self.setWindowTitle("Scientific Hypertrophy Database")
        self.setMinimumSize(1100, 700)
        self.setStyleSheet("background-color: #11111b; color: #cdd6f4; font-family: 'Segoe UI', sans-serif;")
        
        self.init_ui()
        self.load_exercises("All")

    def init_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # --- LEFT SIDEBAR (Filter) ---
        sidebar = QFrame()
        sidebar.setFixedWidth(220)
        sidebar.setStyleSheet("background-color: #181825; border-right: 1px solid #313244;")
        side_layout = QVBoxLayout(sidebar)
        side_layout.setSpacing(10)
        side_layout.setContentsMargins(15, 20, 15, 20)
        
        side_layout.addWidget(self.create_label("MUSCLE GROUP", "#fab387", 12))
        
        self.cat_list = QListWidget()
        self.cat_list.setStyleSheet("""
            QListWidget { background: transparent; border: none; outline: none; }
            QListWidget::item { padding: 12px; border-radius: 6px; margin-bottom: 4px; color: #a6adc8; font-weight: 600; }
            QListWidget::item:selected { background-color: #313244; color: #ffffff; border-left: 3px solid #89b4fa; }
            QListWidget::item:hover { background-color: #313244; color: white; }
        """)
        categories = ["All", "Chest", "Back", "Quads", "Hamstrings", "Glutes", "Shoulders", "Triceps", "Biceps"]
        self.cat_list.addItems(categories)
        self.cat_list.setCurrentRow(0)
        self.cat_list.currentItemChanged.connect(self.filter_exercises)
        side_layout.addWidget(self.cat_list)
        
        new_btn = QPushButton("+ New Entry")
        new_btn.setStyleSheet("""
            QPushButton {
                background-color: #a6e3a1; color: #1e1e2e; font-weight: 800; 
                padding: 12px; border-radius: 6px; text-align: center; border: none;
            }
            QPushButton:hover { background-color: #94e2d5; }
        """)
        new_btn.clicked.connect(self.show_create_form)
        side_layout.addWidget(new_btn)
        
        main_layout.addWidget(sidebar)
        
        # --- MIDDLE (List) ---
        list_frame = QFrame()
        list_frame.setFixedWidth(300)
        list_frame.setStyleSheet("background-color: #1e1e2e; border-right: 1px solid #313244;")
        list_layout = QVBoxLayout(list_frame)
        list_layout.setContentsMargins(0, 0, 0, 0)
        
        header_container = QFrame()
        header_container.setStyleSheet("background-color: #181825; border-bottom: 1px solid #313244;")
        h_layout = QVBoxLayout(header_container)
        h_layout.setContentsMargins(15, 20, 15, 20)
        h_layout.addWidget(self.create_label("EXERCISE INDEX", "#89b4fa", 12))
        list_layout.addWidget(header_container)
        
        self.ex_list = QListWidget()
        self.ex_list.setStyleSheet("""
            QListWidget { background: transparent; border: none; outline: none; }
            QListWidget::item { padding: 15px; border-bottom: 1px solid #313244; color: #cdd6f4; font-size: 14px; }
            QListWidget::item:selected { background-color: #313244; color: #89b4fa; font-weight: bold; }
            QListWidget::item:hover { background-color: #262639; }
        """)
        self.ex_list.currentItemChanged.connect(self.display_details)
        list_layout.addWidget(self.ex_list)
        
        main_layout.addWidget(list_frame)
        
        # --- RIGHT (Details) ---
        self.stack = QStackedWidget()
        self.stack.setStyleSheet("background-color: #1e1e2e;")
        
        self.details_page = QFrame()
        self.setup_details_view(self.details_page)
        self.stack.addWidget(self.details_page)
        
        self.create_page = QFrame()
        self.setup_create_form(self.create_page)
        self.stack.addWidget(self.create_page)
        
        main_layout.addWidget(self.stack)

    def setup_details_view(self, parent):
        layout = QVBoxLayout(parent)
        layout.setSpacing(20)
        layout.setContentsMargins(40, 40, 40, 40)
        
        # 1. Header Area
        self.lbl_name = QLabel("Select an Exercise")
        self.lbl_name.setStyleSheet("font-size: 32px; font-weight: 900; color: #ffffff;")
        self.lbl_name.setWordWrap(True)
        layout.addWidget(self.lbl_name)
        
        # Tags
        self.tags_layout = QHBoxLayout()
        layout.addLayout(self.tags_layout)
        
        layout.addSpacing(10)
        
        # 2. STAT CARDS
        stats_layout = QHBoxLayout()
        stats_layout.setSpacing(15)
        
        self.card_stability = self.create_stat_card("STABILITY", "0/10", "#a6e3a1")
        self.card_profile = self.create_stat_card("PROFILE", "-", "#fab387")
        self.card_bias = self.create_stat_card("BIAS", "-", "#cba6f7")
        
        stats_layout.addWidget(self.card_stability)
        stats_layout.addWidget(self.card_profile)
        stats_layout.addWidget(self.card_bias)
        
        layout.addLayout(stats_layout)
        
        # 3. Description
        layout.addWidget(self.create_label("NEUROMECHANICAL ANALYSIS", "#89b4fa", 13))
        
        self.txt_desc = QTextEdit()
        self.txt_desc.setReadOnly(True)
        self.txt_desc.setStyleSheet("""
            QTextEdit { 
                background-color: #181825; 
                border: 1px solid #313244; 
                border-radius: 8px; 
                padding: 20px; 
                font-size: 15px; 
                line-height: 1.6; 
                color: #cdd6f4; 
            }
        """)
        layout.addWidget(self.txt_desc)
        
        # 4. Action Button
        self.btn_select = QPushButton("Use This Exercise")
        self.btn_select.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_select.setStyleSheet("""
            QPushButton {
                background-color: #89b4fa; color: #1e1e2e; 
                font-weight: 800; font-size: 16px; padding: 16px; border-radius: 8px; border: none;
            }
            QPushButton:hover { background-color: #b4befe; }
        """)
        self.btn_select.clicked.connect(self.select_current_exercise)
        layout.addWidget(self.btn_select)

    def create_stat_card(self, title, value, color):
        frame = QFrame()
        frame.setStyleSheet(f"""
            QFrame {{
                background-color: #181825;
                border: 1px solid #313244;
                border-radius: 12px;
            }}
        """)
        frame.setFixedHeight(90)
        
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(5)
        
        lbl_title = QLabel(title)
        lbl_title.setStyleSheet("color: #6c7086; font-weight: bold; font-size: 11px; letter-spacing: 1px;")
        
        lbl_val = QLabel(value)
        lbl_val.setStyleSheet(f"color: {color}; font-weight: 900; font-size: 18px;")
        
        layout.addWidget(lbl_title)
        layout.addWidget(lbl_val)
        
        # Return frame and keep ref to value label
        frame.val_label = lbl_val
        return frame

    def setup_create_form(self, parent):
        layout = QVBoxLayout(parent)
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(20)
        
        title = QLabel("ADD NEW EXERCISE")
        title.setStyleSheet("font-size: 24px; font-weight: 900; color: #a6e3a1;")
        layout.addWidget(title)
        
        self.inp_name = QLineEdit()
        self.inp_name.setPlaceholderText("Exercise Name")
        layout.addWidget(self.create_label("NAME", "#ffffff"))
        layout.addWidget(self.inp_name)
        
        grid = QGridLayout()
        self.inp_muscle = QComboBox()
        self.inp_muscle.addItems(["Chest", "Back", "Quads", "Hamstrings", "Shoulders", "Arms", "Calves"])
        grid.addWidget(self.create_label("MUSCLE", "#ffffff"), 0, 0)
        grid.addWidget(self.inp_muscle, 1, 0)
        
        self.inp_type = QComboBox()
        self.inp_type.addItems(["Compound", "Isolation"])
        grid.addWidget(self.create_label("TYPE", "#ffffff"), 0, 1)
        grid.addWidget(self.inp_type, 1, 1)
        layout.addLayout(grid)
        
        self.inp_uni = QCheckBox("Allows Unilateral Loading?")
        self.inp_uni.setStyleSheet("color: white; font-weight: bold; margin-top: 10px;")
        layout.addWidget(self.inp_uni)
        
        layout.addWidget(self.create_label("DESCRIPTION / NOTES", "#ffffff"))
        self.inp_desc = QTextEdit()
        self.inp_desc.setPlaceholderText("Enter scientific details...")
        layout.addWidget(self.inp_desc)
        
        btn_row = QHBoxLayout()
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(lambda: self.stack.setCurrentIndex(0))
        btn_cancel.setStyleSheet("background-color: #313244; color: white; padding: 12px; border-radius: 6px;")
        
        btn_save = QPushButton("Save to Library")
        btn_save.clicked.connect(self.save_new_exercise)
        btn_save.setStyleSheet("background-color: #a6e3a1; color: #1e1e2e; font-weight: bold; padding: 12px; border-radius: 6px;")
        
        btn_row.addWidget(btn_cancel)
        btn_row.addWidget(btn_save)
        layout.addLayout(btn_row)

    # --- HELPERS ---
    def create_label(self, text, color, size=12):
        lbl = QLabel(text)
        lbl.setStyleSheet(f"color: {color}; font-weight: bold; font-size: {size}px; letter-spacing: 1px;")
        return lbl

    def clear_layout(self, layout):
        while layout.count():
            child = layout.takeAt(0)
            if child.widget(): child.widget().deleteLater()

    # --- LOGIC ---
    def load_exercises(self, category):
        self.ex_list.clear()
        cursor = self.db.conn.cursor()
        
        if category == "All":
            query = "SELECT id, name FROM exercises ORDER BY name"
            params = ()
        else:
            query = "SELECT id, name FROM exercises WHERE muscle_group_primary LIKE ? ORDER BY name"
            params = (f"%{category}%",)
            
        rows = cursor.execute(query, params).fetchall()
        for row in rows:
            item = QListWidgetItem(row['name'])
            item.setData(Qt.ItemDataRole.UserRole, row['id'])
            self.ex_list.addItem(item)

    def filter_exercises(self, item):
        if item: self.load_exercises(item.text())

    def display_details(self, item):
        if not item: return
        self.stack.setCurrentIndex(0)
        ex_id = item.data(Qt.ItemDataRole.UserRole)
        
        # --- THE FIX: Convert Row to Dict ---
        row_obj = self.db.get_exercise_by_id(ex_id)
        if not row_obj: return
        ex = dict(row_obj) # Convert SQLite Row to Dict!
        
        self.lbl_name.setText(ex['name'])
        
        # Tags
        self.clear_layout(self.tags_layout)
        tags = []
        if ex['is_compound']: tags.append(("COMPOUND", "#f38ba8"))
        else: tags.append(("ISOLATION", "#89dceb"))
        if ex.get('is_unilateral'): tags.append(("UNILATERAL", "#f9e2af"))
        
        for text, color in tags:
            lbl = QLabel(f"  {text}  ")
            lbl.setStyleSheet(f"color: {color}; border: 1px solid {color}; border-radius: 12px; font-weight: bold; font-size: 10px; padding: 4px 8px;")
            self.tags_layout.addWidget(lbl)
        self.tags_layout.addStretch()
        
        # Safe Get
        stab = ex.get('stability_score')
        self.card_stability.val_label.setText(f"{stab}/10" if stab else "N/A")
        
        prof = ex.get('resistance_profile')
        self.card_profile.val_label.setText(prof if prof else "Standard")
        
        reg = ex.get('regional_bias')
        self.card_bias.val_label.setText(reg if reg else "General")
        
        desc = ex.get('instructions') or "<i>No advanced scientific data available for this exercise.</i>"
        self.txt_desc.setHtml(desc)

    def show_create_form(self):
        self.stack.setCurrentIndex(1)
        self.inp_name.clear()
        self.inp_desc.clear()

    def save_new_exercise(self):
        name = self.inp_name.text()
        if not name: return
        try:
            cursor = self.db.conn.cursor()
            cursor.execute("""
                INSERT INTO exercises (name, category, muscle_group_primary, equipment, difficulty_level, is_compound, instructions, is_unilateral)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (name, "Custom", self.inp_muscle.currentText(), "N/A", "N/A", 
                  "Compound" in self.inp_type.currentText(), self.inp_desc.toPlainText(), self.inp_uni.isChecked()))
            self.db.conn.commit()
            QMessageBox.information(self, "Success", "Exercise added!")
            self.load_exercises("All")
            self.stack.setCurrentIndex(0)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def select_current_exercise(self):
        item = self.ex_list.currentItem()
        if not item: return
        ex_id = item.data(Qt.ItemDataRole.UserRole)
        name = item.text()
        
        # Safety fetch
        cursor = self.db.conn.cursor()
        row = cursor.execute("SELECT is_unilateral FROM exercises WHERE id=?", (ex_id,)).fetchone()
        is_uni = False
        if row:
            d = dict(row)
            is_uni = bool(d.get('is_unilateral'))
        
        self.exercise_selected.emit(ex_id, name, is_uni)
        self.accept()