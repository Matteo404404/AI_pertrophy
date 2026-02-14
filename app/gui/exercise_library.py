"""
Scientific Exercise Library
- Browsing by Muscle Group
- Scientific Descriptions (Biometrics)
- Custom Exercise Creation
"""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QListWidget, QFrame,
    QPushButton, QTextEdit, QLineEdit, QComboBox, QCheckBox, QStackedWidget,
    QGridLayout, QGroupBox, QMessageBox
)
from PyQt6.QtCore import Qt, pyqtSignal

class ExerciseLibraryDialog(QDialog):
    # Signal: Returns ID and Name when an exercise is selected
    exercise_selected = pyqtSignal(int, str, bool) 

    def __init__(self, db_manager, parent=None):
        super().__init__(parent)
        self.db = db_manager
        self.setWindowTitle("Scientific Exercise Database")
        self.setMinimumSize(900, 600)
        self.setStyleSheet("background-color: #1e1e2e; color: #cdd6f4;")
        
        self.init_ui()
        self.load_exercises("All")

    def init_ui(self):
        main_layout = QHBoxLayout(self)
        
        # --- LEFT: Categories ---
        cat_frame = QFrame()
        cat_frame.setFixedWidth(200)
        cat_frame.setStyleSheet("background-color: #181825; border-right: 1px solid #313244;")
        cat_layout = QVBoxLayout(cat_frame)
        
        cat_label = QLabel("Muscle Groups")
        cat_label.setStyleSheet("font-weight: bold; color: #89b4fa; font-size: 14px;")
        cat_layout.addWidget(cat_label)
        
        self.cat_list = QListWidget()
        self.cat_list.setStyleSheet("border: none; font-size: 14px;")
        categories = ["All", "Chest", "Back", "Quads", "Hamstrings", "Glutes", "Shoulders", "Triceps", "Biceps", "Calves"]
        self.cat_list.addItems(categories)
        self.cat_list.setCurrentRow(0)
        self.cat_list.currentItemChanged.connect(self.filter_exercises)
        cat_layout.addWidget(self.cat_list)
        
        # Create New Button
        new_btn = QPushButton("+ Create Custom")
        new_btn.setStyleSheet("background-color: #a6e3a1; color: #1e1e2e; font-weight: bold; padding: 10px;")
        new_btn.clicked.connect(self.show_create_form)
        cat_layout.addWidget(new_btn)
        
        main_layout.addWidget(cat_frame)
        
        # --- CENTER: Exercise List ---
        list_frame = QFrame()
        list_frame.setFixedWidth(300)
        list_layout = QVBoxLayout(list_frame)
        
        self.ex_list = QListWidget()
        self.ex_list.setStyleSheet("background-color: #262639; border-radius: 8px;")
        self.ex_list.currentItemChanged.connect(self.display_details)
        list_layout.addWidget(self.ex_list)
        
        select_btn = QPushButton("Use This Exercise")
        select_btn.setStyleSheet("background-color: #89b4fa; color: #1e1e2e; font-weight: bold; padding: 12px;")
        select_btn.clicked.connect(self.select_current_exercise)
        list_layout.addWidget(select_btn)
        
        main_layout.addWidget(list_frame)
        
        # --- RIGHT: Details / Create Form (Stacked) ---
        self.stack = QStackedWidget()
        
        # Page 1: Details View
        self.details_page = QFrame()
        details_layout = QVBoxLayout(self.details_page)
        
        self.lbl_name = QLabel("Select an Exercise")
        self.lbl_name.setStyleSheet("font-size: 24px; font-weight: bold; color: #fab387;")
        details_layout.addWidget(self.lbl_name)
        
        self.lbl_tags = QLabel("")
        self.lbl_tags.setStyleSheet("color: #a6adc8; font-style: italic;")
        details_layout.addWidget(self.lbl_tags)
        
        details_layout.addWidget(QLabel("Biomechanics & Science:"))
        self.txt_desc = QTextEdit()
        self.txt_desc.setReadOnly(True)
        self.txt_desc.setStyleSheet("background-color: #181825; border: 1px solid #313244; font-size: 14px; line-height: 1.4;")
        details_layout.addWidget(self.txt_desc)
        
        self.stack.addWidget(self.details_page)
        
        # Page 2: Creation Form
        self.create_page = QFrame()
        self.setup_create_form(self.create_page)
        self.stack.addWidget(self.create_page)
        
        main_layout.addWidget(self.stack)

    def setup_create_form(self, parent):
        layout = QVBoxLayout(parent)
        layout.setSpacing(15)
        
        layout.addWidget(QLabel("🆕 Create Custom Exercise"))
        
        self.inp_name = QLineEdit()
        self.inp_name.setPlaceholderText("Exercise Name (e.g. Reverse Pec Deck)")
        layout.addWidget(QLabel("Name:"))
        layout.addWidget(self.inp_name)
        
        # Metadata Grid
        grid = QGridLayout()
        
        self.inp_muscle = QComboBox()
        self.inp_muscle.addItems(["Chest", "Back", "Quads", "Hamstrings", "Shoulders", "Arms"])
        grid.addWidget(QLabel("Target Muscle:"), 0, 0)
        grid.addWidget(self.inp_muscle, 0, 1)
        
        self.inp_type = QComboBox()
        self.inp_type.addItems(["Compound (Multi-Joint)", "Isolation (Single-Joint)"])
        grid.addWidget(QLabel("Type:"), 1, 0)
        grid.addWidget(self.inp_type, 1, 1)
        
        self.inp_uni = QCheckBox("Allows Unilateral Loading?")
        self.inp_uni.setToolTip("Can you do this with one arm/leg at a time?")
        grid.addWidget(self.inp_uni, 2, 0, 1, 2)
        
        layout.addLayout(grid)
        
        layout.addWidget(QLabel("Scientific / Technique Notes:"))
        self.inp_desc = QTextEdit()
        self.inp_desc.setPlaceholderText("e.g., 'Trains the muscle in the shortened position. Focus on peak contraction.'")
        layout.addWidget(self.inp_desc)
        
        btn_row = QHBoxLayout()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(lambda: self.stack.setCurrentIndex(0))
        
        save_btn = QPushButton("Save to Database")
        save_btn.setStyleSheet("background-color: #a6e3a1; color: #1e1e2e;")
        save_btn.clicked.connect(self.save_new_exercise)
        
        btn_row.addWidget(cancel_btn)
        btn_row.addWidget(save_btn)
        layout.addLayout(btn_row)
        layout.addStretch()

    # --- LOGIC ---
    
    def load_exercises(self, category):
        self.ex_list.clear()
        
        # Fetch from DB
        cursor = self.db.conn.cursor()
        if category == "All":
            query = "SELECT id, name, muscle_group_primary FROM exercises ORDER BY name"
            params = ()
        else:
            query = "SELECT id, name, muscle_group_primary FROM exercises WHERE muscle_group_primary LIKE ? ORDER BY name"
            params = (f"%{category}%",)
            
        rows = cursor.execute(query, params).fetchall()
        
        for row in rows:
            from PyQt6.QtWidgets import QListWidgetItem
            item = QListWidgetItem(row['name'])
            item.setData(Qt.ItemDataRole.UserRole, row['id'])
            self.ex_list.addItem(item)

    def filter_exercises(self, item):
        if item:
            self.load_exercises(item.text())

    def display_details(self, item):
        if not item: return
        self.stack.setCurrentIndex(0) # Switch to details view
        
        ex_id = item.data(Qt.ItemDataRole.UserRole)
        ex = self.db.get_exercise_by_id(ex_id)
        
        if ex:
            self.lbl_name.setText(ex['name'])
            
            # Format Tags
            tags = f"{ex['muscle_group_primary']} • {'Compound' if ex['is_compound'] else 'Isolation'}"
            if ex.get('is_unilateral'):
                tags += " • Unilateral Capable"
            self.lbl_tags.setText(tags)
            
            # Description
            desc = ex.get('instructions') or "No scientific data available for this exercise yet."
            self.txt_desc.setText(desc)

    def show_create_form(self):
        self.stack.setCurrentIndex(1)
        self.inp_name.clear()
        self.inp_desc.clear()

    def save_new_exercise(self):
        name = self.inp_name.text()
        if not name: return
        
        muscle = self.inp_muscle.currentText()
        is_compound = "Compound" in self.inp_type.currentText()
        is_uni = self.inp_uni.isChecked()
        desc = self.inp_desc.toPlainText()
        
        # Save to DB
        new_id = self.db.add_custom_exercise(
            name, "Custom", muscle, "N/A", "N/A", is_compound, desc, is_uni
        )
        
        if new_id:
            QMessageBox.information(self, "Success", "Exercise added to library!")
            self.load_exercises("All")
            self.stack.setCurrentIndex(0)
        else:
            QMessageBox.critical(self, "Error", "Failed to add exercise.")

    def select_current_exercise(self):
        item = self.ex_list.currentItem()
        if not item: return
        
        ex_id = item.data(Qt.ItemDataRole.UserRole)
        name = item.text()
        
        # Check unilateral capability from DB to auto-set checkbox in parent
        ex = self.db.get_exercise_by_id(ex_id)
        is_uni = ex.get('is_unilateral', False)
        
        self.exercise_selected.emit(ex_id, name, bool(is_uni))
        self.accept()