"""
Scientific Hypertrophy Trainer - Tracking Interface
- Complete UI Overhaul (Horizontal Date Strip with Activity Dots)
- Full Workout AI Session Analyzer
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame, 
    QTabWidget, QDoubleSpinBox, QSpinBox, QTextEdit,
    QComboBox, QScrollArea, QMessageBox, QTableWidget, 
    QTableWidgetItem, QHeaderView, QDialog, QGridLayout, QSlider,
    QInputDialog
)
from PyQt6.QtCore import Qt, QDate, pyqtSignal
from datetime import date
import json

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

try:
    from gui.exercise_library import ExerciseLibraryDialog
except ImportError:
    from app.gui.exercise_library import ExerciseLibraryDialog

try:
    from ml_engine.inference.hybrid_predictor import HybridPredictor
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

try:
    from gui.food_search_dialog import FoodSearchDialog
except ImportError:
    from app.gui.food_search_dialog import FoodSearchDialog

MICROS_CONFIG = {
    "Vitamins (Fat Soluble)":[("Vitamin A", "IU"), ("Vitamin D3", "IU"), ("Vitamin E", "mg"), ("Vitamin K", "mcg")],
    "Vitamins (Water Soluble)":[("Vitamin C", "mg"), ("B-Complex", "mg"), ("Folate", "mcg")],
    "Minerals":[("Magnesium", "mg"), ("Zinc", "mg"), ("Iron", "mg"), ("Calcium", "mg"), ("Sodium", "mg")],
    "Other":[("Water", "L"), ("Fiber", "g"), ("Omega-3", "g")]
}

class HorizontalWeekCalendar(QFrame):
    date_selected = pyqtSignal(QDate)

    def __init__(self):
        super().__init__()
        self.current_anchor = QDate.currentDate()
        self.selected_date = QDate.currentDate()
        self.activity_flags = {} 
        self.days_buttons =[]
        self.setStyleSheet("background: #181825; border-radius: 8px; border: 1px solid #313244;")
        self.setFixedHeight(105)
        self._build_ui()

    def _build_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 5, 10, 5)
        layout.setSpacing(10)

        btn_prev = QPushButton("◀")
        btn_prev.setFixedSize(40, 50)
        btn_prev.clicked.connect(lambda: self.shift_week(-7))
        btn_prev.setStyleSheet("background: #262639; color: white; border: none; border-radius: 6px; font-weight: bold;")
        layout.addWidget(btn_prev)

        self.days_layout = QHBoxLayout()
        for i in range(7):
            btn = QPushButton()
            btn.setFixedSize(85, 80)
            btn.clicked.connect(lambda checked, idx=i: self.select_day_index(idx))
            self.days_layout.addWidget(btn)
            self.days_buttons.append(btn)
        layout.addLayout(self.days_layout)

        btn_next = QPushButton("▶")
        btn_next.setFixedSize(40, 50)
        btn_next.clicked.connect(lambda: self.shift_week(7))
        btn_next.setStyleSheet("background: #262639; color: white; border: none; border-radius: 6px; font-weight: bold;")
        layout.addWidget(btn_next)

        btn_today = QPushButton("Today")
        btn_today.setFixedHeight(50)
        btn_today.clicked.connect(self.jump_to_today)
        btn_today.setStyleSheet("background: #89b4fa; color: #1e1e2e; font-weight: bold; border: none; border-radius: 6px; padding: 0 15px;")
        layout.addWidget(btn_today)

        self._update_buttons()

    def shift_week(self, days):
        self.current_anchor = self.current_anchor.addDays(days)
        self._update_buttons()

    def jump_to_today(self):
        self.current_anchor = QDate.currentDate()
        self.selected_date = QDate.currentDate()
        self._update_buttons()
        self.date_selected.emit(self.selected_date)

    def select_day_index(self, idx):
        start_date = self.current_anchor.addDays(-3)
        self.selected_date = start_date.addDays(idx)
        self._update_buttons()
        self.date_selected.emit(self.selected_date)

    def set_activity_flags(self, flags_dict):
        self.activity_flags = flags_dict
        self._update_buttons()

    def _update_buttons(self):
        start_date = self.current_anchor.addDays(-3)
        for i in range(7):
            date_obj = start_date.addDays(i)
            date_str = date_obj.toString("yyyy-MM-dd")
            btn = self.days_buttons[i]
            
            flags = self.activity_flags.get(date_str, [])
            icons =[]
            if "workout" in flags: icons.append("💪")
            if "diet" in flags: icons.append("🥑")
            if "sleep" in flags: icons.append("💤")
            
            icon_str = "".join(icons)
            day_str = f"{date_obj.toString('ddd')}\n{date_obj.toString('d')}"
            if icon_str:
                day_str += f"\n{icon_str}"
            else:
                day_str += "\n"
                
            btn.setText(day_str)
            
            if date_obj == self.selected_date:
                btn.setStyleSheet("background: #a6e3a1; color: #1e1e2e; font-weight: 900; font-size: 14px; border-radius: 8px; border: none;")
            elif date_obj == QDate.currentDate():
                btn.setStyleSheet("background: #313244; color: #89b4fa; font-weight: bold; font-size: 13px; border-radius: 8px; border: 1px solid #89b4fa;")
            else:
                btn.setStyleSheet("background: transparent; color: #a6adc8; font-weight: bold; font-size: 13px; border-radius: 8px; border: none;")

class TrackingWidget(QWidget):
    def __init__(self, db_manager, tracking_system, user_manager):
        super().__init__()
        self.db = db_manager
        self.tracking_system = tracking_system
        self.user_manager = user_manager
        
        self.selected_date = QDate.currentDate()
        self.micro_inputs = {}
        
        if ML_AVAILABLE:
            self.ai_predictor = HybridPredictor(self.db, "ml_engine/models/strength_predictor.pt")
        else:
            self.ai_predictor = None
            
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)
        
        header_row = QHBoxLayout()
        title = QLabel("Session Console")
        title.setStyleSheet("font-size: 24px; font-weight: 900; color: #89b4fa;")
        header_row.addWidget(title)
        
        self.date_info_label = QLabel(f"Active Date: {self.selected_date.toString('yyyy-MM-dd')}")
        self.date_info_label.setStyleSheet("font-weight: bold; color: #fab387; font-size: 16px; margin-left: 15px;")
        header_row.addWidget(self.date_info_label)
        header_row.addStretch()
        
        self.calendar_strip = HorizontalWeekCalendar()
        self.calendar_strip.date_selected.connect(self.on_date_selected)
        main_layout.addWidget(self.calendar_strip)
        
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabWidget::pane { border: none; background: transparent; }
            QTabBar::tab { background: #181825; padding: 12px 24px; border-radius: 6px; color: #a6adc8; font-weight: bold; margin-right: 5px; border: 1px solid #313244;}
            QTabBar::tab:selected { background: #313244; color: #a6e3a1; border-bottom: 3px solid #a6e3a1;}
        """)
        
        self.tabs.addTab(self.create_workout_tab(), "🏋️ Protocol Execution")
        self.tabs.addTab(self.create_diet_tab(), "🧪 Nutrition HUD")
        self.tabs.addTab(self.create_sleep_tab(), "😴 Recovery Log")
        
        main_layout.addWidget(self.tabs)
        self.refresh_templates()
        self.refresh_data()

    def create_workout_tab(self):
        tab = QWidget()
        layout = QHBoxLayout(tab)
        layout.setContentsMargins(0, 15, 0, 0)
        
        p_frame = QFrame()
        p_frame.setFixedWidth(280)
        p_frame.setStyleSheet("background: #181825; border-radius: 12px; border: 1px solid #313244;")
        p_layout = QVBoxLayout(p_frame)
        
        p_layout.addWidget(QLabel("<b>LOAD PROTOCOL</b>"))
        self.combo_templates = QComboBox()
        self.combo_templates.setStyleSheet("background: #262639; color: white; padding: 10px; border-radius: 6px; border: none;")
        p_layout.addWidget(self.combo_templates)
        
        btn_load = QPushButton("Inject Protocol")
        btn_load.clicked.connect(self.load_protocol_to_grid)
        btn_load.setStyleSheet("background-color: #fab387; color: #1e1e2e; font-weight: bold; padding: 12px; border-radius: 6px;")
        p_layout.addWidget(btn_load)
        p_layout.addStretch()
        
        btn_save_prot = QPushButton("Save Matrix as Template")
        btn_save_prot.clicked.connect(self.save_grid_as_protocol)
        btn_save_prot.setStyleSheet("background-color: #313244; color: #cdd6f4; font-weight: bold; padding: 12px; border-radius: 6px;")
        p_layout.addWidget(btn_save_prot)
        layout.addWidget(p_frame)
        
        grid_frame = QFrame()
        grid_frame.setStyleSheet("background: #181825; border-radius: 12px; border: 1px solid #313244;")
        g_layout = QVBoxLayout(grid_frame)
        
        lbl_matrix = QLabel("SESSION EXECUTION MATRIX")
        lbl_matrix.setStyleSheet("font-size: 14px; font-weight: 900; color: #a6e3a1; letter-spacing: 1px; border: none;")
        g_layout.addWidget(lbl_matrix)
        
        self.session_table = QTableWidget()
        self.session_table.setColumnCount(4)
        self.session_table.setHorizontalHeaderLabels(["Exercise", "Load (kg)", "Reps", "RIR"])
        self.session_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.session_table.setStyleSheet("""
            QTableWidget { background: transparent; color: white; border: none; font-size: 14px;}
            QTableWidget::item { padding: 8px; border-bottom: 1px solid #313244;}
            QTableWidget::item:selected { background-color: #313244; }
            QHeaderView::section { background-color: #262639; color: #89b4fa; font-weight: bold; padding: 10px; border: none; border-radius: 4px;}
        """)
        g_layout.addWidget(self.session_table)
        
        controls_layout = QHBoxLayout()
        btn_add_ex = QPushButton("+ Single Exercise")
        btn_add_ex.clicked.connect(self.open_exercise_library)
        btn_add_ex.setStyleSheet("background-color: #313244; color: white; padding: 12px; border-radius: 6px; font-weight: bold;")
        controls_layout.addWidget(btn_add_ex)
        
        controls_layout.addStretch()
        
        self.btn_ai_single = QPushButton("🔍 Analyze Row")
        self.btn_ai_single.clicked.connect(self.on_predict_click)
        self.btn_ai_single.setStyleSheet("background-color: #313244; color: #89b4fa; padding: 12px; border-radius: 6px; font-weight: bold;")
        controls_layout.addWidget(self.btn_ai_single)
        
        self.btn_ai_full = QPushButton("🧠 ANALYZE FULL WORKOUT")
        self.btn_ai_full.clicked.connect(self.on_predict_full_session)
        self.btn_ai_full.setStyleSheet("background-color: #89b4fa; color: #1e1e2e; padding: 12px; border-radius: 6px; font-weight: bold;")
        controls_layout.addWidget(self.btn_ai_full)
        
        btn_commit = QPushButton("✅ COMMIT LOG")
        btn_commit.clicked.connect(self.commit_full_session)
        btn_commit.setStyleSheet("background-color: #a6e3a1; color: #11111b; padding: 12px; border-radius: 6px; font-weight: 900;")
        controls_layout.addWidget(btn_commit)
        
        g_layout.addLayout(controls_layout)
        layout.addWidget(grid_frame, 1)
        return tab

    def create_diet_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 15, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("border: none; background: transparent;")
        form_widget = QWidget()
        form_layout = QVBoxLayout(form_widget)
        
        btn_food_search = QPushButton("🔍 Search USDA Food Database")
        btn_food_search.clicked.connect(self.open_food_search)
        btn_food_search.setStyleSheet("background-color: #89b4fa; color: #1e1e2e; font-weight: 900; padding: 15px; border-radius: 8px; font-size: 14px;")
        form_layout.addWidget(btn_food_search)

        macro_card = QFrame()
        macro_card.setStyleSheet("background: #181825; border-radius: 12px; border: 1px solid #313244; padding: 15px;")
        m_layout = QGridLayout(macro_card)
        
        m_lbl = QLabel("MACRONUTRIENTS")
        m_lbl.setStyleSheet("font-size: 12px; font-weight: 900; color: #fab387; letter-spacing: 1px; border: none;")
        m_layout.addWidget(m_lbl, 0, 0, 1, 4)

        self.diet_protein = self._create_sleek_input("Protein", "g", m_layout, 1, 0)
        self.diet_carbs = self._create_sleek_input("Carbs", "g", m_layout, 1, 2)
        self.diet_fats = self._create_sleek_input("Fats", "g", m_layout, 2, 0)
        
        self.diet_calories = QSpinBox()
        self.diet_calories.setRange(0, 10000)
        self.diet_calories.setSuffix(" kcal")
        self.diet_calories.setStyleSheet("background: #262639; color: #a6e3a1; font-weight: bold; font-size: 16px; padding: 10px; border-radius: 6px; border: none;")
        
        cal_lbl = QLabel("🔥 Calories:")
        cal_lbl.setStyleSheet("color: white; font-weight: bold; border: none;")
        m_layout.addWidget(cal_lbl, 2, 2)
        m_layout.addWidget(self.diet_calories, 2, 3)
        form_layout.addWidget(macro_card)

        for category, items in MICROS_CONFIG.items():
            card = QFrame()
            card.setStyleSheet("background: #181825; border-radius: 12px; border: 1px solid #313244; padding: 15px;")
            c_layout = QGridLayout(card)
            
            c_lbl = QLabel(category.upper())
            c_lbl.setStyleSheet("font-size: 12px; font-weight: 900; color: #cba6f7; letter-spacing: 1px; border: none;")
            c_layout.addWidget(c_lbl, 0, 0, 1, 4)

            row, col = 1, 0
            for name, unit in items:
                inp = QDoubleSpinBox()
                inp.setRange(0, 50000)
                inp.setSuffix(f" {unit}")
                inp.setStyleSheet("background: #262639; color: white; padding: 8px; border-radius: 4px; border: none;")
                self.micro_inputs[name] = inp
                
                n_lbl = QLabel(name)
                n_lbl.setStyleSheet("color: #a6adc8; font-weight: bold; border: none;")
                
                c_layout.addWidget(n_lbl, row, col)
                c_layout.addWidget(inp, row, col + 1)
                col += 2
                if col >= 4: 
                    col = 0
                    row += 1
            form_layout.addWidget(card)
        
        save_btn = QPushButton("💾 Save Nutrition Profile")
        save_btn.clicked.connect(self.save_diet_entry)
        save_btn.setStyleSheet("background-color: #a6e3a1; color: #1e1e2e; font-weight: 900; padding: 15px; border-radius: 8px; font-size: 14px;")
        form_layout.addWidget(save_btn)
        
        scroll.setWidget(form_widget)
        layout.addWidget(scroll)
        return tab

    def _create_sleek_input(self, label, suffix, layout, row, col):
        lbl = QLabel(label)
        lbl.setStyleSheet("color: white; font-weight: bold; border: none;")
        inp = QDoubleSpinBox()
        inp.setRange(0, 10000)
        inp.setSuffix(f" {suffix}")
        inp.valueChanged.connect(self.auto_calculate_calories)
        inp.setStyleSheet("background: #262639; color: white; padding: 10px; border-radius: 6px; border: none; font-size: 14px;")
        layout.addWidget(lbl, row, col)
        layout.addWidget(inp, row, col + 1)
        return inp

    def auto_calculate_calories(self):
        p = self.diet_protein.value() * 4
        c = self.diet_carbs.value() * 4
        f = self.diet_fats.value() * 9
        self.diet_calories.setValue(int(p + c + f))

    def open_food_search(self):
        dialog = FoodSearchDialog(self)
        dialog.food_selected.connect(self._apply_food_macros)
        dialog.exec()

    def _apply_food_macros(self, macros):
        for key, val in macros.items():
            if key == "calories": self.diet_calories.setValue(self.diet_calories.value() + int(val))
            elif key == "protein_g": self.diet_protein.setValue(self.diet_protein.value() + val)
            elif key == "carbs_g": self.diet_carbs.setValue(self.diet_carbs.value() + val)
            elif key == "fats_g": self.diet_fats.setValue(self.diet_fats.value() + val)
            elif key in self.micro_inputs: self.micro_inputs[key].setValue(self.micro_inputs[key].value() + val)

    def create_sleep_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(40, 40, 40, 40)

        header = QLabel("OVERNIGHT RECOVERY")
        header.setStyleSheet("font-size: 24px; font-weight: 900; color: #cba6f7;")
        layout.addWidget(header)

        card = QFrame()
        card.setStyleSheet("background: #181825; border-radius: 16px; border: 1px solid #313244;")
        card_layout = QVBoxLayout(card)
        card_layout.setSpacing(40)
        card_layout.setContentsMargins(30, 40, 30, 40)

        dur_layout = QVBoxLayout()
        dur_header = QHBoxLayout()
        dur_lbl = QLabel("Sleep Duration")
        dur_lbl.setStyleSheet("color: #a6adc8; font-weight: bold; font-size: 16px; border: none;")
        self.dur_val = QLabel("7.5 hrs")
        self.dur_val.setStyleSheet("color: white; font-weight: 900; font-size: 20px; border: none;")
        dur_header.addWidget(dur_lbl); dur_header.addStretch(); dur_header.addWidget(self.dur_val)

        self.slider_duration = QSlider(Qt.Orientation.Horizontal)
        self.slider_duration.setRange(0, 140)
        self.slider_duration.setValue(75)
        self.slider_duration.setStyleSheet(self._slider_style("#89b4fa"))
        self.slider_duration.valueChanged.connect(lambda v: self.dur_val.setText(f"{v/10:.1f} hrs"))

        dur_layout.addLayout(dur_header); dur_layout.addWidget(self.slider_duration)
        card_layout.addLayout(dur_layout)

        qual_layout = QVBoxLayout()
        qual_header = QHBoxLayout()
        qual_lbl = QLabel("Sleep Quality (CNS Readiness)")
        qual_lbl.setStyleSheet("color: #a6adc8; font-weight: bold; font-size: 16px; border: none;")
        self.qual_val = QLabel("7 / 10")
        self.qual_val.setStyleSheet("color: #a6e3a1; font-weight: 900; font-size: 20px; border: none;")
        qual_header.addWidget(qual_lbl); qual_header.addStretch(); qual_header.addWidget(self.qual_val)

        self.slider_quality = QSlider(Qt.Orientation.Horizontal)
        self.slider_quality.setRange(1, 10)
        self.slider_quality.setValue(7)
        self.slider_quality.setStyleSheet(self._slider_style("#a6e3a1"))
        self.slider_quality.valueChanged.connect(self._update_quality_color)

        qual_layout.addLayout(qual_header); qual_layout.addWidget(self.slider_quality)
        card_layout.addLayout(qual_layout)

        layout.addWidget(card)

        save_btn = QPushButton("💾 Log Recovery Metrics")
        save_btn.clicked.connect(self.save_sleep_entry)
        save_btn.setStyleSheet("background-color: #cba6f7; color: #1e1e2e; font-weight: 900; padding: 15px; border-radius: 8px; font-size: 16px; margin-top: 20px;")
        layout.addWidget(save_btn)
        layout.addStretch()
        return tab

    def _update_quality_color(self, v):
        color = "#f38ba8" if v < 5 else "#f9e2af" if v < 8 else "#a6e3a1"
        self.qual_val.setText(f"{v} / 10")
        self.qual_val.setStyleSheet(f"color: {color}; font-weight: 900; font-size: 20px; border: none;")

    def _slider_style(self, color):
        return f"""
            QSlider::groove:horizontal {{ height: 8px; background: #313244; border-radius: 4px; }}
            QSlider::sub-page:horizontal {{ background: {color}; border-radius: 4px; }}
            QSlider::handle:horizontal {{ background: white; width: 20px; margin: -6px 0; border-radius: 10px; }}
        """

    def refresh_templates(self):
        self.combo_templates.clear()
        try:
            templates = self.db.conn.execute("SELECT * FROM workout_templates").fetchall()
            for t in templates: self.combo_templates.addItem(t['name'], t['id'])
        except: pass

    def load_protocol_to_grid(self):
        tid = self.combo_templates.currentData()
        if not tid: return
        self.session_table.setRowCount(0)
        try:
            exercises = self.db.conn.execute("""
                SELECT te.*, e.name FROM template_exercises te
                JOIN exercises e ON te.exercise_id = e.id
                WHERE te.template_id = ? ORDER BY te.order_index ASC
            """, (tid,)).fetchall()
            for ex in exercises:
                for _ in range(ex['target_sets']):
                    self._add_grid_row(ex['exercise_id'], ex['name'], "0.0", "0", "0")
        except: pass

    def save_grid_as_protocol(self):
        user = self.user_manager.get_current_user()
        if not user: return
        if self.session_table.rowCount() == 0:
            QMessageBox.warning(self, "Empty", "Add exercises to the grid first.")
            return
            
        name, ok = QInputDialog.getText(self, "Save Protocol", "Enter Protocol Name:")
        if not ok or not name: return
        desc, ok2 = QInputDialog.getText(self, "Protocol Goal", "Enter Intent (e.g. 'Metabolic stress'):")
        
        try:
            exercises_to_save =[]
            for row in range(self.session_table.rowCount()):
                item = self.session_table.item(row, 0)
                exercises_to_save.append({'id': item.data(Qt.ItemDataRole.UserRole), 'sets': 1})
            
            collapsed =[]
            for ex in exercises_to_save:
                if collapsed and collapsed[-1]['id'] == ex['id']: collapsed[-1]['sets'] += 1
                else: collapsed.append(ex)
                    
            cursor = self.db.conn.cursor()
            cursor.execute("INSERT INTO workout_templates (user_id, name, description) VALUES (?, ?, ?)", (user['id'], name, desc))
            tid = cursor.lastrowid
            
            for i, ex in enumerate(collapsed):
                cursor.execute("INSERT INTO template_exercises (template_id, exercise_id, order_index, target_sets) VALUES (?, ?, ?, ?)", (tid, ex['id'], i, ex['sets']))
            self.db.conn.commit()
            self.refresh_templates()
            QMessageBox.information(self, "Success", "Protocol Saved to Library!")
        except Exception as e: QMessageBox.critical(self, "DB Error", str(e))

    def open_exercise_library(self):
        dialog = ExerciseLibraryDialog(self.db, self)
        dialog.exercise_selected.connect(self.on_single_exercise_added)
        dialog.exec()

    def on_single_exercise_added(self, ex_id, name, is_uni):
        self._add_grid_row(ex_id, name, "0.0", "0", "0")

    def _add_grid_row(self, ex_id, name, weight, reps, rir):
        row = self.session_table.rowCount()
        self.session_table.insertRow(row)
        name_item = QTableWidgetItem(name)
        name_item.setData(Qt.ItemDataRole.UserRole, ex_id)
        name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable) 
        self.session_table.setItem(row, 0, name_item)
        self.session_table.setItem(row, 1, QTableWidgetItem(str(weight)))
        self.session_table.setItem(row, 2, QTableWidgetItem(str(reps)))
        self.session_table.setItem(row, 3, QTableWidgetItem(str(rir)))

    def commit_full_session(self):
        if self.session_table.rowCount() == 0: return
        user = self.user_manager.get_current_user()
        if not user: return
        try:
            session_id = self.db.create_workout_session(user['id'], self.selected_date.toString('yyyy-MM-dd'), "Scientific Grid Log")
            for row in range(self.session_table.rowCount()):
                ex_id = self.session_table.item(row, 0).data(Qt.ItemDataRole.UserRole)
                try:
                    w = float(self.session_table.item(row, 1).text())
                    r = int(self.session_table.item(row, 2).text())
                    rir = int(self.session_table.item(row, 3).text())
                    if w > 0 or r > 0:
                        self.db.add_exercise_performance(
                            workout_session_id=session_id, exercise_id=ex_id, 
                            set_number=row+1, weight_kg=w, reps_completed=r, rir_actual=rir)
                except ValueError: continue 
            self.refresh_data()
            QMessageBox.information(self, "Success", "Session Committed!")
        except Exception as e: QMessageBox.critical(self, "DB Error", str(e))

    def on_predict_click(self):
        if not self.ai_predictor: return
        row = self.session_table.currentRow()
        if row < 0: 
            QMessageBox.warning(self, "Selection", "Please click on a row to analyze that specific exercise.")
            return
        ex_name = self.session_table.item(row, 0).text()
        user = self.user_manager.get_current_user()
        try:
            result = self.ai_predictor.predict(user['id'], ex_name)
            self.show_prediction_results(ex_name, result)
        except Exception as e: QMessageBox.warning(self, "AI Error", str(e))

    def on_predict_full_session(self):
        if not self.ai_predictor: return
        if self.session_table.rowCount() == 0:
            QMessageBox.warning(self, "Empty", "Load a protocol or add exercises first.")
            return

        user = self.user_manager.get_current_user()
        
        exercises =[]
        total_volume = 0
        total_sets = 0
        
        for row in range(self.session_table.rowCount()):
            ex = self.session_table.item(row, 0).text()
            if ex not in exercises: exercises.append(ex)
            try:
                load = float(self.session_table.item(row, 1).text())
                reps = int(self.session_table.item(row, 2).text())
                total_volume += (load * reps)
                total_sets += 1
            except ValueError:
                total_sets += 1
            
        self.btn_ai_full.setText("⏳ Analyzing Workout...")
        from PyQt6.QtWidgets import QApplication
        QApplication.processEvents()
        
        results =[]
        for ex in exercises:
            try:
                res = self.ai_predictor.predict(user['id'], ex)
                results.append((ex, res))
            except: pass
            
        self.btn_ai_full.setText("🧠 ANALYZE FULL WORKOUT")
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Full Session Macro-Analytics")
        dialog.setMinimumSize(700, 600)
        dialog.setStyleSheet("background-color: #1e1e2e; color: white;")
        layout = QVBoxLayout()
        
        header = QLabel("<b>Holistic Workout Analysis</b>")
        header.setStyleSheet("font-size: 20px; color: #89b4fa;")
        layout.addWidget(header)
        
        stats_layout = QHBoxLayout()
        s1 = QLabel(f"Total Movements: {len(exercises)}")
        s2 = QLabel(f"Total Sets: {total_sets}")
        s3 = QLabel(f"Est. Volume: {total_volume:.1f} kg")
        for s in[s1, s2, s3]:
            s.setStyleSheet("background: #262639; padding: 10px; border-radius: 6px; font-weight: bold;")
            stats_layout.addWidget(s)
        layout.addLayout(stats_layout)
        
        if total_sets > 22:
            warn = QLabel("⚠️ WARNING: Total sets exceed 22. This likely exceeds session MRV (Maximum Recoverable Volume) and shifts SFR to junk volume.")
            warn.setStyleSheet("color: #f38ba8; font-weight: bold; padding: 10px; background: rgba(243, 139, 168, 0.1); border-radius: 6px;")
            layout.addWidget(warn)
            
        likely_gains = sum(1 for _, r in results if 'LIKELY' in r['prediction'].upper())
        summary = QLabel(f"Growth Potential: {likely_gains}/{len(exercises)} movements show High Adaptation Probability.")
        summary.setStyleSheet("color: #a6e3a1; font-weight: bold; font-size: 16px; margin: 10px 0;")
        layout.addWidget(summary)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("border: none;")
        w = QWidget()
        v_lay = QVBoxLayout(w)
        
        for ex, res in results:
            card = QFrame()
            col = '#a6e3a1' if 'LIKELY' in res['prediction'].upper() else '#f38ba8'
            card.setStyleSheet(f"background: #181825; border-left: 4px solid {col}; border-radius: 6px; padding: 10px;")
            c_lay = QVBoxLayout(card)
            
            c_lay.addWidget(QLabel(f"<span style='font-size:16px; font-weight:bold; color:{col};'>{ex}</span>"))
            c_lay.addWidget(QLabel(f"<b>Forecast:</b> {res['prediction'].upper()} | <b>RIR Target:</b> {res.get('recommendations',{}).get('rir','N/A')}"))
            
            reason = QLabel(f"<i>{res.get('reason', '')}</i>")
            reason.setStyleSheet("color: #a6adc8;")
            reason.setWordWrap(True)
            c_lay.addWidget(reason)
            v_lay.addWidget(card)
            
        scroll.setWidget(w)
        layout.addWidget(scroll)
        
        btn = QPushButton("Close")
        btn.clicked.connect(dialog.accept)
        btn.setStyleSheet("background-color: #313244; padding: 10px; border-radius: 4px; font-weight: bold;")
        layout.addWidget(btn)
        dialog.setLayout(layout)
        dialog.exec()

    def show_prediction_results(self, exercise, result):
        dialog = QDialog(self)
        dialog.setWindowTitle(f"AI Analytics: {exercise}")
        dialog.setMinimumSize(750, 600)
        dialog.setStyleSheet("background-color: #1e1e2e; color: white;")
        layout = QVBoxLayout()
        
        layout.addWidget(QLabel(f"Model: {result['type']}"))
        outcome = QLabel(f"Forecast: {result['prediction'].upper()}")
        outcome.setStyleSheet(f"font-size: 22px; font-weight: 800; color: {'#a6e3a1' if 'LIKELY' in result['prediction'].upper() else '#f38ba8'};")
        layout.addWidget(outcome)
        
        fig = Figure(figsize=(6, 4), dpi=100)
        fig.patch.set_facecolor('#1e1e2e')
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        ax.set_facecolor('#1e1e2e')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        
        user = self.user_manager.get_current_user()
        history_df = self.db.get_user_workout_history_df(user['id'], exercise)
        
        if not history_df.empty and len(history_df) > 1:
            dates = range(len(history_df))
            weights = history_df['weight_kg'].values
            ax.plot(dates, weights, color='#89b4fa', linewidth=2, marker='o', label='History')
            
            last_x = dates[-1]
            pred_y = result.get('predicted_weight') or weights[-1]
            ax.plot([last_x, last_x+1], [weights[-1], pred_y], color='#a6adc8', linestyle='--', alpha=0.5)
            
            uncert = result.get('uncertainty_range', 0)
            ax.errorbar([last_x+1],[pred_y], yerr=uncert, fmt='*', color='#f38ba8', markersize=15, label='Forecast')
            ax.legend(facecolor='#313244', edgecolor='none', labelcolor='white')
        else:
            ax.text(0.5, 0.5, "Need >2 sessions for trend line", ha='center', color='white', transform=ax.transAxes)
            
        ax.grid(True, alpha=0.1)
        layout.addWidget(canvas)
        
        layout.addWidget(QLabel(f"Target RIR: {result.get('recommendations', {}).get('rir', 'N/A')} | Rest: {result.get('recommendations', {}).get('rest', 'N/A')}s"))
        layout.addWidget(QLabel(f"<i>{result.get('reason', 'Analysis complete.')}</i>"))
        
        btn = QPushButton("Close")
        btn.clicked.connect(dialog.accept)
        btn.setStyleSheet("background-color: #313244; padding: 10px; border-radius: 4px;")
        layout.addWidget(btn)
        dialog.setLayout(layout)
        dialog.exec()

    def refresh_calendar_dots(self):
        user = self.user_manager.get_current_user()
        if not user: return
        
        start_date = self.calendar_strip.current_anchor.addDays(-3).toString("yyyy-MM-dd")
        end_date = self.calendar_strip.current_anchor.addDays(3).toString("yyyy-MM-dd")
        
        flags = {}
        cursor = self.db.conn.cursor()
        
        w_rows = cursor.execute("SELECT session_date FROM workout_sessions WHERE user_id=? AND session_date >= ? AND session_date <= ?", (user['id'], start_date, end_date)).fetchall()
        for r in w_rows:
            d = r[0]
            if d not in flags: flags[d] = []
            if "workout" not in flags[d]: flags[d].append("workout")
            
        d_rows = cursor.execute("SELECT entry_date FROM diet_entries WHERE user_id=? AND entry_date >= ? AND entry_date <= ?", (user['id'], start_date, end_date)).fetchall()
        for r in d_rows:
            d = r[0]
            if d not in flags: flags[d] =[]
            if "diet" not in flags[d]: flags[d].append("diet")
            
        s_rows = cursor.execute("SELECT entry_date FROM sleep_entries WHERE user_id=? AND entry_date >= ? AND entry_date <= ?", (user['id'], start_date, end_date)).fetchall()
        for r in s_rows:
            d = r[0]
            if d not in flags: flags[d] = []
            if "sleep" not in flags[d]: flags[d].append("sleep")
            
        self.calendar_strip.set_activity_flags(flags)

    def refresh_data(self):
        self.session_table.setRowCount(0)
        self.diet_calories.setValue(0); self.diet_protein.setValue(0); self.diet_carbs.setValue(0); self.diet_fats.setValue(0)
        self.slider_duration.setValue(75); self.slider_quality.setValue(7)
        for w in self.micro_inputs.values(): w.setValue(0)
        
        user = self.user_manager.get_current_user()
        if not user: return
        date_str = self.selected_date.toString('yyyy-MM-dd')
        cursor = self.db.conn.cursor()
        
        row = cursor.execute("SELECT * FROM diet_entries WHERE user_id=? AND entry_date=?", (user['id'], date_str)).fetchone()
        if row:
            self.diet_calories.setValue(row['total_calories'] or 0)
            self.diet_protein.setValue(row['protein_g'] or 0)
            self.diet_carbs.setValue(row['carbs_g'] or 0)
            self.diet_fats.setValue(row['fats_g'] or 0)
            if row['notes'] and "MICROS_DATA::" in row['notes']:
                try:
                    data = json.loads(row['notes'].split("MICROS_DATA::")[1])
                    for k, v in data.items():
                        if k in self.micro_inputs: self.micro_inputs[k].setValue(v)
                except: pass

        row = cursor.execute("SELECT * FROM sleep_entries WHERE user_id=? AND entry_date=?", (user['id'], date_str)).fetchone()
        if row:
            self.slider_duration.setValue(int((row['sleep_duration_hours'] or 7.5) * 10))
            self.slider_quality.setValue(row['sleep_quality'] or 7)

        session = cursor.execute("SELECT id FROM workout_sessions WHERE user_id=? AND session_date=?", (user['id'], date_str)).fetchone()
        if session:
            rows = cursor.execute("SELECT ep.*, e.name FROM exercise_performances ep JOIN exercises e ON ep.exercise_id=e.id WHERE workout_session_id=?", (session['id'],)).fetchall()
            for r in rows:
                self._add_grid_row(r['exercise_id'], r['name'], r['weight_kg'], r['reps_completed'], r['rir_actual'])

        self.refresh_calendar_dots()

    def save_diet_entry(self):
        user = self.user_manager.get_current_user()
        micros = {k: v.value() for k, v in self.micro_inputs.items() if v.value() > 0}
        self.db.save_diet_entry(user['id'], self.selected_date.toString('yyyy-MM-dd'), total_calories=self.diet_calories.value(), protein_g=self.diet_protein.value(), carbs_g=self.diet_carbs.value(), fat_g=self.diet_fats.value(), notes=f"MICROS_DATA::{json.dumps(micros)}")
        self.refresh_calendar_dots()
        QMessageBox.information(self, "Saved", "Nutrition Logged")

    def save_sleep_entry(self):
        user = self.user_manager.get_current_user()
        self.db.save_sleep_entry(user['id'], self.selected_date.toString('yyyy-MM-dd'), sleep_duration_hours=self.slider_duration.value()/10, sleep_quality=self.slider_quality.value())
        self.refresh_calendar_dots()
        QMessageBox.information(self, "Saved", "Recovery Logged")

    def on_date_selected(self, date_obj):
        self.selected_date = date_obj
        self.date_info_label.setText(f"Active Date: {self.selected_date.toString('yyyy-MM-dd')}")
        self.refresh_data()