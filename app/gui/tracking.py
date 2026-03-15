"""
Scientific Hypertrophy Trainer - Tracking Interface v5.3 (FINAL FIX)
- Fixed: Restored 'load_protocol_to_grid' method 
- Fixed: Matplotlib empty legend warning
- Features: Multi-Exercise Grid, Protocol Studio, AI Graphing
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame, 
    QTabWidget, QCalendarWidget, QDoubleSpinBox, QSpinBox, QTextEdit,
    QComboBox, QGroupBox, QScrollArea, QMessageBox, QTableWidget, 
    QTableWidgetItem, QHeaderView, QDialog, QCheckBox, QGridLayout,
    QListWidget, QListWidgetItem, QInputDialog
)
from PyQt6.QtCore import Qt, QDate
from datetime import date
import numpy as np
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

MICROS_CONFIG = {
    "Vitamins (Fat Soluble)":[("Vitamin A", "IU"), ("Vitamin D3", "IU"), ("Vitamin E", "mg"), ("Vitamin K", "mcg")],
    "Vitamins (Water Soluble)":[("Vitamin C", "mg"), ("B-Complex", "mg"), ("Folate", "mcg")],
    "Minerals":[("Magnesium", "mg"), ("Zinc", "mg"), ("Iron", "mg"), ("Calcium", "mg"), ("Sodium", "mg")],
    "Other":[("Water", "L"), ("Fiber", "g"), ("Omega-3", "g")]
}

class TrackingWidget(QWidget):
    def __init__(self, db_manager, tracking_system, user_manager):
        super().__init__()
        self.db = db_manager
        self.tracking_system = tracking_system
        self.user_manager = user_manager
        self.selected_date = date.today()
        self.micro_inputs = {}
        
        self.current_session_data =[] 
        self.current_exercise_info = None 
        
        if ML_AVAILABLE:
            self.ai_predictor = HybridPredictor(self.db, "ml_engine/models/strength_predictor.pt")
        else:
            self.ai_predictor = None
            
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # --- TOP HEADER & PROTOCOL LOADER ---
        header_row = QHBoxLayout()
        title = QLabel("Session Console")
        title.setStyleSheet("font-size: 24px; font-weight: 800; color: #89b4fa;")
        header_row.addWidget(title)
        
        header_row.addStretch()
        
        header_row.addWidget(QLabel("Load Protocol:"))
        self.combo_templates = QComboBox()
        self.combo_templates.setStyleSheet("background: #181825; color: white; padding: 5px; border-radius: 4px;")
        header_row.addWidget(self.combo_templates)
        
        # FIX: The button is now correctly wired to the method
        btn_load = QPushButton("Load Folder")
        btn_load.clicked.connect(self.load_protocol_to_grid) 
        btn_load.setStyleSheet("background-color: #fab387; color: #1e1e2e; font-weight: bold; padding: 8px 15px; border-radius: 4px;")
        header_row.addWidget(btn_load)
        
        main_layout.addLayout(header_row)

        content_layout = QHBoxLayout()
        self.create_calendar_section(content_layout)
        
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabWidget::pane { border: 1px solid #313244; border-radius: 8px; background: #1e1e2e; }
            QTabBar::tab { background: #181825; padding: 12px 24px; border-radius: 4px; color: #a6adc8; font-weight: bold;}
            QTabBar::tab:selected { background: #313244; color: #a6e3a1; border-bottom: 3px solid #a6e3a1;}
        """)
        
        self.tabs.addTab(self.create_workout_tab(), "🏋️ Protocol Execution")
        self.tabs.addTab(self.create_diet_tab(), "🧪 Nutrition")
        self.tabs.addTab(self.create_sleep_tab(), "😴 Recovery")
        
        content_layout.addWidget(self.tabs, 1)
        main_layout.addLayout(content_layout)
        
        self.refresh_templates()
        self.refresh_data()

    def create_calendar_section(self, parent_layout):
        frame = QFrame()
        frame.setFixedWidth(300)
        frame.setStyleSheet("background: #181825; border-radius: 12px; border: 1px solid #313244;")
        layout = QVBoxLayout(frame)
        self.calendar = QCalendarWidget()
        self.calendar.setStyleSheet("background: white; color: black; border-radius: 4px;")
        self.calendar.setSelectedDate(QDate.currentDate())
        self.calendar.clicked.connect(self.on_date_selected)
        layout.addWidget(self.calendar)
        
        self.date_info_label = QLabel(f"Active Date: {self.selected_date}")
        self.date_info_label.setStyleSheet("font-weight: bold; margin-top: 10px; color: #fab387; font-size: 14px;")
        self.date_info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.date_info_label)
        layout.addStretch()
        parent_layout.addWidget(frame)

    # ==========================================
    # 🏋️ WORKOUT TAB (PROTOCOL STUDIO)
    # ==========================================
    def create_workout_tab(self):
        tab = QWidget()
        layout = QHBoxLayout(tab)
        layout.setSpacing(20)
        
        # --- LEFT PANEL: Protocol Manager ---
        protocol_frame = QFrame()
        protocol_frame.setFixedWidth(280)
        protocol_frame.setStyleSheet("background: #262639; border-radius: 8px;")
        p_layout = QVBoxLayout(protocol_frame)
        
        lbl_prot = QLabel("SAVED PROTOCOLS")
        lbl_prot.setStyleSheet("font-size: 14px; font-weight: 900; color: #cba6f7; letter-spacing: 1px;")
        p_layout.addWidget(lbl_prot)
        
        self.protocol_list = QListWidget()
        self.protocol_list.setStyleSheet("""
            QListWidget { background: #181825; border: 1px solid #313244; border-radius: 6px; outline: none; }
            QListWidget::item { padding: 12px; color: #cdd6f4; border-bottom: 1px solid #313244;}
            QListWidget::item:selected { background: #313244; color: #cba6f7; font-weight: bold; border-left: 3px solid #cba6f7;}
        """)
        self.protocol_list.currentItemChanged.connect(self.on_protocol_selected)
        p_layout.addWidget(self.protocol_list)
        
        self.lbl_protocol_desc = QTextEdit()
        self.lbl_protocol_desc.setReadOnly(True)
        self.lbl_protocol_desc.setFixedHeight(100)
        self.lbl_protocol_desc.setStyleSheet("background: #181825; color: #a6adc8; font-size: 12px; border: 1px solid #313244;")
        p_layout.addWidget(self.lbl_protocol_desc)
        
        btn_load_list = QPushButton("Load Protocol From List")
        btn_load_list.clicked.connect(self.load_protocol_to_grid)
        btn_load_list.setStyleSheet("background-color: #cba6f7; color: #11111b; font-weight: bold; padding: 10px; border-radius: 6px;")
        p_layout.addWidget(btn_load_list)
        
        btn_save_prot = QPushButton("Save Current as New Protocol")
        btn_save_prot.clicked.connect(self.save_grid_as_protocol)
        btn_save_prot.setStyleSheet("background-color: #313244; color: #cdd6f4; font-weight: bold; padding: 10px; border-radius: 6px;")
        p_layout.addWidget(btn_save_prot)
        
        layout.addWidget(protocol_frame)
        
        # --- RIGHT PANEL: Execution Grid ---
        grid_frame = QFrame()
        g_layout = QVBoxLayout(grid_frame)
        g_layout.setContentsMargins(0, 0, 0, 0)
        
        lbl_matrix = QLabel("SESSION EXECUTION MATRIX")
        lbl_matrix.setStyleSheet("font-size: 14px; font-weight: 900; color: #a6e3a1; letter-spacing: 1px;")
        g_layout.addWidget(lbl_matrix)
        
        self.session_table = QTableWidget()
        self.session_table.setColumnCount(4)
        self.session_table.setHorizontalHeaderLabels(["Exercise", "Load (kg)", "Reps", "RIR"])
        self.session_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.session_table.setStyleSheet("""
            QTableWidget { background: #181825; color: white; border: 1px solid #313244; border-radius: 8px; font-size: 14px;}
            QTableWidget::item { padding: 5px; }
            QTableWidget::item:selected { background-color: #313244; }
            QHeaderView::section { background-color: #313244; color: #a6e3a1; font-weight: bold; padding: 8px; border: none; }
        """)
        g_layout.addWidget(self.session_table)
        
        controls_layout = QHBoxLayout()
        
        btn_add_ex = QPushButton("+ Add Single Exercise")
        btn_add_ex.clicked.connect(self.open_exercise_library)
        btn_add_ex.setStyleSheet("background-color: #313244; color: white; padding: 12px; border-radius: 6px; font-weight: bold;")
        controls_layout.addWidget(btn_add_ex)
        
        controls_layout.addStretch()
        
        self.btn_ai = QPushButton("🧠 AI Analyze Selected Row")
        self.btn_ai.clicked.connect(self.on_predict_click)
        self.btn_ai.setStyleSheet("background-color: #89b4fa; color: #11111b; padding: 12px; border-radius: 6px; font-weight: bold;")
        controls_layout.addWidget(self.btn_ai)
        
        btn_commit = QPushButton("✅ COMMIT SESSION LOG")
        btn_commit.clicked.connect(self.commit_full_session)
        btn_commit.setStyleSheet("background-color: #a6e3a1; color: #11111b; padding: 12px; border-radius: 6px; font-weight: 900;")
        controls_layout.addWidget(btn_commit)
        
        g_layout.addLayout(controls_layout)
        layout.addWidget(grid_frame, 1)
        
        return tab

    # --- PROTOCOL LOGIC ---
    def refresh_templates(self):
        self.combo_templates.clear()
        self.protocol_list.clear()
        user = self.user_manager.get_current_user()
        u_id = user['id'] if user else None
        
        try:
            query = "SELECT * FROM workout_templates WHERE user_id IS NULL OR user_id = ?"
            templates = self.db.conn.execute(query, (u_id,)).fetchall()
            for t in templates:
                self.combo_templates.addItem(t['name'], t['id'])
                item = QListWidgetItem(t['name'])
                item.setData(Qt.ItemDataRole.UserRole, t['id'])
                item.setData(Qt.ItemDataRole.UserRole + 1, t['description'])
                self.protocol_list.addItem(item)
        except Exception as e:
            print(f"Template DB Error: {e}")

    def on_protocol_selected(self, item):
        if not item: return
        desc = item.data(Qt.ItemDataRole.UserRole + 1)
        self.lbl_protocol_desc.setText(desc or "No scientific notes provided for this protocol.")

    def load_protocol_to_grid(self):
        """FIXED: Checks if a protocol is selected from EITHER the list or the combo box"""
        tid = None
        
        # Prioritize the side list selection if clicked
        if self.protocol_list.currentItem():
            tid = self.protocol_list.currentItem().data(Qt.ItemDataRole.UserRole)
        # Otherwise fallback to whatever is in the combo box at the top
        elif self.combo_templates.currentData():
            tid = self.combo_templates.currentData()
            
        if not tid: 
            QMessageBox.warning(self, "Select", "Select a protocol first.")
            return
            
        self.session_table.setRowCount(0)
        try:
            exercises = self.db.conn.execute("""
                SELECT te.*, e.name 
                FROM template_exercises te
                JOIN exercises e ON te.exercise_id = e.id
                WHERE te.template_id = ? ORDER BY te.order_index ASC
            """, (tid,)).fetchall()
            
            for ex in exercises:
                # Add row for each set requested in template
                for _ in range(ex['target_sets']):
                    self._add_grid_row(ex['exercise_id'], ex['name'], "0.0", "0", "0")
            
            QMessageBox.information(self, "Loaded", "Protocol loaded into matrix. Double-click cells to input your loads.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load: {e}")

    def save_grid_as_protocol(self):
        user = self.user_manager.get_current_user()
        if not user: return
        if self.session_table.rowCount() == 0:
            QMessageBox.warning(self, "Empty", "Add exercises to the grid first.")
            return
            
        name, ok = QInputDialog.getText(self, "Save Protocol", "Enter Scientific Protocol Name:")
        if not ok or not name: return
        
        desc, ok2 = QInputDialog.getText(self, "Protocol Goal", "Enter Scientific Intent (e.g. 'Metabolic stress'):")
        
        try:
            exercises_to_save =[]
            for row in range(self.session_table.rowCount()):
                item = self.session_table.item(row, 0)
                ex_id = item.data(Qt.ItemDataRole.UserRole)
                exercises_to_save.append({'id': ex_id, 'sets': 1})
            
            collapsed =[]
            for ex in exercises_to_save:
                if collapsed and collapsed[-1]['id'] == ex['id']:
                    collapsed[-1]['sets'] += 1
                else:
                    collapsed.append(ex)
                    
            cursor = self.db.conn.cursor()
            cursor.execute("INSERT INTO workout_templates (user_id, name, description) VALUES (?, ?, ?)", 
                           (user['id'], name, desc))
            tid = cursor.lastrowid
            
            for i, ex in enumerate(collapsed):
                cursor.execute("INSERT INTO template_exercises (template_id, exercise_id, order_index, target_sets) VALUES (?, ?, ?, ?)",
                               (tid, ex['id'], i, ex['sets']))
            self.db.conn.commit()
            
            QMessageBox.information(self, "Success", "Protocol Saved to Library!")
            self.refresh_templates()
        except Exception as e:
            QMessageBox.critical(self, "DB Error", str(e))

    # --- GRID MANAGEMENT ---
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
        if self.session_table.rowCount() == 0:
            QMessageBox.warning(self, "Empty", "No data to commit.")
            return
            
        user = self.user_manager.get_current_user()
        if not user: return
        
        try:
            session_id = self.db.create_workout_session(user['id'], self.selected_date, "Scientific Grid Log")
            
            rows_saved = 0
            for row in range(self.session_table.rowCount()):
                ex_id_data = self.session_table.item(row, 0).data(Qt.ItemDataRole.UserRole)
                ex_id = int(ex_id_data) if ex_id_data else 1
                try:
                    w = float(self.session_table.item(row, 1).text())
                    r = int(self.session_table.item(row, 2).text())
                    rir = int(self.session_table.item(row, 3).text())
                except ValueError:
                    continue 
                
                if w > 0 or r > 0:
                    self.db.add_exercise_performance(
                        workout_session_id=session_id, 
                        exercise_id=ex_id, 
                        set_number=rows_saved+1, 
                        weight_kg=w, 
                        reps_completed=r, 
                        rir_actual=rir
                    )
                    rows_saved += 1
            
            QMessageBox.information(self, "Success", f"Session Committed! {rows_saved} sets logged.")
            self.refresh_data()
        except Exception as e:
            QMessageBox.critical(self, "DB Error", str(e))

    # ==========================================
    # 🧠 AI PREDICTION
    # ==========================================
    def on_predict_click(self):
        if not self.ai_predictor: 
            QMessageBox.warning(self, "Error", "AI Engine missing.")
            return
            
        row = self.session_table.currentRow()
        if row < 0:
            QMessageBox.warning(self, "Selection", "Please click on a row in the table to analyze that specific exercise.")
            return
            
        ex_name = self.session_table.item(row, 0).text()
        user = self.user_manager.get_current_user()
        
        self.btn_ai.setText("⏳ Processing...")
        try:
            result = self.ai_predictor.predict(user['id'], ex_name)
            self.show_prediction_results(ex_name, result)
        except Exception as e:
            QMessageBox.warning(self, "AI Error", str(e))
        finally:
            self.btn_ai.setText("🧠 AI Analyze Selected Row")

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
        
        # Check if we have valid data to plot
        if not history_df.empty and len(history_df) > 1:
            dates = range(len(history_df))
            weights = history_df['weight_kg'].values
            ax.plot(dates, weights, color='#89b4fa', linewidth=2, marker='o', label='History')
            
            last_x = dates[-1]
            pred_y = result.get('predicted_weight') or weights[-1]
            ax.plot([last_x, last_x+1], [weights[-1], pred_y], color='#a6adc8', linestyle='--', alpha=0.5)
            
            uncert = result.get('uncertainty_range')
            if uncert is None: uncert = 0
            
            ax.errorbar([last_x+1],[pred_y], yerr=uncert, fmt='*', color='#f38ba8', markersize=15, label='Forecast')
            
            # FIX: Legend is only drawn if there are actual labeled plots
            ax.legend(facecolor='#313244', edgecolor='none', labelcolor='white')
        else:
            # FIX: Fallback visualization for empty data
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

    # ==========================================
    # DIET / SLEEP / REFRESH
    # ==========================================
    def create_diet_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        form_widget = QWidget()
        form_layout = QVBoxLayout(form_widget)
        
        macros_group = QGroupBox("Macronutrients")
        macros_group.setStyleSheet("QGroupBox { color: white; font-weight: bold; border: 1px solid #313244; }")
        grid = QGridLayout(macros_group)
        self.diet_protein = self._create_macro_input("Protein", "g", grid, 0, 0)
        self.diet_carbs = self._create_macro_input("Carbs", "g", grid, 0, 2)
        self.diet_fats = self._create_macro_input("Fats", "g", grid, 1, 0)
        self.diet_calories = QSpinBox()
        self.diet_calories.setRange(0, 10000)
        self.diet_calories.setSuffix(" kcal")
        grid.addWidget(QLabel("🔥 Calories:"), 1, 2)
        grid.addWidget(self.diet_calories, 1, 3)
        form_layout.addWidget(macros_group)

        for category, items in MICROS_CONFIG.items():
            group = QGroupBox(category)
            group.setStyleSheet("QGroupBox { color: white; font-weight: bold; border: 1px solid #313244; }")
            g_layout = QGridLayout(group)
            row, col = 0, 0
            for name, unit in items:
                g_layout.addWidget(QLabel(name + ":"), row, col)
                inp = QDoubleSpinBox()
                inp.setRange(0, 50000)
                self.micro_inputs[name] = inp
                g_layout.addWidget(inp, row, col + 1)
                col += 2
                if col >= 6: col = 0; row += 1
            form_layout.addWidget(group)
        
        save_btn = QPushButton("💾 Save Nutrition")
        save_btn.clicked.connect(self.save_diet_entry)
        save_btn.setStyleSheet("background-color: #a6e3a1; color: black; font-weight: bold; padding: 10px;")
        form_layout.addWidget(save_btn)
        
        scroll.setWidget(form_widget)
        layout.addWidget(scroll)
        return tab

    def _create_macro_input(self, label, suffix, layout, row, col):
        lbl = QLabel(f"{label}:")
        inp = QDoubleSpinBox()
        inp.setRange(0, 1000)
        inp.valueChanged.connect(self.auto_calculate_calories)
        layout.addWidget(lbl, row, col)
        layout.addWidget(inp, row, col + 1)
        return inp

    def auto_calculate_calories(self):
        p = self.diet_protein.value() * 4
        c = self.diet_carbs.value() * 4
        f = self.diet_fats.value() * 9
        self.diet_calories.setValue(int(p + c + f))

    def create_sleep_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        group = QGroupBox("Sleep Metrics")
        group.setStyleSheet("QGroupBox { color: white; font-weight: bold; border: 1px solid #313244; }")
        grid = QGridLayout(group)
        self.sleep_duration = QDoubleSpinBox()
        self.sleep_duration.setRange(0, 24)
        grid.addWidget(QLabel("Hours:"), 0, 0)
        grid.addWidget(self.sleep_duration, 0, 1)
        self.sleep_quality = QSpinBox()
        self.sleep_quality.setRange(1, 10)
        grid.addWidget(QLabel("Quality (1-10):"), 1, 0)
        grid.addWidget(self.sleep_quality, 1, 1)
        layout.addWidget(group)
        
        save = QPushButton("Save Recovery")
        save.clicked.connect(self.save_sleep_entry)
        save.setStyleSheet("background-color: #a6e3a1; color: black; font-weight: bold; padding: 10px;")
        layout.addWidget(save)
        layout.addStretch()
        return tab

    def refresh_data(self):
        self.load_existing_data()

    def load_existing_data(self):
        self.clear_all_forms()
        user = self.user_manager.get_current_user()
        if not user: return
        date_str = self.selected_date.strftime('%Y-%m-%d')
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
            self.sleep_duration.setValue(row['sleep_duration_hours'] or 0)
            self.sleep_quality.setValue(row['sleep_quality'] or 0)

        session = cursor.execute("SELECT id FROM workout_sessions WHERE user_id=? AND session_date=?", (user['id'], date_str)).fetchone()
        if session:
            rows = cursor.execute("SELECT ep.*, e.name FROM exercise_performances ep JOIN exercises e ON ep.exercise_id=e.id WHERE workout_session_id=?", (session['id'],)).fetchall()
            for r in rows:
                self._add_grid_row(r['exercise_id'], r['name'], r['weight_kg'], r['reps_completed'], r['rir_actual'])

    def save_diet_entry(self):
        user = self.user_manager.get_current_user()
        micros = {k: v.value() for k, v in self.micro_inputs.items() if v.value() > 0}
        try:
            self.db.save_diet_entry(user['id'], self.selected_date, total_calories=self.diet_calories.value(), protein_g=self.diet_protein.value(), carbs_g=self.diet_carbs.value(), fat_g=self.diet_fats.value(), notes=f"MICROS_DATA::{json.dumps(micros)}")
            QMessageBox.information(self, "Saved", "Nutrition Logged")
        except: pass

    def save_sleep_entry(self):
        user = self.user_manager.get_current_user()
        try:
            self.db.save_sleep_entry(user['id'], self.selected_date, sleep_duration_hours=self.sleep_duration.value(), sleep_quality=self.sleep_quality.value())
            QMessageBox.information(self, "Saved", "Recovery Logged")
        except: pass

    def clear_all_forms(self):
        self.session_table.setRowCount(0)
        self.diet_calories.setValue(0); self.diet_protein.setValue(0)
        self.sleep_duration.setValue(0); self.sleep_quality.setValue(1)
        for w in self.micro_inputs.values(): w.setValue(0)

    def on_date_selected(self):
        self.selected_date = self.calendar.selectedDate().toPyDate()
        self.date_info_label.setText(f"Active Date: {self.selected_date}")
        self.refresh_data()

    def jump_to_today(self):
        self.calendar.setSelectedDate(QDate.currentDate())
        self.on_date_selected()