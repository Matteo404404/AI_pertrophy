"""
Scientific Hypertrophy Trainer - Elite Analytics Command Center
Visualizes data with interactive custom charts, SFR Matrix, and LLM Debrief.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QScrollArea, QComboBox, QGridLayout, QPushButton
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
import numpy as np
from datetime import datetime, timedelta
import requests

from utils.charts import (
    create_sra_curve_chart, 
    create_tonnage_chart,
    create_muscle_readiness_chart,
    create_sfr_scatter_plot,
    create_1rm_trend_chart
)

class AIDebriefWorker(QThread):
    finished = pyqtSignal(str)
    
    def __init__(self, data_context):
        super().__init__()
        self.data_context = data_context

    def run(self):
        prompt = (
            "Act as an elite sports scientist analyzing a week of training data for an athlete. "
            f"DATA: {self.data_context}\n"
            "Write a highly professional, 3-paragraph weekly debrief. "
            "1. Praise the good (e.g., volume, consistency). "
            "2. Identify the bottleneck (e.g., high systemic fatigue, low sleep, or junk volume). "
            "3. Give 1 scientific, actionable change for next week's mesocycle. "
            "Use markdown formatting. Do not use generic filler."
        )
        try:
            resp = requests.post("http://localhost:11434/api/generate", 
                                 json={"model": "qwen3:1.7b", "prompt": prompt, "stream": False}, 
                                 timeout=60)
            if resp.status_code == 200:
                self.finished.emit(resp.json().get("response", "").strip())
            else:
                self.finished.emit("⚠️ Connection to AI Engine failed.")
        except Exception as e:
            self.finished.emit(f"⚠️ AI Engine Offline or Timeout: {str(e)}")


class AnalyticsWidget(QWidget):
    def __init__(self, db_manager, tracking_system, user_manager):
        super().__init__()
        self.db = db_manager
        self.tracking_system = tracking_system
        self.user_manager = user_manager
        self.days_to_fetch = 30
        self.workers =[]
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(40, 40, 40, 40)
        main_layout.setSpacing(30)
        
        # --- HEADER & CONTROLS ---
        header_layout = QHBoxLayout()
        header = QLabel("PERFORMANCE ANALYTICS")
        header.setStyleSheet("font-size: 32px; font-weight: 900; color: #89b4fa; letter-spacing: 2px;")
        header_layout.addWidget(header)
        
        header_layout.addStretch()
        
        header_layout.addWidget(QLabel("Timeframe:"))
        self.combo_timeframe = QComboBox()
        self.combo_timeframe.addItems(["Last 14 Days", "Last 30 Days", "Last 90 Days"])
        self.combo_timeframe.setCurrentIndex(1)
        self.combo_timeframe.setStyleSheet("background: #181825; color: white; padding: 8px 15px; border-radius: 6px; border: 1px solid #313244; font-weight: bold;")
        self.combo_timeframe.currentIndexChanged.connect(self._on_timeframe_changed)
        header_layout.addWidget(self.combo_timeframe)
        
        main_layout.addLayout(header_layout)
        
        # --- SCROLL AREA ---
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setStyleSheet("border: none; background: transparent;")
        
        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setSpacing(30)
        self.scroll.setWidget(self.content_widget)
        
        main_layout.addWidget(self.scroll)

    def _on_timeframe_changed(self, index):
        days_map = [14, 30, 90]
        self.days_to_fetch = days_map[index]
        self.refresh_data()

    def _create_chart_card(self, widget):
        frame = QFrame()
        frame.setStyleSheet("QFrame { background: #181825; border-radius: 16px; border: 1px solid #2a2b3c; } QFrame:hover { border: 1px solid #45475a; }")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.addWidget(widget)
        return frame

    def refresh_data(self):
        while self.content_layout.count():
            child = self.content_layout.takeAt(0)
            if child.widget(): child.widget().deleteLater()
            
        user = self.user_manager.get_current_user()
        if not user:
            self.content_layout.addWidget(QLabel("Please select an athlete profile."))
            return

        # ROW 1: SRA Curve
        sra_dates, fitness, fatigue, readiness = self._get_sra_data(user['id'], self.days_to_fetch)
        self.content_layout.addWidget(self._create_chart_card(create_sra_curve_chart(sra_dates, fitness, fatigue, readiness)))

        # ROW 2: SFR Scatter Plot (NEW)
        ex_names, stims, fats = self._get_sfr_data(user['id'], self.days_to_fetch)
        self.content_layout.addWidget(self._create_chart_card(create_sfr_scatter_plot(ex_names, stims, fats)))

        # ROW 3: Grid (1RM Trend, Tonnage, Muscle Readiness)
        grid_layout = QGridLayout()
        grid_layout.setSpacing(30)
        
        rm_dates, rm_data = self._get_1rm_data(user['id'], self.days_to_fetch)
        if rm_data: grid_layout.addWidget(self._create_chart_card(create_1rm_trend_chart(rm_dates, rm_data)), 0, 0, 1, 2)
        
        ton_dates, ton_data = self._get_tonnage_data(user['id'], self.days_to_fetch)
        grid_layout.addWidget(self._create_chart_card(create_tonnage_chart(ton_dates, ton_data)), 1, 0)
        
        muscles, r_scores = self._get_readiness_data(user['id'])
        grid_layout.addWidget(self._create_chart_card(create_muscle_readiness_chart(muscles, r_scores)), 1, 1)

        grid_widget = QWidget()
        grid_widget.setLayout(grid_layout)
        self.content_layout.addWidget(grid_widget)

        # ROW 4: AI WEEKLY DEBRIEF (NEW)
        ai_frame = QFrame()
        ai_frame.setStyleSheet("background: #181825; border-radius: 16px; border: 1px solid #89b4fa;")
        ai_layout = QVBoxLayout(ai_frame)
        
        ai_header = QLabel("🧠 Sports Scientist: Weekly Debrief")
        ai_header.setStyleSheet("font-size: 18px; font-weight: 900; color: #89b4fa; border: none;")
        ai_layout.addWidget(ai_header)
        
        self.ai_textbox = QLabel("Click the button below to generate a holistic, AI-powered analysis of your entire training week.")
        self.ai_textbox.setWordWrap(True)
        self.ai_textbox.setStyleSheet("color: #cdd6f4; font-size: 15px; line-height: 1.5; padding: 10px; border: none;")
        ai_layout.addWidget(self.ai_textbox)
        
        self.btn_gen_debrief = QPushButton("Generate Weekly Analysis")
        self.btn_gen_debrief.setStyleSheet("background-color: #89b4fa; color: #1e1e2e; font-weight: 900; padding: 15px; border-radius: 8px; font-size: 14px;")
        self.btn_gen_debrief.clicked.connect(lambda: self._generate_ai_debrief(user['id'], ton_data, sets_data=len(ex_names)))
        ai_layout.addWidget(self.btn_gen_debrief)
        
        self.content_layout.addWidget(ai_frame)
        self.content_layout.addStretch()

    def _generate_ai_debrief(self, user_id, tonnage_data, sets_data):
        self.btn_gen_debrief.setText("⏳ AI Coach is analyzing your biometrics...")
        self.btn_gen_debrief.setEnabled(False)
        
        # Build Data Context
        cursor = self.db.conn.cursor()
        sleep = cursor.execute("SELECT AVG(sleep_quality) FROM sleep_entries WHERE user_id=? AND entry_date >= date('now', '-7 days')", (user_id,)).fetchone()[0] or 6.0
        cals = cursor.execute("SELECT AVG(total_calories) FROM diet_entries WHERE user_id=? AND entry_date >= date('now', '-7 days')", (user_id,)).fetchone()[0] or 2500
        ton = sum(tonnage_data) if tonnage_data else 0
        
        context = f"Total Weekly Tonnage: {ton}kg. Number of Exercises logged: {sets_data}. Avg Sleep Quality: {sleep}/10. Avg Daily Calories: {cals}kcal."
        
        worker = AIDebriefWorker(context)
        def on_complete(text):
            try:
                self.ai_textbox.setText(text)
                self.btn_gen_debrief.setText("Regenerate Analysis")
                self.btn_gen_debrief.setEnabled(True)
            except RuntimeError: pass # Safe UI exit
        worker.finished.connect(on_complete)
        self.workers.append(worker)
        worker.start()

    # --- DATA ALGORITHMS ---
    def _get_sra_data(self, user_id, days):
        cursor = self.db.conn.cursor()
        rows = cursor.execute(f"SELECT session_date, COUNT(*) FROM workout_sessions ws JOIN exercise_performances ep ON ws.id = ep.workout_session_id WHERE ws.user_id = ? AND ws.session_date >= date('now', '-{days} days') GROUP BY session_date ORDER BY session_date ASC", (user_id,)).fetchall()
        if len(rows) > 3:
            dates = [r[0][-5:] for r in rows] 
            sets =[r[1] for r in rows]
            fitness, fatigue, readiness = [],[],[]
            fit_val, fat_val = 10.0, 5.0
            for s in sets:
                fit_val = (fit_val * 0.9) + (s * 0.8) 
                fat_val = (fat_val * 0.7) + (s * 1.5) 
                fitness.append(fit_val); fatigue.append(fat_val); readiness.append(fit_val - fat_val)
            return dates, fitness, fatigue, readiness
        
        # Simulated Default
        dates =[(datetime.now() - timedelta(days=i)).strftime("%m-%d") for i in range(14, 0, -1)]
        x = np.arange(14)
        fitness = 20 + x * 0.5 + np.sin(x/3) * 3
        fatigue = 15 * np.exp(-x/5) + np.sin(x/1.5) * 5 + (x%4==0)*8
        return dates, fitness, fatigue, fitness - fatigue

    def _get_sfr_data(self, user_id, days):
        """Calculates Stimulus vs Fatigue mapping for Junk Volume detection."""
        cursor = self.db.conn.cursor()
        rows = cursor.execute(f"""
            SELECT e.name, e.stability_score, AVG(ep.rir_actual), COUNT(ep.id)
            FROM exercise_performances ep
            JOIN exercises e ON ep.exercise_id = e.id
            JOIN workout_sessions ws ON ep.workout_session_id = ws.id
            WHERE ws.user_id = ? AND ws.session_date >= date('now', '-{days} days')
            GROUP BY e.id
        """, (user_id,)).fetchall()
        
        if not rows:
            # Dummy data so chart renders beautifully for new users
            return["Squat", "Leg Ext", "RDL", "Bicep Curl", "Bench"], [6, 8, 5, 9, 7],[9, 3, 10, 2, 6]
            
        ex_names, stims, fats = [],[],[]
        for name, stab, rir, sets in rows:
            stab = stab if stab else 5
            rir = rir if rir is not None else 2
            # Math: High RIR = Low Stim. Low Stability = High Fatigue.
            stim = (10 - rir) * (sets * 0.5) 
            fat = (11 - stab) * sets * (1.2 if rir < 2 else 0.8)
            
            ex_names.append(name)
            stims.append(min(10, stim))
            fats.append(min(10, fat))
        return ex_names, stims, fats

    def _get_readiness_data(self, user_id):
        """Maps specific muscle group fatigue decay over last 72 hours."""
        cursor = self.db.conn.cursor()
        rows = cursor.execute("""
            SELECT e.muscle_group_primary, COUNT(ep.id) 
            FROM exercise_performances ep
            JOIN exercises e ON ep.exercise_id = e.id
            JOIN workout_sessions ws ON ep.workout_session_id = ws.id
            WHERE ws.user_id = ? AND ws.session_date >= date('now', '-3 days')
            GROUP BY e.muscle_group_primary
        """, (user_id,)).fetchall()
        
        base_muscles = {"Chest": 100, "Back": 100, "Quads": 100, "Hamstrings": 100, "Shoulders": 100, "Biceps": 100, "Triceps": 100}
        for m, sets in rows:
            if m in base_muscles:
                # 1 set drops readiness by ~5%. 10 sets = 50% readiness.
                base_muscles[m] = max(10, 100 - (sets * 5))
                
        return list(base_muscles.keys()), list(base_muscles.values())

    def _get_1rm_data(self, user_id, days):
        cursor = self.db.conn.cursor()
        top_ex = cursor.execute(f"SELECT e.id, e.name FROM exercise_performances ep JOIN exercises e ON ep.exercise_id = e.id JOIN workout_sessions ws ON ep.workout_session_id = ws.id WHERE ws.user_id = ? AND ws.session_date >= date('now', '-{days} days') GROUP BY e.id ORDER BY COUNT(*) DESC LIMIT 3", (user_id,)).fetchall()
        if not top_ex: return[], {}
        
        dates_rows = cursor.execute(f"SELECT DISTINCT session_date FROM workout_sessions WHERE user_id = ? AND session_date >= date('now', '-{days} days') ORDER BY session_date ASC", (user_id,)).fetchall()
        dates = [r[0][-5:] for r in dates_rows]
        if len(dates) < 2: return[], {}
        
        rm_data = {}
        for ex_id, ex_name in top_ex:
            perf_rows = cursor.execute("SELECT ws.session_date, MAX(ep.weight_kg * (1 + 0.0333 * ep.reps_completed)) FROM exercise_performances ep JOIN workout_sessions ws ON ep.workout_session_id = ws.id WHERE ws.user_id = ? AND ep.exercise_id = ? GROUP BY ws.session_date ORDER BY ws.session_date ASC", (user_id, ex_id)).fetchall()
            perf_dict = {r[0][-5:]: r[1] for r in perf_rows}
            y_vals, last_val =[], list(perf_dict.values())[0] if perf_dict else 0
            for d in dates:
                if d in perf_dict: last_val = perf_dict[d]
                y_vals.append(last_val)
            rm_data[ex_name] = y_vals
        return dates, rm_data

    def _get_tonnage_data(self, user_id, days):
        cursor = self.db.conn.cursor()
        rows = cursor.execute(f"SELECT session_date, SUM(ep.weight_kg * ep.reps_completed * 1) FROM exercise_performances ep JOIN workout_sessions ws ON ep.workout_session_id = ws.id WHERE ws.user_id = ? AND ws.session_date >= date('now', '-{days} days') GROUP BY session_date ORDER BY session_date ASC", (user_id,)).fetchall()
        if len(rows) > 2: return [r[0][-5:] for r in rows], [r[1] for r in rows]
        
        dates =[(datetime.now() - timedelta(days=i)).strftime("%m-%d") for i in range(10, 0, -2)]
        return dates,[3500, 4200, 3800, 5100, 4900]
