"""
Food search dialog - Ultra-modern HUD with dynamic unit conversion (grams, cups, oz).
"""
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QLineEdit, 
    QListWidget, QListWidgetItem, QComboBox, QDoubleSpinBox, QMessageBox, QFrame, QGridLayout
)
from PyQt6.QtCore import pyqtSignal, Qt

try:
    from core.nutrition_lookup import NutritionLookup
except ImportError:
    from app.core.nutrition_lookup import NutritionLookup

DARK_BG, SURFACE, BORDER, ACCENT, TEXT, SUBTEXT = "#1e1e2e", "#181825", "#313244", "#89b4fa", "#cdd6f4", "#a6adc8"

class FoodSearchDialog(QDialog):
    food_selected = pyqtSignal(dict) # Passes fully calculated macros

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Food Database Search")
        self.setMinimumSize(900, 600)
        self.setStyleSheet(f"background-color: {DARK_BG}; color: {TEXT}; font-family: 'Segoe UI', sans-serif;")
        self.lookup = NutritionLookup()
        self.results =[]
        self.current_food = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        layout.setContentsMargins(30, 30, 30, 30)

        # Search Bar
        search_row = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search for a food (e.g. cooked white rice, chicken breast)...")
        self.search_input.setStyleSheet(f"background: {SURFACE}; padding: 15px; border: 1px solid {BORDER}; border-radius: 8px; font-size: 16px;")
        self.search_input.returnPressed.connect(self._do_search)
        search_row.addWidget(self.search_input)

        btn_search = QPushButton("Search")
        btn_search.clicked.connect(self._do_search)
        btn_search.setStyleSheet(f"background-color: {ACCENT}; color: {DARK_BG}; font-weight: bold; padding: 15px 30px; border-radius: 8px; font-size: 14px;")
        search_row.addWidget(btn_search)
        layout.addLayout(search_row)

        self.source_label = QLabel("")
        self.source_label.setStyleSheet(f"color: {SUBTEXT}; font-style: italic;")
        layout.addWidget(self.source_label)

        # Split View: Results List (Left) | Food Details & Calc (Right)
        split_layout = QHBoxLayout()
        
        # Results List
        self.list_widget = QListWidget()
        self.list_widget.setStyleSheet(f"""
            QListWidget {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 8px; outline: none; }}
            QListWidget::item {{ padding: 15px; border-bottom: 1px solid {BORDER}; font-size: 15px; }}
            QListWidget::item:selected {{ background-color: {BORDER}; color: {ACCENT}; font-weight: bold; border-left: 4px solid {ACCENT}; }}
        """)
        self.list_widget.currentItemChanged.connect(self._on_food_selected)
        split_layout.addWidget(self.list_widget, 1)

        # Details Panel
        self.details_frame = QFrame()
        self.details_frame.setStyleSheet(f"background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 8px;")
        d_layout = QVBoxLayout(self.details_frame)
        d_layout.setSpacing(20)
        
        self.lbl_food_name = QLabel("Select a food")
        self.lbl_food_name.setStyleSheet("font-size: 22px; font-weight: 900; color: white;")
        self.lbl_food_name.setWordWrap(True)
        d_layout.addWidget(self.lbl_food_name)

        # Quantity & Unit Controls
        input_row = QHBoxLayout()
        self.spin_qty = QDoubleSpinBox()
        self.spin_qty.setRange(0.1, 10000)
        self.spin_qty.setValue(100)
        self.spin_qty.setStyleSheet(f"background: {DARK_BG}; padding: 12px; border: 1px solid {BORDER}; border-radius: 6px; font-size: 18px; font-weight: bold; color: white;")
        self.spin_qty.valueChanged.connect(self._update_macro_preview)
        
        self.combo_unit = QComboBox()
        self.combo_unit.setStyleSheet(f"background: {DARK_BG}; padding: 12px; border: 1px solid {BORDER}; border-radius: 6px; font-size: 16px; color: white;")
        self.combo_unit.currentIndexChanged.connect(self._update_macro_preview)
        
        input_row.addWidget(self.spin_qty, 1)
        input_row.addWidget(self.combo_unit, 2)
        d_layout.addLayout(input_row)

        # Live Macro Preview
        self.preview_grid = QGridLayout()
        self.lbl_cal = self._create_preview_label("Calories", "#fab387")
        self.lbl_pro = self._create_preview_label("Protein", "#a6e3a1")
        self.lbl_car = self._create_preview_label("Carbs", "#89b4fa")
        self.lbl_fat = self._create_preview_label("Fats", "#f38ba8")
        
        self.preview_grid.addWidget(self.lbl_cal, 0, 0)
        self.preview_grid.addWidget(self.lbl_pro, 0, 1)
        self.preview_grid.addWidget(self.lbl_car, 1, 0)
        self.preview_grid.addWidget(self.lbl_fat, 1, 1)
        d_layout.addLayout(self.preview_grid)

        d_layout.addStretch()

        self.btn_add = QPushButton("Add to Today's Log")
        self.btn_add.clicked.connect(self._on_add)
        self.btn_add.setStyleSheet("background-color: #a6e3a1; color: #1e1e2e; font-weight: 900; padding: 15px; border-radius: 8px; font-size: 16px;")
        self.btn_add.setEnabled(False)
        d_layout.addWidget(self.btn_add)

        split_layout.addWidget(self.details_frame, 1)
        layout.addLayout(split_layout)

    def _create_preview_label(self, title, color):
        lbl = QLabel(f"{title}\n0.0")
        lbl.setStyleSheet(f"background: {DARK_BG}; padding: 15px; border-radius: 6px; font-size: 16px; font-weight: bold; color: {color};")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        return lbl

    def _do_search(self):
        query = self.search_input.text().strip()
        if not query: return
        self.source_label.setText("Searching Database...")
        self.list_widget.clear()
        self.details_frame.hide()
        
        from PyQt6.QtWidgets import QApplication
        QApplication.processEvents()

        self.results = self.lookup.search(query)
        if self.results:
            self.source_label.setText(f"Source: {self.lookup.last_source}  ({len(self.results)} results)")
            for i, food in enumerate(self.results):
                item = QListWidgetItem(food.get("name", "Unknown"))
                item.setData(Qt.ItemDataRole.UserRole, i)
                self.list_widget.addItem(item)
        else:
            self.source_label.setText("No results found.")

    def _on_food_selected(self, item):
        if not item: return
        idx = item.data(Qt.ItemDataRole.UserRole)
        self.current_food = self.results[idx]
        
        self.lbl_food_name.setText(self.current_food["name"])
        
        # Populate Units
        self.combo_unit.blockSignals(True)
        self.combo_unit.clear()
        for i, measure in enumerate(self.current_food["measures"]):
            self.combo_unit.addItem(measure["name"], measure["weight_g"])
        self.combo_unit.blockSignals(False)
        
        self.details_frame.show()
        self.btn_add.setEnabled(True)
        
        # Auto-set to grams if available, default to 100
        self.combo_unit.setCurrentIndex(0)
        self.spin_qty.setValue(100 if self.combo_unit.currentText() == "g" else 1.0)
        self._update_macro_preview()

    def _calculate_multiplier(self):
        if not self.current_food: return 0
        qty = self.spin_qty.value()
        weight_in_g = self.combo_unit.currentData() or 1.0
        # USDA data is per 100g. Formula: (Quantity * Weight of Unit in Grams) / 100
        return (qty * weight_in_g) / 100.0

    def _update_macro_preview(self):
        if not self.current_food: return
        mult = self._calculate_multiplier()
        macros = self.current_food["macros_per_100g"]
        
        cal = macros.get("calories", 0) * mult
        pro = macros.get("protein_g", 0) * mult
        car = macros.get("carbs_g", 0) * mult
        fat = macros.get("fats_g", 0) * mult
        
        self.lbl_cal.setText(f"Calories\n{cal:.0f} kcal")
        self.lbl_pro.setText(f"Protein\n{pro:.1f} g")
        self.lbl_car.setText(f"Carbs\n{car:.1f} g")
        self.lbl_fat.setText(f"Fats\n{fat:.1f} g")

    def _on_add(self):
        if not self.current_food: return
        mult = self._calculate_multiplier()
        
        final_data = {}
        for key, val in self.current_food["macros_per_100g"].items():
            final_data[key] = round(val * mult, 2)
                
        self.food_selected.emit(final_data)
        self.accept()