"""
Scientific Hypertrophy Trainer - Main Entry Point
Evidence-based muscle building through progressive knowledge assessment
"""

import sys
import os
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt

# --- FOLDER PATH FIX ---
base_dir = os.path.dirname(os.path.abspath(__file__))
app_dir = os.path.join(base_dir, 'app')
sys.path.insert(0, app_dir)
sys.path.insert(0, base_dir)

try:
    from gui.main_window import MainWindow
except ImportError as e:
    print(f"❌ Critical Import Error: {e}")
    sys.exit(1)

def apply_modern_theme(app):
    """
    Applies a professional 'Scientific Dark' theme.
    Colors: Deep Blue/Grey background, clean white text, vibrant accents.
    """
    theme = """
    /* GLOBAL RESET */
    QWidget {
        background-color: #1e1e2e; /* Dark Navy Background */
        color: #cdd6f4;            /* Soft White Text */
        font-family: 'Segoe UI', 'Roboto', sans-serif;
        font-size: 14px;
    }

    /* CARD SYSTEM */
    QFrame, QGroupBox {
        background-color: #262639; /* Lighter Card Background */
        border: 1px solid #313244;
        border-radius: 12px;
    }
    
    QGroupBox {
        margin-top: 24px;
        padding-top: 15px;
        font-weight: bold;
        font-size: 13px;
        text-transform: uppercase;
        color: #89b4fa; /* Accent Blue */
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        left: 15px;
        padding: 0 5px;
        background-color: #1e1e2e; /* Matches Window Bg to look floating */
    }

    /* INPUT FIELDS */
    QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QTimeEdit {
        background-color: #181825;
        border: 1px solid #45475a;
        border-radius: 6px;
        padding: 8px;
        color: #ffffff;
        font-weight: bold;
    }
    QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {
        border: 1px solid #89b4fa; /* Focus Blue */
        background-color: #1e1e2e;
    }

    /* BUTTONS */
    QPushButton {
        background-color: #313244;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        color: #cdd6f4;
        font-weight: 600;
    }
    QPushButton:hover {
        background-color: #45475a;
    }
    QPushButton:pressed {
        background-color: #585b70;
    }
    
    /* SPECIFIC BUTTON COLORS */
    QPushButton[class="action_btn"] {
        background-color: #89b4fa; /* Blue */
        color: #1e1e2e;
    }
    QPushButton[class="success_btn"] {
        background-color: #a6e3a1; /* Green */
        color: #1e1e2e;
    }
    QPushButton[class="danger_btn"] {
        background-color: #f38ba8; /* Red */
        color: #1e1e2e;
    }

    /* TABS */
    QTabWidget::pane {
        border: 1px solid #313244;
        border-radius: 12px;
        background: #262639;
    }
    QTabBar::tab {
        background: #1e1e2e;
        color: #a6adc8;
        padding: 12px 24px;
        border-top-left-radius: 8px;
        border-top-right-radius: 8px;
        margin-right: 2px;
    }
    QTabBar::tab:selected {
        background: #262639;
        color: #ffffff;
        border-bottom: 2px solid #89b4fa;
    }

    /* SCROLL BARS */
    QScrollBar:vertical {
        background: #1e1e2e;
        width: 10px;
    }
    QScrollBar::handle:vertical {
        background: #45475a;
        border-radius: 5px;
    }
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
        height: 0px;
    }
    
    /* TABLES */
    QTableWidget {
        background-color: #181825;
        gridline-color: #313244;
        border-radius: 8px;
    }
    QHeaderView::section {
        background-color: #313244;
        padding: 8px;
        border: none;
        font-weight: bold;
        color: #cdd6f4;
    }
    """
    app.setStyleSheet(theme)

def main():
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    
    app = QApplication(sys.argv)
    app.setApplicationName("Scientific Hypertrophy Trainer")
    app.setOrganizationName("Hypertrophy AI")
    
    # APPLY THE NEW LOOK
    apply_modern_theme(app)
    
    try:
        window = MainWindow()
        window.db.seed_scientific_exercises()
        
        window.show()
        sys.exit(app.exec())
    except Exception as e:
        print(f"❌ CRASHED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()