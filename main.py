"""
Scientific Hypertrophy Trainer - Main Entry Point
Evidence-based muscle building through progressive knowledge assessment
"""

import sys
import os
import subprocess
import requests
import time
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

def ensure_ollama():
    """Silently starts Ollama in the background if it isn't running."""
    try:
        requests.get("http://localhost:11434/", timeout=1)
        print("✅ Ollama AI Engine is already running.")
        return None
    except requests.exceptions.ConnectionError:
        print("⏳ Starting Ollama AI Engine in the background...")
        process = subprocess.Popen(["ollama", "serve"], 
            stdout=subprocess.DEVNULL, 
            stderr=subprocess.DEVNULL
        )
        time.sleep(2) 
        print("✅ Ollama AI Engine Online.")
        return process

def apply_modern_theme(app):
    """
    ULTRA-PREMIUM 'Silicon Valley' Dark Theme.
    Sleek pill buttons, glowing focus states, floating cards, ultra-thin progress bars.
    """
    theme = """
    /* GLOBAL RESET */
    QWidget {
        background-color: #11111b; /* Deepest Navy */
        color: #cdd6f4;
        font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
        font-size: 14px;
    }

    /* FLOATING CARD SYSTEM */
    QFrame, QGroupBox {
        background-color: #181825; /* Elevated Surface */
        border: 1px solid #2a2b3c;
        border-radius: 16px; /* Smooth rounded corners */
    }
    
    QGroupBox {
        margin-top: 24px;
        padding-top: 20px;
        font-weight: 900;
        font-size: 12px;
        letter-spacing: 1px;
        text-transform: uppercase;
        color: #cba6f7; /* Purple Accent */
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        left: 20px;
        padding: 0 8px;
        background-color: #11111b; 
        border-radius: 6px;
    }

    /* SLEEK INPUT FIELDS */
    QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QTimeEdit {
        background-color: #1e1e2e;
        border: 1px solid #313244;
        border-radius: 8px;
        padding: 10px 15px;
        color: #ffffff;
        font-weight: 600;
        selection-background-color: #89b4fa;
    }
    QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
        border: 1px solid #89b4fa; /* Neon Blue Glow */
        background-color: #262639;
    }
    
    QComboBox::drop-down { border: none; }
    QComboBox::down-arrow { 
        image: none; 
        border-left: 5px solid transparent;
        border-right: 5px solid transparent;
        border-top: 5px solid #a6adc8;
        margin-right: 10px;
    }

    /* MODERN PILL BUTTONS */
    QPushButton {
        background-color: #313244;
        border: none;
        border-radius: 10px; /* Pill shape */
        padding: 12px 24px;
        color: #cdd6f4;
        font-weight: 800;
        font-size: 13px;
        letter-spacing: 0.5px;
    }
    QPushButton:hover {
        background-color: #45475a;
    }
    QPushButton:pressed {
        background-color: #585b70;
        padding-top: 14px; /* Press effect */
        padding-bottom: 10px;
    }

    /* MINIMALIST TABS */
    QTabWidget::pane {
        border: none;
        background: transparent;
    }
    QTabBar::tab {
        background: transparent;
        color: #6c7086;
        padding: 10px 20px;
        font-weight: 800;
        font-size: 15px;
        margin-right: 10px;
        border-bottom: 3px solid transparent;
    }
    QTabBar::tab:hover {
        color: #a6adc8;
    }
    QTabBar::tab:selected {
        color: #89b4fa;
        border-bottom: 3px solid #89b4fa; /* Just a sleek bottom line */
    }

    /* MAC-STYLE INVISIBLE SCROLLBARS */
    QScrollBar:vertical {
        background: transparent;
        width: 8px;
        margin: 0px;
    }
    QScrollBar::handle:vertical {
        background: #313244;
        border-radius: 4px;
        min-height: 30px;
    }
    QScrollBar::handle:vertical:hover {
        background: #45475a;
    }
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
        height: 0px;
    }

    /* BEAUTIFUL PROGRESS BARS */
    QProgressBar {
        background-color: #1e1e2e;
        border: none;
        border-radius: 4px;
        height: 8px;
        text-align: center;
        color: transparent; /* Hide internal text */
    }
    QProgressBar::chunk {
        background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #89b4fa, stop:1 #cba6f7);
        border-radius: 4px;
    }

    /* TABLES */
    QTableWidget {
        background-color: #181825;
        gridline-color: #2a2b3c;
        border: none;
        border-radius: 12px;
    }
    QHeaderView::section {
        background-color: #1e1e2e;
        padding: 12px;
        border: none;
        font-weight: 900;
        color: #a6adc8;
        text-transform: uppercase;
        font-size: 11px;
        letter-spacing: 1px;
    }
    QTableWidget::item {
        padding: 10px;
        border-bottom: 1px solid #2a2b3c;
    }
    QTableWidget::item:selected {
        background-color: rgba(137, 180, 250, 0.15); /* Transparent blue highlight */
        color: #89b4fa;
    }
    """
    app.setStyleSheet(theme)

def main():
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    
    ollama_process = ensure_ollama()
    
    app = QApplication(sys.argv)
    app.setApplicationName("Scientific Hypertrophy Trainer")
    app.setOrganizationName("Hypertrophy AI")
    
    apply_modern_theme(app)
    
    try:
        window = MainWindow()
        window.show()
        exit_code = app.exec()
        
        if ollama_process:
            print("🛑 Shutting down background AI Engine...")
            ollama_process.terminate()
            
        sys.exit(exit_code)
    except Exception as e:
        print(f"❌ CRASHED: {e}")
        if ollama_process:
            ollama_process.terminate()
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()