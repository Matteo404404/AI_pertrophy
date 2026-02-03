"""
Scientific Hypertrophy Trainer - Main Entry Point
Evidence-based muscle building through progressive knowledge assessment
"""

import sys
import os
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt

# --- FOLDER PATH FIX ---
# 1. Get the absolute path to the project root (where main.py is)
base_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Define the path to the 'app' folder
app_dir = os.path.join(base_dir, 'app')

# 3. Add 'app' to Python's search path
# This tricks Python into thinking 'gui' and 'core' are right here,
# allowing "from gui.main_window import MainWindow" to work.
sys.path.insert(0, app_dir)

# 4. Add root to path so we can also find 'ml_engine'
sys.path.insert(0, base_dir)
# -----------------------

# Now we can safely import from the folders inside 'app'
try:
    from gui.main_window import MainWindow
except ImportError as e:
    print("❌ CRITICAL IMPORT ERROR")
    print(f"Could not find 'gui.main_window'.")
    print(f"Python is looking in: {sys.path[0]}")
    print(f"Detailed error: {e}")
    sys.exit(1)


def main():
    """Main application entry point"""
    # Enable high DPI scaling for modern monitors
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    
    app = QApplication(sys.argv)
    app.setApplicationName("Scientific Hypertrophy Trainer")
    app.setOrganizationName("Hypertrophy AI")
    
    try:
        # Create and show main window
        window = MainWindow()
        window.show()
        
        # Start event loop
        sys.exit(app.exec())
    except Exception as e:
        print(f"❌ CRASHED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()