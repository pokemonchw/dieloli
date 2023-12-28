import sys
from PySide6.QtWidgets import QApplication
from ui.main_window import MainWindow
import load_csv

load_csv.load_config()
app = QApplication(sys.argv)
main_window = MainWindow()
main_window.show()
sys.exit(app.exec())
