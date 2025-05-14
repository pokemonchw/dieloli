#!/usr/bin/python3
import sys
from PySide6.QtWidgets import QApplication
from ui.main_window import MainWindow
import load_csv
import map_config

map_config.init_map_data()
load_csv.load_config()
app = QApplication(sys.argv)
main_window = MainWindow()
main_window.show()
sys.exit(app.exec())
