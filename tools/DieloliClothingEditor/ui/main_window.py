from PySide6.QtWidgets import QMainWindow
from ui.main_tabs import MainTabs
from ui.menu_bar import MenuBar


class MainWindow(QMainWindow):
    """ 编辑器主窗口 """

    def __init__(self):
        super().__init__()
        self.setMenuBar(MenuBar())
        self.setCentralWidget(MainTabs())

