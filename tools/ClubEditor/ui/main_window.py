from PySide6.QtWidgets import QMainWindow, QWidget, QHBoxLayout
from ui.club_list_group import ClubListGroup
from ui.menu_bar import MenuBar
from ui.club_widget import ClubWidget


class MainWindow(QMainWindow):
    """ 编辑器主窗口 """

    def __init__(self):
        super().__init__()
        menu_bar = MenuBar()
        self.setMenuBar(menu_bar)
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        club_list_group = ClubListGroup()
        main_layout.addWidget(club_list_group)
        club_widget = ClubWidget()
        main_layout.addWidget(club_widget)
        self.setLayout(main_layout)

