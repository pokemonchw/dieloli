from PySide6.QtWidgets import QWidget, QTabWidget, QVBoxLayout
from ui.club_info_widget import ClubInfoWidget


class ClubWidget(QWidget):
    """ 社团编辑面板 """

    def __init__(self):
        super().__init__()
        tabs = QTabWidget()
        info = ClubInfoWidget()
        tabs.addTab(info, "社团信息")
        main_layout = QVBoxLayout()
        main_layout.addWidget(tabs)
        self.setLayout(main_layout)
