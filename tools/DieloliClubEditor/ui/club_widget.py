from PySide6.QtWidgets import QWidget, QTabWidget, QVBoxLayout, QPushButton
from ui.club_info_widget import ClubInfoWidget
from ui.club_activity_widget import ClubActivityWidget


class ClubWidget(QWidget):
    """ 社团编辑面板 """

    def __init__(self):
        super().__init__()
        tabs = QTabWidget()
        info = ClubInfoWidget()
        tabs.addTab(info, "社团信息")
        activity = ClubActivityWidget()
        tabs.addTab(activity, "活动信息")
        main_layout = QVBoxLayout()
        main_layout.addWidget(tabs)
        self.setLayout(main_layout)
