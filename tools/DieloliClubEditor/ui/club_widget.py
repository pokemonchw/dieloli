from PySide6.QtWidgets import QWidget, QTabWidget
from ui.club_info_widget import ClubInfoWidget
from ui.club_active_widget import ClubActiveWidget


class ClubWidget(QWidget):
    """ 社团面板 """

    def __init__(self):
        """ 初始化ui """
        super().__init__()
        self.tabs = QTabWidget()
        self.info = ClubInfoWidget()
        self.active = ClubActiveWidget()
        self.tabs.addTab(self.info, "社团信息")
        self.tabs.addTab(self.active, "活动信息")
