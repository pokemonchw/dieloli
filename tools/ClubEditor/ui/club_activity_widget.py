from PySide6.QtWidgets import QWidget, QHBoxLayout
from ui.club_activity_list import ClubActivityList

class ClubActivityWidget(QWidget):
    """ 社团活动设置面板 """

    def __init__(self):
        super().__init__()
        main_layout = QHBoxLayout(self)
        activity_list = ClubActivityList()
        main_layout.addWidget(activity_list)

