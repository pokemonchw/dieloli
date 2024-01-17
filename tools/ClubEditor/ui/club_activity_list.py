from PySide6.QtWidgets import QGroupBox, QVBoxLayout, QListWidget

class ClubActivityList(QGroupBox):
    """ 社团活动列表面板 """

    def __init__(self):
        super().__init__("活动列表")
        main_layout = QVBoxLayout(self)
        self.activity_list = QListWidget()
        main_layout.addWidget(self.activity_list)

    def _update(self):
        pass

