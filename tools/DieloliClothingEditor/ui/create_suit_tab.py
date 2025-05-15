from PySide6.QtWidgets import QWidget, QHBoxLayout
from ui.suit_list_group import SuitListGroup
from ui.suit_info_widget import SuitInfoWidget


class CreateSuitTab(QWidget):
    """ 套装列表编辑 """

    def __init__(self):
        super().__init__()
        layout = QHBoxLayout(self)
        suit_list_group = SuitListGroup()
        suit_info_widget = SuitInfoWidget()
        layout.addWidget(suit_list_group, 1)
        layout.addWidget(suit_info_widget, 5)
        self.setLayout(layout)
