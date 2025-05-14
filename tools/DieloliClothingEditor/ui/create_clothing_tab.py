from PySide6.QtWidgets import QWidget, QHBoxLayout
from ui.clothing_list_group import ClothingListGroup


class CreateClothingTab(QWidget):
    """ 服装列表编辑 """

    def __init__(self):
        super().__init__()
        layout = QHBoxLayout(self)
        clothing_list_group = ClothingListGroup()
        layout.addWidget(clothing_list_group, 1)
        self.setLayout(layout)
