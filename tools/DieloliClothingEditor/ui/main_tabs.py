from PySide6.QtWidgets import QTabWidget
from ui.create_clothing_tab import CreateClothingTab
from ui.create_suit_tab import CreateSuitTab

class MainTabs(QTabWidget):
    """ 编辑器标签页控件 """

    def __init__(self):
        super().__init__()
        self.create_clothing_tab = CreateClothingTab()
        self.create_suit_tab = CreateSuitTab()
        self.addTab(self.create_clothing_tab, "服装列表")
        self.addTab(self.create_suit_tab, "套装列表")
