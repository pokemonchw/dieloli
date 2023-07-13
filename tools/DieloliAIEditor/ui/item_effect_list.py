from PySide6.QtWidgets import QListWidgetItem, QListWidget, QWidget, QVBoxLayout, QLabel
from PySide6.QtGui import QFont
import cache_control


class ItemEffectList(QWidget):
    """效果表单主体"""

    def __init__(self):
        """初始化前提表单主体"""
        super(ItemEffectList, self).__init__()
        self.font = QFont()
        self.font.setPointSize(16)
        self.setFont(self.font)
        layout = QVBoxLayout()
        label = QLabel()
        label.setText("效果列表")
        layout.addWidget(label)
        self.item_list = QListWidget()
        layout.addWidget(self.item_list)
        self.setLayout(layout)

    def update(self):
        """更新前提列表"""
        self.item_list.clear()
        if cache_control.now_target_id == "":
            return
        for premise in cache_control.now_target_data[cache_control.now_target_id].effect:
            item = QListWidgetItem(cache_control.premise_data[premise])
            item.setToolTip(item.text())
            self.item_list.addItem(item)
