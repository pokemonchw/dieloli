from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFrame
import cache_control
from ui.suit_wear_menu import SuitWearMenu


class SuitWearItemWidget(QWidget):
    """
    套装穿戴位置设定组件
    Keyword arguments:
    position -- 穿戴位置
    """

    def __init__(self, position: int):
        super().__init__()
        line_frame = QFrame(self)
        line_frame.setFrameShape(QFrame.HLine)
        line_frame.setFrameShadow(QFrame.Sunken)
        line_frame.setLineWidth(1)
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(line_frame)
        self.position = position
        self.position_label = QLabel(cache_control.wear_type_data[position] + ":未指定")
        main_layout.addWidget(self.position_label)
        self.wear_button = QPushButton(self)
        main_layout.addWidget(self.wear_button)
        self.setLayout(main_layout)
        cache_control.update_signal.connect(self._update)
        self.wear_button.clicked.connect(self._change_wear)
        self._update()

    def _update(self):
        """ 更新位置对应服装信息 """
        if cache_control.now_suit_id == "":
            self.wear_button.setText("未指定")
            self.wear_button.setDisabled(True)
            self.position_label.setText(cache_control.wear_type_data[self.position] + ":" + "未指定")
            return
        else:
            self.wear_button.setDisabled(False)
        suit_data = cache_control.suit_list_data[cache_control.now_suit_id]
        if self.position in suit_data.clothing_wear and suit_data.clothing_wear[self.position] != "":
            clothing_data = cache_control.clothing_list_data[suit_data.clothing_wear[self.position]]
            self.wear_button.setText(clothing_data.name)
            self.position_label.setText(cache_control.wear_type_data[self.position] + ":" + clothing_data.name)
        else:
            self.wear_button.setText("未指定")
            self.position_label.setText(cache_control.wear_type_data[self.position] + ":" + "未指定")

    def _change_wear(self):
        suit_wear_menu = SuitWearMenu(self.position)
        suit_wear_menu.exec()


class SuitWearWidget(QWidget):
    """ 套装穿戴位置设定面板 """

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        for key in cache_control.wear_type_data:
            now_item = SuitWearItemWidget(key)
            layout.addWidget(now_item, 1)
        self.setLayout(layout)
