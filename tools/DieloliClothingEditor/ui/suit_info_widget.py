from PySide6.QtWidgets import QWidget, QLineEdit, QVBoxLayout, QLabel
from PySide6.QtCore import QEvent
import cache_control
from ui.suit_wear_widget import SuitWearWidget


class SuitNameEdit(QLineEdit):
    """ 套装名称输入框 """

    def __init__(self):
        super().__init__()
        self.installEventFilter(self)
        cache_control.update_signal.connect(self._update)
        self._update()

    def _update(self):
        if cache_control.now_suit_id in cache_control.suit_list_data:
            suit_data = cache_control.suit_list_data[cache_control.now_suit_id]
            self.setText(suit_data.name)

    def eventFilter(self, obj, event):
        if event.type() == QEvent.FocusOut:
            if cache_control.now_suit_id != "":
                if cache_control.now_suit_id in  cache_control.suit_list_data:
                    suit_data = cache_control.suit_list_data[cache_control.now_suit_id]
                    suit_data.name = self.text()
                    cache_control.update_signal.emit()
        return super().eventFilter(obj, event)


class SuitInfoWidget(QWidget):
    """ 套装信息面板 """

    def __init__(self):
        super().__init__()
        head_widget = QWidget()
        head_layout = QVBoxLayout()
        head_layout.addWidget(QLabel("套装名称:"))
        self.suit_name_edit = SuitNameEdit()
        head_layout.addWidget(self.suit_name_edit)
        head_widget.setLayout(head_layout)
        main_layout = QVBoxLayout()
        main_layout.addWidget(head_widget)
        main_layout.addWidget(SuitWearWidget())
        self.setLayout(main_layout)
