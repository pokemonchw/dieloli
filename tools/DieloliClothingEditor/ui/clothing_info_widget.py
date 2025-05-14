from PySide6.QtWidgets import QLineEdit
from PySide6.QtCore import QEvent
import cache_control


class ClothingNameEdi(QLineEdit):
    """ 服装名称输入框 """

    def __init__(self):
        super().__init__()
        self.installEventFilter(self)
        cache_control.update_signal.connect(self._update)
        self._update()

    def _update(self):
        if cache_control.now_clothing_id in cache_control.clothing_list_data:
            clothing_data = cache_control.clothing_list_data[cache_control.now_clothing_id]
            self.setText(clothing_data.name)

    def eventFilter(self, obj, event):
        if event.type() == QEvent.FocusOut:
            if cache_control.now_clothing_id != "":
                if cache_control.now_clothing_id in  cache_control.clothing_list_data:
                    clothing_data = cache_control.clothing_list_data[cache_control.now_clothing_id]
                    clothing_data.name = self.text()
                    cache_control.update_signal.emit()
        return super().eventFilter(obj, event)
