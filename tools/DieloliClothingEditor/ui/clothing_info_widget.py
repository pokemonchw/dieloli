from PySide6.QtWidgets import QLineEdit, QWidget, QVBoxLayout, QLabel, QTextEdit
from PySide6.QtCore import QEvent
import cache_control
from ui.clothing_gender_type_combo import ClothingGenderTypeCombo
from ui.clothing_wear_type_combo import ClothingWearTypeCombo


class ClothingNameEdit(QLineEdit):
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


class ClothingDescribeEdit(QTextEdit):
    """ 服装描述输入框 """

    def __init__(self):
        super().__init__()
        self.installEventFilter(self)
        cache_control.update_signal.connect(self._update)
        self._update()

    def _update(self):
        if cache_control.now_clothing_id in cache_control.clothing_list_data:
            clothing_data = cache_control.clothing_list_data[cache_control.now_clothing_id]
            self.setPlainText(clothing_data.describe)

    def eventFilter(self, obj, event):
        if event.type() == QEvent.FocusOut:
            if cache_control.now_clothing_id != "":
                if cache_control.now_clothing_id in  cache_control.clothing_list_data:
                    clothing_data = cache_control.clothing_list_data[cache_control.now_clothing_id]
                    clothing_data.describe = self.toPlainText()
                    cache_control.update_signal.emit()
        return super().eventFilter(obj, event)


class ClothingInfoWidget(QWidget):
    """ 服装信息面板 """

    def __init__(self):
        super().__init__()
        head_widget = QWidget()
        head_layout = QVBoxLayout(self)
        head_layout.addWidget(QLabel("服装名称:"))
        self.clothing_name_edit = ClothingNameEdit()
        head_layout.addWidget(self.clothing_name_edit)
        head_layout.addWidget(QLabel("性别设置:"))
        self.gender_type_combo = ClothingGenderTypeCombo()
        head_layout.addWidget(self.gender_type_combo)
        head_layout.addWidget(QLabel("穿戴位置:"))
        self.wear_type_combo = ClothingWearTypeCombo()
        head_layout.addWidget(self.wear_type_combo)
        head_widget.setLayout(head_layout)
        describe_widget = QWidget()
        describe_layout = QVBoxLayout(self)
        describe_layout.addWidget(QLabel("服装描述:"))
        self.clothing_describe_edit = ClothingDescribeEdit()
        describe_layout.addWidget(self.clothing_describe_edit)
        describe_widget.setLayout(describe_layout)
        main_layout = QVBoxLayout()
        main_layout.addWidget(head_widget, 1)
        main_layout.addWidget(describe_widget,4)
        self.setLayout(main_layout)
