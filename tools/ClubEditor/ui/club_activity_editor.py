from PySide6.QtWidgets import QWidget, QVBoxLayout, QFormLayout, QLineEdit, QTextEdit, QComboBox

class ClubActivityEditor(QWidget):
    """ 社团活动信息编辑面板 """

    def __init__(self):
        super().__init__()
        main_layout = QVBoxLayout(self)
        form_layout = QFormLayout()
        self.activity_name = QLineEdit()
        self.activity_localtion = QLineEdit()
        form_layout.addRow("活动名称:", self.activity_name)
        form_layout.addRow("活动地点:", self.activity_localtion)
        self.description_combo = QComboBox()
        self.description_combo.addItems(["唱歌","跳舞","篮球"])

