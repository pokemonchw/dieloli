from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QLineEdit, QTextEdit, QComboBox,
    QTableWidget, QDialog, QTableWidgetItem, QHBoxLayout, QPushButton, QHeaderView
)
from PySide6.QtGui import QStandardItemModel, QStandardItem
from PySide6.QtCore import Qt, Signal, QEvent
from ui.club_activity_time_editor import ClubActivityTimeEditor
import cache_control


class TableSelectorDialog(QDialog):
    def __init__(self, data, selectedText, parent=None):
        super().__init__(parent)
        self.table = QTableWidget(self)
        self.num_cols = 3
        self.selectedText = selectedText
        self.initUI(data)
        self.adjustSizeAndLayout()

    def initUI(self, data):
        num_rows = (len(data) + self.num_cols - 1) // self.num_cols
        self.table.setRowCount(num_rows)
        self.table.setColumnCount(self.num_cols)

        for i, item in enumerate(data):
            row, col = divmod(i, self.num_cols)
            tableItem = QTableWidgetItem(item)
            self.table.setItem(row, col, tableItem)
            # 判断并设置初始选中项
            if item == self.selectedText:
                self.table.setCurrentCell(row, col)

        layout = QVBoxLayout()
        layout.addWidget(self.table)
        self.setLayout(layout)

        self.table.cellClicked.connect(self.onCellClicked)

    def adjustSizeAndLayout(self):
        self.resize(400, 300)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

    def onCellClicked(self, row, column):
        item = self.table.item(row, column)
        if item:
            self.selectedValue = item.text()
            self.accept()

class ClickableLineEdit(QLineEdit):
    clicked = Signal()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setReadOnly(True)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)

class CustomComboBox(QWidget):
    def __init__(self, data, parent=None):
        super().__init__(parent)
        self.data = data
        self.initUI()
        cache_control.update_signal.connect(self._update)

    def initUI(self):
        self.layout = QVBoxLayout(self)
        self.lineEdit = ClickableLineEdit(self)
        self.lineEdit.clicked.connect(self.onLineEditClicked)
        self.layout.addWidget(self.lineEdit)

    def onLineEditClicked(self):
        if cache_control.now_club_id == "":
            return
        if cache_control.now_activity_id =="":
            club_data = cache_control.club_list_data[cache_control.now_club_id]
            if len(club_data.activity_list):
                cache_control.now_activity_id = list(club_data.activity_list.keys())[0]
        if cache_control.now_activity_id == "":
            return
        currentText = self.lineEdit.text()
        selector = TableSelectorDialog(self.data, currentText, self)
        if selector.exec():
            self.lineEdit.setText(selector.selectedValue)

    def _update(self):
        if cache_control.now_club_id == "":
            return
        club_data = cache_control.club_list_data[cache_control.now_club_id]
        if cache_control.now_activity_id =="":
            if len(club_data.activity_list):
                cache_control.now_activity_id = list(club_data.activity_list.keys())[0]
        if cache_control.now_activity_id == "":
            return
        activity_data = club_data.activity_list[cache_control.now_activity_id]
        self.lineEdit.setText(cache_control.activity_list[activity_data.description])


class ActivityNameEdit(QLineEdit):
    """ 活动名称输入框 """

    def __init__(self):
        """ 初始化输入框 """
        super().__init__()
        self.installEventFilter(self)
        cache_control.update_signal.connect(self._update)
        self._update()
        self.inside = False

    def _update(self):
        if cache_control.now_club_id == "":
            if len(cache_control.club_list_data):
                cache_control.now_club_id = list(cache_control.club_list_data.keys())[0]
        if cache_control.now_club_id == "":
            return
        club_data = cache_control.club_list_data[cache_control.now_club_id]
        if cache_control.now_activity_id == "":
            if len(club_data.activity_list):
                cache_control.now_activity_id = list(club_data.activity_list.keys())[0]
        if cache_control.now_activity_id == "":
            return
        activity_data = club_data.activity_list[cache_control.now_activity_id]
        self.setText(activity_data.name)

    def eventFilter(self, obj, event):
        # 检测焦点丢失事件
        print(event.type())
        now_judge:
        if event.type() == QEvent.MouseMove:
            # 更新鼠标是否在QLineEdit内的状态
            self.inside = obj.rect().contains(event.pos())
        if event.type() == QEvent.FocusOut:
            now_judge = True
        if now_judge:
            if cache_control.now_club_id == "":
                if len(cache_control.club_list_data):
                    cache_control.now_club_id = list(cache_control.club_list_data.keys())[0]
            if cache_control.now_club_id == "":
                return
            club_data = cache_control.club_list_data[cache_control.now_club_id]
            if cache_control.now_activity_id == "":
                if len(club_data.activity_list):
                    cache_control.now_activity_id = list(club_data.activity_list.keys())[0]
            if cache_control.now_activity_id == "":
                return
            activity_data = club_data.activity_list[cache_control.now_activity_id]
            activity_data.name = self.text()
            cache_control.update_signal.emit()
        return super().eventFilter(obj, event)


class ClubActivityEditor(QWidget):
    """ 社团活动信息编辑面板 """

    def __init__(self):
        super().__init__()
        main_layout = QVBoxLayout(self)
        form_layout = QFormLayout()
        self.activity_name = ActivityNameEdit()
        self.activity_localtion = QLineEdit()
        form_layout.addRow("活动名称:", self.activity_name)
        form_layout.addRow("活动地点:", self.activity_localtion)
        self.description_combo = CustomComboBox(list(cache_control.activity_list_data.keys()))
        form_layout.addRow("活动内容:", self.description_combo)
        main_layout.addLayout(form_layout)
        activity_time = ClubActivityTimeEditor()
        main_layout.addWidget(activity_time)
        self.setLayout(main_layout)

