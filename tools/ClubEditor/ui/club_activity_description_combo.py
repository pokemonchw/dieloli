from PySide6.QtWidgets import (
    QDialog, QTableWidget, QTableWidgetItem, QVBoxLayout, QHeaderView, QLineEdit,
    QWidget
)
from PySide6.QtCore import Signal, Qt
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


class ClubActivityDescriptionCombo(QWidget):
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
        if cache_control.now_activity_id == "":
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
        if cache_control.now_activity_id == "":
            if len(club_data.activity_list):
                cache_control.now_activity_id = list(club_data.activity_list.keys())[0]
        if cache_control.now_activity_id == "":
            return
        activity_data = club_data.activity_list[cache_control.now_activity_id]
        self.lineEdit.setText(cache_control.activity_list[activity_data.description])
