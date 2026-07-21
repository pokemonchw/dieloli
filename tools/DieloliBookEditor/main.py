import sys
import os
import uuid
import json
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QListWidget, QPushButton, QLabel, QLineEdit, QSplitter,
    QMessageBox, QFileDialog, QGroupBox, QTreeWidget, QTreeWidgetItem, QDialog, QComboBox,
    QTextEdit
)
from PySide6.QtCore import Qt
from typing import Dict, Set

# --- 数据定义 ---
settle_data: Dict[str, str] = {}
settle_type_data: Dict[str, Dict[str, Set[str]]] = {}
settle_tree_state: Dict[str, bool] = {}

class BookData:
    def __init__(self):
        self.uid: str = str(uuid.uuid4())
        self.name: str = "新建图书"
        self.info: str = ""
        self.settle_list: list = []
        self.type: int = 0

book_data_dict: Dict[str, BookData] = {}

def load_settle_data():
    if os.path.exists("../../tools/Settle.csv"):
        path = "../../tools/Settle.csv"
    elif os.path.exists("../Settle.csv"):
        path = "../Settle.csv"
    else:
        print("Settle.csv not found!")
        return
    
    with open(path, encoding="utf-8") as f:
        import csv
        reader = csv.DictReader(f)
        for row in reader:
            if "settle_id" in row and "settle_info" in row:
                sid = row["settle_id"]
                settle_data[sid] = row["settle_info"]
                s_type = row["settle_type"]
                settle_type_data.setdefault(s_type, {})
                # Group by first 2 chars of settle_info as sub_group
                info = row["settle_info"]
                sub_group = info[:2] if len(info) >= 2 else "其他"
                settle_type_data[s_type].setdefault(sub_group, set()).add(sid)

# --- UI组件 ---
class TwoLevelGroupedMultiSelectDialog(QDialog):
    def __init__(self, available_items: Dict[str, str], group_mapping: Dict[str, Dict[str, Set[str]]],
                 selected_items: Dict[str, int], title: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(800, 600)
        self.available_items = available_items
        self.group_mapping = group_mapping
        self.selected_items = selected_items.copy()

        layout = QVBoxLayout(self)
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("搜索...")
        layout.addWidget(self.search_edit)

        h_split = QSplitter(Qt.Horizontal)
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        h_split.addWidget(self.tree)
        self.selected_list = QListWidget()
        h_split.addWidget(self.selected_list)
        layout.addWidget(h_split)

        self.tree.itemDoubleClicked.connect(self.handle_item_double_clicked)
        self.selected_list.itemDoubleClicked.connect(self.handle_preview_item_double_clicked)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        self.ok_button = QPushButton("确定")
        self.cancel_button = QPushButton("取消")
        btn_layout.addWidget(self.ok_button)
        btn_layout.addWidget(self.cancel_button)
        layout.addLayout(btn_layout)

        self.search_edit.textChanged.connect(self.update_tree)
        self.tree.itemChanged.connect(self.handle_item_changed)
        self.tree.itemExpanded.connect(self.handle_item_expanded)
        self.tree.itemCollapsed.connect(self.handle_item_collapsed)
        self.ok_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)

        self.update_tree()

    def update_tree(self) -> None:
        filter_text = self.search_edit.text().lower()
        self.tree.blockSignals(True)
        self.tree.clear()
        for big_group, subdict in self.group_mapping.items():
            big_item = QTreeWidgetItem(self.tree)
            big_item.setText(0, big_group)
            big_item.setFlags(big_item.flags() & ~Qt.ItemIsUserCheckable)
            expanded = settle_tree_state.get(big_group, False)
            big_item.setExpanded(expanded)
            for sub_group, id_set in subdict.items():
                filtered_ids = [sid for sid in id_set if filter_text in self.available_items.get(sid, sid).lower()]
                if not filtered_ids:
                    continue
                sub_item = QTreeWidgetItem(big_item)
                sub_item.setText(0, sub_group)
                sub_item.setFlags(sub_item.flags() & ~Qt.ItemIsUserCheckable)
                key = f"{big_group}::{sub_group}"
                expanded_sub = settle_tree_state.get(key, False)
                sub_item.setExpanded(expanded_sub)
                for sid in sorted(filtered_ids):
                    name = self.available_items.get(sid, sid)
                    child = QTreeWidgetItem(sub_item)
                    child.setText(0, name)
                    child.setData(0, Qt.UserRole, sid)
                    child.setFlags(child.flags() | Qt.ItemIsUserCheckable)
                    child.setCheckState(0, Qt.Checked if sid in self.selected_items else Qt.Unchecked)
        self.tree.blockSignals(False)
        self.update_selected_list()

    def handle_item_changed(self, item: QTreeWidgetItem, column: int) -> None:
        if item.data(0, Qt.UserRole) is not None:
            sid = item.data(0, Qt.UserRole)
            if item.checkState(0) == Qt.Checked:
                self.selected_items[sid] = 1
            else:
                self.selected_items.pop(sid, None)
            self.update_selected_list()

    def handle_item_expanded(self, item: QTreeWidgetItem) -> None:
        parent = item.parent()
        if parent is None:
            settle_tree_state[item.text(0)] = True
        else:
            key = f"{parent.text(0)}::{item.text(0)}"
            settle_tree_state[key] = True

    def handle_item_collapsed(self, item: QTreeWidgetItem) -> None:
        parent = item.parent()
        if parent is None:
            settle_tree_state[item.text(0)] = False
        else:
            key = f"{parent.text(0)}::{item.text(0)}"
            settle_tree_state[key] = False

    def handle_item_double_clicked(self, item: QTreeWidgetItem, column: int) -> None:
        sid = item.data(0, Qt.UserRole)
        if sid is not None:
            item.setCheckState(0, Qt.Unchecked if item.checkState(0) == Qt.Checked else Qt.Checked)

    def handle_preview_item_double_clicked(self, item) -> None:
        sid = item.data(Qt.UserRole)
        self.selected_items.pop(sid, None)
        self.update_tree()

    def update_selected_list(self) -> None:
        self.selected_list.clear()
        for sid in self.selected_items:
            name = self.available_items.get(sid, sid)
            list_item = self.selected_list.addItem(name)
            self.selected_list.item(self.selected_list.count() - 1).setData(Qt.UserRole, sid)

    def get_selected(self) -> Dict[str, int]:
        return self.selected_items

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dieloli 图书编辑器")
        self.resize(1000, 600)
        self.current_book_id = None
        
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # Left panel
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        self.book_list = QListWidget()
        self.book_list.currentRowChanged.connect(self.on_book_selected)
        left_layout.addWidget(self.book_list)

        btn_layout = QHBoxLayout()
        self.btn_add = QPushButton("添加图书")
        self.btn_add.clicked.connect(self.add_book)
        self.btn_del = QPushButton("删除图书")
        self.btn_del.clicked.connect(self.del_book)
        btn_layout.addWidget(self.btn_add)
        btn_layout.addWidget(self.btn_del)
        left_layout.addLayout(btn_layout)

        file_btn_layout = QHBoxLayout()
        self.btn_load = QPushButton("加载 default.json")
        self.btn_load.clicked.connect(self.load_data)
        self.btn_save = QPushButton("保存 default.json")
        self.btn_save.clicked.connect(self.save_data)
        file_btn_layout.addWidget(self.btn_load)
        file_btn_layout.addWidget(self.btn_save)
        left_layout.addLayout(file_btn_layout)

        main_layout.addWidget(left_panel, 1)

        # Right panel
        self.right_panel = QWidget()
        right_layout = QVBoxLayout(self.right_panel)
        
        form_layout = QHBoxLayout()
        form_layout.addWidget(QLabel("图书名字:"))
        self.name_edit = QLineEdit()
        self.name_edit.textChanged.connect(self.on_name_changed)
        form_layout.addWidget(self.name_edit)
        right_layout.addLayout(form_layout)

        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("图书分类:"))
        self.type_combo = QComboBox()
        self.type_combo.addItems([
            "小学一年级教材", "小学二年级教材", "小学三年级教材", "小学四年级教材", "小学五年级教材", "小学六年级教材",
            "初中一年级教材", "初中二年级教材", "初中三年级教材",
            "高中一年级教材", "高中二年级教材", "高中三年级教材",
        ])
        self.type_combo.currentIndexChanged.connect(self.on_type_changed)
        type_layout.addWidget(self.type_combo)
        right_layout.addLayout(type_layout)

        info_layout = QHBoxLayout()
        info_layout.addWidget(QLabel("书籍信息:"))
        self.info_edit = QTextEdit()
        self.info_edit.textChanged.connect(self.on_info_changed)
        info_layout.addWidget(self.info_edit)
        right_layout.addLayout(info_layout)

        settle_group = QGroupBox("阅读经验(结算器)")
        s_layout = QVBoxLayout(settle_group)
        self.settle_list_widget = QListWidget()
        self.btn_edit_settle = QPushButton("编辑结算器")
        self.btn_edit_settle.clicked.connect(self.edit_settle)
        s_layout.addWidget(self.settle_list_widget)
        s_layout.addWidget(self.btn_edit_settle)
        right_layout.addWidget(settle_group)

        main_layout.addWidget(self.right_panel, 3)
        self.right_panel.setEnabled(False)

        load_settle_data()

    def update_book_list(self):
        self.book_list.clear()
        for uid, book in book_data_dict.items():
            self.book_list.addItem(book.name)
            self.book_list.item(self.book_list.count()-1).setData(Qt.UserRole, uid)

    def add_book(self):
        b = BookData()
        book_data_dict[b.uid] = b
        self.update_book_list()
        self.book_list.setCurrentRow(self.book_list.count() - 1)

    def del_book(self):
        if not self.current_book_id:
            return
        del book_data_dict[self.current_book_id]
        self.current_book_id = None
        self.update_book_list()

    def on_book_selected(self, row):
        if row < 0:
            self.right_panel.setEnabled(False)
            self.current_book_id = None
            return
        self.right_panel.setEnabled(True)
        uid = self.book_list.item(row).data(Qt.UserRole)
        self.current_book_id = uid
        book = book_data_dict[uid]
        
        self.name_edit.blockSignals(True)
        self.name_edit.setText(book.name)
        self.name_edit.blockSignals(False)
        
        self.info_edit.blockSignals(True)
        self.info_edit.setPlainText(book.info)
        self.info_edit.blockSignals(False)
        
        self.type_combo.blockSignals(True)
        self.type_combo.setCurrentIndex(book.type)
        self.type_combo.blockSignals(False)
        
        self.update_settle_list_ui()

    def on_name_changed(self, text):
        if self.current_book_id:
            book_data_dict[self.current_book_id].name = text
            # update list name
            row = self.book_list.currentRow()
            if row >= 0:
                self.book_list.item(row).setText(text)

    def on_info_changed(self):
        if self.current_book_id:
            book_data_dict[self.current_book_id].info = self.info_edit.toPlainText()

    def on_type_changed(self, index):
        if self.current_book_id:
            book_data_dict[self.current_book_id].type = index

    def update_settle_list_ui(self):
        self.settle_list_widget.clear()
        if not self.current_book_id:
            return
        book = book_data_dict[self.current_book_id]
        for sid in book.settle_list:
            name = settle_data.get(sid, sid)
            self.settle_list_widget.addItem(name)

    def edit_settle(self):
        if not self.current_book_id:
            return
        book = book_data_dict[self.current_book_id]
        current_dict = {sid: 1 for sid in book.settle_list}
        dlg = TwoLevelGroupedMultiSelectDialog(settle_data, settle_type_data, current_dict, "编辑结算器", self)
        if dlg.exec():
            selected = dlg.get_selected()
            book.settle_list = list(selected.keys())
            self.update_settle_list_ui()

    def load_data(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择 default.json", ".", "JSON Files (*.json)")
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            book_data_dict.clear()
            for k, v in data.items():
                b = BookData()
                b.uid = v.get("uid", b.uid)
                b.name = v.get("name", "未命名图书")
                b.info = v.get("info", "")
                b.settle_list = v.get("settle_list", [])
                b.type = v.get("type", 0)
                book_data_dict[b.uid] = b
            self.update_book_list()
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载失败: {e}")

    def save_data(self):
        path, _ = QFileDialog.getSaveFileName(self, "保存 default.json", "default.json", "JSON Files (*.json)")
        if not path:
            return
        try:
            out_data = {}
            for uid, book in book_data_dict.items():
                out_data[uid] = {
                    "uid": uid,
                    "name": book.name,
                    "info": book.info,
                    "type": book.type,
                    "settle_list": book.settle_list
                }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(out_data, f, ensure_ascii=False, indent=2)
            QMessageBox.information(self, "成功", "保存成功！")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存失败: {e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
