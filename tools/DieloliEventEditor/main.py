#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import os
import json
import csv
import uuid
from typing import Optional, Dict, Set, Any
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QListWidget, QListWidgetItem, QTextEdit,
    QSplitter, QTabWidget, QPushButton, QFileDialog,
    QMenuBar, QStatusBar, QMessageBox, QTreeWidget,
    QTreeWidgetItem, QDialog, QLabel, QLineEdit,
    QMenu, QSizePolicy, QGroupBox, QComboBox
)
from PySide6.QtCore import Qt, Signal, QObject
from PySide6.QtGui import QFontMetrics


premise_data: Dict[str, str] = {}
""" 前提配置数据 {前提id:前提描述} """
premise_type_data: Dict[str, Set[str]] = {}
""" 前提类型配置数据 {前提类型:{前提id集合}} """
status_data: Dict[str, str] = {}
""" 状态机配置数据 {状态机id:状态机描述} """
settle_data: Dict[str, str] = {}
""" 结算器配置数据 {结算器id:结算器描述} """
settle_type_data: Dict[str, Dict[str, Set[str]]] = {}
""" 结算器类型配置数据 {结算器类型:{结算器子类:{结算器id集合}}} """
macro_data: Dict[str, str] = {}
""" 文本宏配置数据 {宏id:宏描述} """


def load_config() -> None:
    """ 载入前提、状态、结算器和宏配置数据 """
    if os.path.exists("../premise.csv"):
        with open("../premise.csv", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cid = row["cid"]
                premise_data[cid] = row["premise"]
                p_type = row["premise_type"]
                premise_type_data.setdefault(p_type, set()).add(cid)
    if os.path.exists("../Status.csv"):
        with open("../Status.csv", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                status_data[row["cid"]] = row["status"]
    if os.path.exists("../Settle.csv"):
        with open("../Settle.csv", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sid = row["settle_id"]
                settle_data[sid] = row["settle_info"]
                s_type = row["settle_type"]
                son_type = row["son_type"]
                settle_type_data.setdefault(
                    s_type, {}
                ).setdefault(son_type, set()).add(sid)
    if os.path.exists("../Macro.csv"):
        with open("../Macro.csv", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                macro_data[row["cid"]] = row["info"]


class Event:
    """
    事件对象，保存事件的各项数据。
    Attributes:
        uid (str): 事件唯一ID。
        adv_id (str): 事件所属广告ID（备用）。
        status_id (str): 事件所属状态ID。
        start (bool): 是否为开始事件（True 为开始，False 为结束）。
        text (str): 事件文本描述。
        premise (dict): 事件前提数据（cid -> 1）。
        settle (dict): 事件结算器数据（settle_id -> 1）。
    """
    def __init__(self) -> None:
        self.uid: str = ""
        self.adv_id: str = ""
        self.status_id: str = ""
        self.start: bool = False
        self.text: str = ""
        self.premise: Dict[str, int] = {}
        self.settle: Dict[str, int] = {}


class EventManager(QObject):
    """ 事件管理器，负责载入、保存、添加和删除事件 """
    event_updated = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.events: Dict[str, Event] = {}
        self.current_event_id: Optional[str] = None

    def load_events(self, file_path: str) -> None:
        """ 从指定文件中载入事件数据 """
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.events.clear()
        for uid, edict in data.items():
            event = Event()
            event.__dict__ = edict
            self.events[uid] = event
        self.event_updated.emit()

    def save_events(self, file_path: str) -> None:
        """ 将事件数据保存到指定文件中 """
        data = {uid: event.__dict__ for uid, event in self.events.items()}
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        QMessageBox.information(None, "提示", "保存成功")

    def add_event(self, text: str = "新事件") -> str:
        """
        添加一个新事件，并自动选中该事件
        Returns:
            str: 新事件的 uid。
        """
        uid = str(uuid.uuid4())
        event = Event()
        event.uid = uid
        event.text = text
        if main_window:
            current_state = main_window.state_filter_widget.list_widget.currentItem()
            if current_state is not None:
                selected_state = current_state.data(Qt.UserRole)
                if selected_state is not None:
                    event.status_id = selected_state
            else:
                event.status_id = list(status_data.keys())[0] if status_data else ""
            current_type = main_window.event_type_combo.currentData()
            event.start = current_type if current_type is not None else True
        else:
            event.status_id = list(status_data.keys())[0] if status_data else ""
            event.start = True
        self.events[uid] = event
        self.current_event_id = uid
        self.event_updated.emit()
        return uid

    def delete_event(self, uid: str) -> None:
        """
        删除指定 uid 的事件。
        """
        if uid in self.events:
            del self.events[uid]
            if self.current_event_id == uid:
                self.current_event_id = None
            self.event_updated.emit()


premise_tree_state: Dict[str, bool] = {}
settle_tree_state: Dict[str, bool] = {}


class MacroTextEdit(QTextEdit):
    """
    支持右键菜单插入宏的文本编辑器。
    """
    def contextMenuEvent(self, event: Any) -> None:
        menu = self.createStandardContextMenu()
        insert_macro_menu = menu.addMenu("插入宏")
        for cid, info in macro_data.items():
            action_text = "{" + cid + "}: " + info
            action = insert_macro_menu.addAction(action_text)
            action.triggered.connect(lambda checked, cid=cid: self.insert_macro(cid))
        menu.exec(event.globalPos())

    def insert_macro(self, cid: str) -> None:
        """
        在当前光标位置插入指定宏。
        """
        cursor = self.textCursor()
        cursor.insertText("{" + cid + "}")


class MacroInfoTextEdit(QTextEdit):
    """
    只读宏信息显示编辑器，显示所有宏信息，每行一条；双击可将对应宏插入主编辑器。
    """
    def __init__(self, main_text_edit: MacroTextEdit, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.main_text_edit = main_text_edit
        self.setReadOnly(True)

    def mouseDoubleClickEvent(self, event: Any) -> None:
        cursor = self.cursorForPosition(event.pos())
        block = cursor.block()
        line_text = block.text().strip()
        if line_text:
            if line_text.startswith("{") and "}" in line_text:
                macro = line_text[1:line_text.index("}")]
                main_cursor = self.main_text_edit.textCursor()
                main_cursor.insertText("{" + macro + "}")
        super().mouseDoubleClickEvent(event)


class StateFilterWidget(QWidget):
    """
    状态筛选面板，显示所有状态并支持搜索过滤，固定宽度由状态中最长文本确定。
    """
    stateSelected = Signal(object)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("搜索状态...")
        layout.addWidget(self.search_edit)
        self.list_widget = QListWidget()
        layout.addWidget(self.list_widget)
        self.all_items = []
        for state_id, state_name in status_data.items():
            item = QListWidgetItem(state_name)
            item.setData(Qt.UserRole, state_id)
            self.list_widget.addItem(item)
            self.all_items.append(item)
        fm = QFontMetrics(self.list_widget.font())
        max_width = max(fm.horizontalAdvance(item.text()) for item in self.all_items) + 40
        self.list_widget.setFixedWidth(max_width)
        self.search_edit.setFixedWidth(max_width)
        self.search_edit.textChanged.connect(self.filter_list)
        self.list_widget.itemClicked.connect(self.on_item_clicked)

    def filter_list(self, text: str) -> None:
        """
        过滤状态列表，显示包含搜索文本的状态。
        """
        text = text.lower()
        for item in self.all_items:
            if item == self.all_items[0]:
                item.setHidden(False)
            else:
                item.setHidden(text not in item.text().lower())

    def on_item_clicked(self, item: QListWidgetItem) -> None:
        """
        当选中状态时发射信号。
        """
        self.stateSelected.emit(item.data(Qt.UserRole))


class GroupedMultiSelectDialog(QDialog):
    """
    分组多选对话框（前提选择）
    根据 group_mapping 显示可选项和已选项（水平并列），支持双击切换勾选及预览列表双击取消选择。
    """
    def __init__(self, available_items: Dict[str, str],
                 group_mapping: Dict[str, Set[str]],
                 selected_items: Dict[str, int],
                 title: str,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(800, 600)
        self.available_items = available_items
        self.group_mapping = group_mapping
        self.selected_items = selected_items
        self.init_ui()

    def init_ui(self) -> None:
        layout = QVBoxLayout(self)
        search_layout = QHBoxLayout()
        search_label = QLabel("搜索:")
        self.search_edit = QLineEdit()
        search_layout.addWidget(search_label)
        search_layout.addWidget(self.search_edit)
        layout.addLayout(search_layout)
        h_split = QSplitter(Qt.Horizontal)
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setColumnCount(1)
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
        for group, id_set in self.group_mapping.items():
            filtered_ids = [key for key in id_set if filter_text in self.available_items.get(key, key).lower()]
            if not filtered_ids:
                continue
            group_item = QTreeWidgetItem(self.tree)
            group_item.setText(0, group)
            group_item.setFlags(group_item.flags() & ~Qt.ItemIsUserCheckable)
            expanded = premise_tree_state.get(group, False)
            group_item.setExpanded(expanded)
            for key in sorted(filtered_ids):
                name = self.available_items.get(key, key)
                child_item = QTreeWidgetItem(group_item)
                child_item.setText(0, name)
                child_item.setData(0, Qt.UserRole, key)
                child_item.setFlags(child_item.flags() | Qt.ItemIsUserCheckable)
                child_item.setCheckState(0, Qt.Checked if key in self.selected_items else Qt.Unchecked)
        self.tree.blockSignals(False)
        self.update_selected_list()

    def handle_item_changed(self, item: QTreeWidgetItem, column: int) -> None:
        if item.data(0, Qt.UserRole) is not None:
            key = item.data(0, Qt.UserRole)
            if item.checkState(0) == Qt.Checked:
                self.selected_items[key] = 1
            else:
                self.selected_items.pop(key, None)
            self.update_selected_list()

    def handle_item_expanded(self, item: QTreeWidgetItem) -> None:
        if item.parent() is None:
            premise_tree_state[item.text(0)] = True

    def handle_item_collapsed(self, item: QTreeWidgetItem) -> None:
        if item.parent() is None:
            premise_tree_state[item.text(0)] = False

    def handle_item_double_clicked(self, item: QTreeWidgetItem, column: int) -> None:
        key = item.data(0, Qt.UserRole)
        if key is not None:
            item.setCheckState(0, Qt.Unchecked if item.checkState(0) == Qt.Checked else Qt.Checked)

    def update_selected_list(self) -> None:
        self.selected_list.clear()
        for key in self.selected_items:
            name = self.available_items.get(key, key)
            list_item = QListWidgetItem(name)
            list_item.setData(Qt.UserRole, key)
            self.selected_list.addItem(list_item)

    def handle_preview_item_double_clicked(self, item: QListWidgetItem) -> None:
        key = item.data(Qt.UserRole)
        if key in self.selected_items:
            self.selected_items.pop(key, None)
            self.update_tree()

    def accept(self) -> None:
        def recurse(item: QTreeWidgetItem) -> None:
            for i in range(item.childCount()):
                child = item.child(i)
                key = child.data(0, Qt.UserRole)
                if key is not None:
                    if child.checkState(0) == Qt.Checked:
                        self.selected_items[key] = 1
                    else:
                        self.selected_items.pop(key, None)
                recurse(child)
        root = self.tree.invisibleRootItem()
        for i in range(root.childCount()):
            recurse(root.child(i))
        super().accept()


class TwoLevelGroupedMultiSelectDialog(QDialog):
    """
    二级分组多选对话框（结算器选择）
    根据二级分组显示可选项，下方实时显示已选项（水平并列），
    支持双击切换勾选状态和预览列表双击取消选择。
    """
    def __init__(self, available_items: Dict[str, str],
                 group_mapping: Dict[str, Dict[str, Set[str]]],
                 selected_items: Dict[str, int],
                 title: str,
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(800, 600)
        self.available_items = available_items
        self.group_mapping = group_mapping
        self.selected_items = selected_items
        self.init_ui()

    def init_ui(self) -> None:
        layout = QVBoxLayout(self)
        search_layout = QHBoxLayout()
        search_label = QLabel("搜索:")
        self.search_edit = QLineEdit()
        search_layout.addWidget(search_label)
        search_layout.addWidget(self.search_edit)
        layout.addLayout(search_layout)
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

    def update_selected_list(self) -> None:
        self.selected_list.clear()
        for sid in self.selected_items:
            list_item = QListWidgetItem(self.available_items.get(sid, sid))
            list_item.setData(Qt.UserRole, sid)
            self.selected_list.addItem(list_item)

    def handle_preview_item_double_clicked(self, item: QListWidgetItem) -> None:
        sid = item.data(Qt.UserRole)
        if sid in self.selected_items:
            self.selected_items.pop(sid, None)
            self.update_tree()

    def accept(self) -> None:
        def recurse(item: QTreeWidgetItem) -> None:
            for i in range(item.childCount()):
                child = item.child(i)
                sid = child.data(0, Qt.UserRole)
                if sid is not None:
                    if child.checkState(0) == Qt.Checked:
                        self.selected_items[sid] = 1
                    else:
                        self.selected_items.pop(sid, None)
                recurse(child)
        root = self.tree.invisibleRootItem()
        for i in range(root.childCount()):
            recurse(root.child(i))
        super().accept()


class MainWindow(QMainWindow):
    """
    编辑器主窗体，负责界面布局、事件列表、文本编辑及各模块交互。
    """
    def __init__(self, event_manager: EventManager) -> None:
        super().__init__()
        self.setWindowTitle("新事件编辑器")
        self.event_manager = event_manager

        main_splitter = QSplitter(Qt.Horizontal)

        # 左侧区域：过滤器面板与事件列表并列显示
        left_panel = QWidget()
        left_layout = QHBoxLayout(left_panel)
        # 过滤器面板：垂直布局，包含事件类型下拉框（上方）和状态筛选组件（下方）
        filter_panel = QWidget()
        filter_layout = QVBoxLayout(filter_panel)
        # 先创建状态筛选组件
        self.state_filter_widget = StateFilterWidget()
        self.state_filter_widget.stateSelected.connect(self.on_state_selected)
        # 计算过滤器宽度使用 StateFilterWidget 的公式
        state_width = self.state_filter_widget_width()
        self.event_type_combo = QComboBox()
        self.event_type_combo.addItem("开始事件", True)
        self.event_type_combo.addItem("结束事件", False)
        self.event_type_combo.setFixedWidth(state_width - 5)
        self.event_type_combo.setCurrentIndex(0)
        self.event_type_combo.currentIndexChanged.connect(self.update_event_list)
        filter_layout.addWidget(QLabel("事件类型:"))
        filter_layout.addWidget(self.event_type_combo)
        filter_layout.addWidget(self.state_filter_widget)
        filter_panel.setFixedWidth(state_width)
        left_layout.addWidget(filter_panel)
        # 事件列表
        self.event_list = QListWidget()
        self.event_list.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.event_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.event_list.customContextMenuRequested.connect(self.show_event_list_context_menu)
        left_layout.addWidget(self.event_list, 1)
        main_splitter.addWidget(left_panel)

        # 右侧区域：标签页
        self.tab_widget = QTabWidget()
        main_splitter.addWidget(self.tab_widget)
        # Tab 1：文本编辑
        text_tab = QWidget()
        text_layout = QHBoxLayout(text_tab)
        self.text_edit = MacroTextEdit()
        self.macro_info = MacroInfoTextEdit(self.text_edit)
        self.macro_info.setMaximumWidth(250)
        self.populate_macro_info()
        text_splitter = QSplitter(Qt.Horizontal)
        text_splitter.addWidget(self.text_edit)
        text_splitter.addWidget(self.macro_info)
        text_layout.addWidget(text_splitter)
        self.tab_widget.addTab(text_tab, "文本")
        # Tab 2：前提与效果并列显示
        pe_tab = QWidget()
        pe_layout = QHBoxLayout(pe_tab)
        premise_group = QGroupBox("前提")
        p_layout = QVBoxLayout(premise_group)
        self.premise_list = QListWidget()
        self.edit_premise_button = QPushButton("编辑前提")
        p_layout.addWidget(self.premise_list)
        p_layout.addWidget(self.edit_premise_button)
        settle_group = QGroupBox("效果")
        s_layout = QVBoxLayout(settle_group)
        self.settle_list = QListWidget()
        self.edit_settle_button = QPushButton("编辑结算器")
        s_layout.addWidget(self.settle_list)
        s_layout.addWidget(self.edit_settle_button)
        pe_layout.addWidget(premise_group)
        pe_layout.addWidget(settle_group)
        self.tab_widget.addTab(pe_tab, "前提与效果")

        central_widget = QWidget()
        central_layout = QVBoxLayout(central_widget)
        central_layout.addWidget(main_splitter)
        self.setCentralWidget(central_widget)

        self._create_menu_bar()
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)

        self.event_list.currentRowChanged.connect(self.load_event_detail)
        self.text_edit.textChanged.connect(self.update_event_text)
        self.edit_premise_button.clicked.connect(self.edit_premise)
        self.edit_settle_button.clicked.connect(self.edit_settle)
        self.event_manager.event_updated.connect(self.update_event_list)

        self.update_event_list()

    def state_filter_widget_width(self) -> int:
        """
        计算状态筛选组件的宽度
        Returns:
            int: 过滤器面板建议宽度
        """
        fm = QFontMetrics(self.state_filter_widget.list_widget.font())
        max_width = max(fm.horizontalAdvance(item.text()) for item in self.state_filter_widget.all_items) + 60
        return max_width

    def on_state_selected(self, state_id: Optional[str]) -> None:
        """
        处理状态筛选面板的选中信号
        """
        self.selected_state = state_id
        self.update_event_list()

    def populate_macro_info(self) -> None:
        """
        填充宏信息显示区域，将所有宏信息逐行显示
        """
        text = ""
        for cid, info in macro_data.items():
            text += "{" + cid + "}: " + info + "\n"
        self.macro_info.setPlainText(text)

    def show_event_list_context_menu(self, pos: Any) -> None:
        """
        在事件列表中显示右键菜单，支持新增、复制、删除事件
        """
        item = self.event_list.itemAt(pos)
        menu = QMenu(self.event_list)
        new_action = menu.addAction("新增事件")
        if item is not None:
            copy_action = menu.addAction("复制事件")
            delete_action = menu.addAction("删除事件")
        action = menu.exec_(self.event_list.mapToGlobal(pos))
        if action == new_action:
            self.new_event()
        elif item is not None and action == copy_action:
            self.copy_event(item)
        elif item is not None and action == delete_action:
            self.delete_event_item(item)

    def new_event(self) -> None:
        """
        新增事件，并自动选中、滚动到事件列表底部，
        自动使用当前过滤器（状态和事件类型）的值
        """
        new_uid = self.event_manager.add_event("新事件")
        current_state = self.state_filter_widget.list_widget.currentItem()
        if current_state is not None:
            selected_state = current_state.data(Qt.UserRole)
            if selected_state is not None:
                self.event_manager.events[new_uid].status_id = selected_state
        current_type = self.event_type_combo.currentData()
        if current_type is not None:
            self.event_manager.events[new_uid].start = current_type
        self.statusBar.showMessage("新事件已添加", 2000)
        self.update_event_list()
        self.event_list.scrollToBottom()

    def copy_event(self, item: QListWidgetItem) -> None:
        """
        复制选中的事件，并自动选中、滚动到事件列表底部
        """
        uid = item.data(Qt.UserRole)
        if uid is None:
            return
        original = self.event_manager.events.get(uid)
        if original is None:
            return
        new_event = Event()
        new_event.uid = str(uuid.uuid4())
        new_event.text = original.text + " (复制)"
        new_event.status_id = original.status_id
        new_event.start = original.start
        new_event.premise = original.premise.copy()
        new_event.settle = original.settle.copy()
        self.event_manager.events[new_event.uid] = new_event
        self.event_manager.current_event_id = new_event.uid
        self.event_manager.event_updated.emit()
        self.event_list.scrollToBottom()

    def delete_event_item(self, item: QListWidgetItem) -> None:
        """
        删除选中的事件。
        """
        uid = item.data(Qt.UserRole)
        if uid is None:
            return
        reply = QMessageBox.question(self, "删除事件", "确定删除此事件吗？", QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.event_manager.delete_event(uid)
            self.statusBar.showMessage("事件已删除", 2000)

    def _create_menu_bar(self) -> None:
        """
        创建菜单栏。
        """
        menu_bar = QMenuBar(self)
        file_menu = menu_bar.addMenu("文件")
        open_action = file_menu.addAction("打开")
        save_action = file_menu.addAction("保存")
        new_action = file_menu.addAction("新建事件")
        delete_action = file_menu.addAction("删除事件")
        open_action.triggered.connect(self.open_file)
        save_action.triggered.connect(self.save_file)
        new_action.triggered.connect(self.new_event)
        delete_action.triggered.connect(self.delete_event)
        exit_action = file_menu.addAction("退出")
        exit_action.triggered.connect(self.close)
        self.setMenuBar(menu_bar)

    def update_event_list(self) -> None:
        """
        根据当前状态和事件类型过滤更新事件列表，并自动选中当前事件
        """
        self.event_list.blockSignals(True)
        self.event_list.clear()
        current_state = self.state_filter_widget.list_widget.currentItem()
        selected_state = current_state.data(Qt.UserRole) if current_state else None
        current_type = self.event_type_combo.currentData()
        for uid, event in self.event_manager.events.items():
            if selected_state is not None and event.status_id != selected_state:
                continue
            if current_type is not None and event.start != current_type:
                continue
            item = QListWidgetItem(f"{event.text} ({uid[:8]})")
            item.setData(Qt.UserRole, uid)
            self.event_list.addItem(item)
        if self.event_manager.current_event_id:
            for i in range(self.event_list.count()):
                item = self.event_list.item(i)
                if item.data(Qt.UserRole) == self.event_manager.current_event_id:
                    self.event_list.setCurrentRow(i)
                    break
        self.event_list.blockSignals(False)

    def load_event_detail(self, row: int) -> None:
        """
        加载选中事件的详细信息到编辑器中
        """
        if row < 0 or row >= self.event_list.count():
            return
        uid = self.event_list.item(row).data(Qt.UserRole)
        self.event_manager.current_event_id = uid
        event = self.event_manager.events.get(uid)
        if event:
            self.text_edit.blockSignals(True)
            self.text_edit.setText(event.text)
            self.text_edit.blockSignals(False)
            self.update_premise_list(event)
            self.update_settle_list(event)

    def update_event_text(self) -> None:
        """
        更新当前事件的文本内容
        """
        uid = self.event_manager.current_event_id
        if uid:
            event = self.event_manager.events.get(uid)
            if event:
                event.text = self.text_edit.toPlainText()
                self.update_event_list()

    def update_premise_list(self, event: Event) -> None:
        """
        更新前提列表显示
        """
        self.premise_list.clear()
        for cid in event.premise:
            self.premise_list.addItem(premise_data.get(cid, cid))

    def update_settle_list(self, event: Event) -> None:
        """
        更新结算器列表显示
        """
        self.settle_list.clear()
        for sid in event.settle:
            self.settle_list.addItem(settle_data.get(sid, sid))

    def edit_premise(self) -> None:
        """
        编辑当前事件的前提
        """
        uid = self.event_manager.current_event_id
        if uid:
            event = self.event_manager.events.get(uid)
            dlg = GroupedMultiSelectDialog(premise_data, premise_type_data, event.premise, "编辑前提", self)
            if dlg.exec() == QDialog.Accepted:
                self.update_premise_list(event)
                self.statusBar.showMessage("前提更新成功", 2000)

    def edit_settle(self) -> None:
        """
        编辑当前事件的结算器（效果）
        """
        uid = self.event_manager.current_event_id
        if uid:
            event = self.event_manager.events.get(uid)
            dlg = TwoLevelGroupedMultiSelectDialog(settle_data, settle_type_data, event.settle, "编辑结算器", self)
            if dlg.exec() == QDialog.Accepted:
                self.update_settle_list(event)
                self.statusBar.showMessage("结算器更新成功", 2000)

    def open_file(self) -> None:
        """
        打开事件文件
        """
        file_path, _ = QFileDialog.getOpenFileName(self, "打开事件文件", ".", "Json 文件 (*.json)")
        if file_path:
            self.event_manager.load_events(file_path)
            self.statusBar.showMessage("加载成功", 2000)

    def save_file(self) -> None:
        """
        保存事件文件
        """
        file_path, _ = QFileDialog.getSaveFileName(self, "保存事件文件", ".", "Json 文件 (*.json)")
        if file_path:
            self.event_manager.save_events(file_path)
            self.statusBar.showMessage("保存成功", 2000)

    def delete_event(self) -> None:
        """
        删除选中的事件
        """
        row = self.event_list.currentRow()
        if row >= 0:
            uid = self.event_list.item(row).data(Qt.UserRole)
            reply = QMessageBox.question(self, "删除事件", "确定删除此事件吗？", QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.event_manager.delete_event(uid)
                self.statusBar.showMessage("事件已删除", 2000)


def main() -> None:
    """
    主程序入口。
    """
    load_config()
    app = QApplication(sys.argv)
    global main_window
    manager = EventManager()
    main_window = MainWindow(manager)
    main_window.resize(1000, 600)
    main_window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
