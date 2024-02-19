import os
import json
from PySide6.QtWidgets import QWidget, QVBoxLayout, QTreeView, QLineEdit, QDialog
from PySide6.QtGui import QStandardItemModel, QStandardItem
from PySide6.QtCore import Qt, Signal
import cache_control
import map_config


class MapSceneTreeView(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.initUI()
        self.loadTree()

    def initUI(self):
        """
        初始化用户界面，设置布局和树视图
        """
        self.layout = QVBoxLayout(self)
        self.treeView = QTreeView(self)
        self.layout.addWidget(self.treeView)
        self.setLayout(self.layout)

    def loadTree(self):
        """
        加载树
        """
        model = QStandardItemModel()
        root_item = model.invisibleRootItem()
        model.setHorizontalHeaderLabels(['学校地图'])
        node_data = {}
        node_data[""] = root_item
        def check_old_map_node(map_path_list: list):
            if len(map_path_list) != 0:
                old_map_path_list = map_path_list[:-1]
                check_old_map_node(old_map_path_list)
                old_map_path_str = map_config.get_map_system_path_str(old_map_path_list)
                old_map_node: QStandardItem = node_data[old_map_path_str]
                map_path_str = map_config.get_map_system_path_str(map_path_list)
                if map_path_str not in node_data:
                    map_data = cache_control.map_data[map_path_str]
                    map_node = QStandardItem(map_data.map_name)
                    old_map_node.appendRow(map_node)
                    node_data[map_path_str] = map_node
        scene_path_list = list(cache_control.scene_data.keys())
        scene_path_list = sorted(scene_path_list)
        for scene_path_str in scene_path_list:
            scene_data = cache_control.scene_data[scene_path_str]
            scene_path_list = map_config.get_map_system_path_for_str(scene_path_str)
            map_path_list = scene_path_list[:-1]
            check_old_map_node(map_path_list)
            map_path_str = map_config.get_map_system_path_str(map_path_list)
            map_node = node_data[map_path_str]
            scene_node = QStandardItem(scene_data.scene_name)
            map_node.appendRow(scene_node)
            node_data[scene_path_str] = scene_node
        self.treeView.setModel(model)
        self.treeView.expandAll()


class ClickableLineEdit(QLineEdit):
    clicked = Signal()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setReadOnly(True)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


class ClubActivityPositionCombo(QWidget):
    """ 社团活动地点选择器 """

    def __init__(self, parent=None):
        """ 初始化对象 """
        super().__init__(parent)
        self.initUI()
        cache_control.update_signal.connect(self._update)

    def initUI(self):
        """ 初始化UI """
        self.layout = QVBoxLayout(self)
        self.lineEdit = ClickableLineEdit(self)
        self.lineEdit.clicked.connect(self.onLineEditClicked)
        self.layout.addWidget(self.lineEdit)

    def onLineEditClicked(self):
        """ 处理点击事件 """
        if cache_control.now_club_id == "":
            return
        if cache_control.now_activity_id == "":
            return
        selector = MapSceneTreeView(self)
        selector.exec()

    def _update(self):
        """ 刷新面板 """
        pass
