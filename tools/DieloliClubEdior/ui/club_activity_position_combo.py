from PySide6.QtWidgets import QWidget, QVBoxLayout, QTreeView, QLineEdit, QDialog, QHeaderView
from PySide6.QtGui import QStandardItemModel, QStandardItem
from PySide6.QtCore import Qt, Signal, QModelIndex
import cache_control
import map_config


class TreeNode(QStandardItem):
    """ 地图树场景节点对象 """

    def __init__(self, path: str):
        super().__init__()
        scene_data = cache_control.scene_data[path]
        self.setText(scene_data.scene_name)
        self.path = path


class CustomTreeView(QTreeView):
    """
    自定义的 QTreeView，用于捕获双击事件
    """
    itemDoubleClicked = Signal(QStandardItem)


class MapSceneTreeView(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.initUI()
        self.loadTree()
        self.treeView.collapseAll()

    def initUI(self):
        """
        初始化用户界面，设置布局和树视图
        """
        self.layout = QVBoxLayout(self)
        self.treeView = CustomTreeView(self)
        self.treeView.doubleClicked.connect(self.onItemDoubleClicked)
        self.layout.addWidget(self.treeView)
        self.setLayout(self.layout)
        self.resize(400, 300)

    def loadTree(self):
        """
        加载树
        """
        model = QStandardItemModel()
        root_item = model.invisibleRootItem()
        root_item.setEditable(False)
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
                    map_node.setEditable(False)
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
            scene_node = TreeNode(scene_path_str)
            scene_node.setEditable(False)
            map_node.appendRow(scene_node)
            node_data[scene_path_str] = scene_node
        self.treeView.setModel(model)
        self.treeView.expandAll()

    def onItemDoubleClicked(self, index: QModelIndex):
        """ 处理双击事件 """
        model = self.treeView.model()
        item = model.itemFromIndex(index)
        if "path" not in item.__dict__:
            return
        club_data = cache_control.club_list_data[cache_control.now_club_id]
        activity_id = club_data.activity_list[cache_control.now_activity_id]
        activity_id.activity_position = map_config.get_map_system_path_for_str(item.path)
        cache_control.update_signal.emit()
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
        if cache_control.now_club_id == "":
            return
        club_data = cache_control.club_list_data[cache_control.now_club_id]
        if cache_control.now_activity_id not in club_data.activity_list:
            cache_control.now_activity_id = ""
        if cache_control.now_activity_id == "":
            if len(club_data.activity_list):
                cache_control.now_activity_id = list(club_data.activity_list.keys())[0]
        if cache_control.now_activity_id == "":
            return
        activity_data = club_data.activity_list[cache_control.now_activity_id]
        position_str = map_config.get_map_system_path_str(activity_data.activity_position)
        if position_str == "":
            self.lineEdit.setText("")
            return
        scene_data = cache_control.scene_data[position_str]
        def find_map_name(position_list: list):
            if not len(position_list):
                return ""
            map_path = position_list[:-1]
            position_path_str = map_config.get_map_system_path_str(position_list)
            map_data = cache_control.map_data[position_path_str]
            return find_map_name(map_path) + map_data.map_name + "-"
        scene_name = find_map_name(activity_data.activity_position[:-1]) + scene_data.scene_name
        self.lineEdit.setText(scene_name)

