from PySide6.QtWidgets import (
    QGroupBox, QListWidget, QAbstractItemView, QVBoxLayout, QListWidgetItem, QWidgetAction,
    QMenu
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QCursor
import uuid
import cache_control
import game_type


class ClothingListGroupItem(QListWidgetItem):
    """ 服装列表的表单对象 """

    def __init__(self, clothing_name: str, uid: str):
        """
        初始化表单对象
        Keyword arguments:
        clothing_name -- 服装名
        uid -- 服装唯一id
        """
        super().__init__(clothing_name)
        self.setToolTip(clothing_name)
        self.uid = uid
        """ 服装唯一id """
        self.name = clothing_name
        """ 服装名 """


class ClothingListGroup(QGroupBox):
    """ 服装列表 """

    def __init__(self):
        super().__init__("服装列表")
        self.clothing_list = QListWidget()
        self.clothing_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.clothing_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.clothing_list.itemClicked.connect(self._item_cliecked)
        list_layout = QVBoxLayout()
        list_layout.addWidget(self.clothing_list)
        self.setLayout(list_layout)
        self.clothing_list.customContextMenuRequested.connect(self._right_button_menu)
        cache_control.update_signal.connect(self.update)

    def update(self):
        """ 更新服装列表 """
        self.clothing_list.clear()
        if cache_control.now_clothing_id not in cache_control.clothing_list_data:
            cache_control.now_clothing_id = ""
        i = 0
        set_item = None
        for clothing_id in cache_control.clothing_list_data:
            if not i and cache_control.now_clothing_id == "":
                cache_control.now_clothing_id = clothing_id
            clothing_data = cache_control.clothing_list_data[clothing_id]
            item = ClothingListGroupItem(clothing_data.name, clothing_id)
            if clothing_id == cache_control.now_clothing_id:
                set_item = item
            self.clothing_list.addItem(item)
            i += 1
        if set_item != None:
            set_item.setSelected(True)

    def _right_button_menu(self, old_position):
        """ 右键菜单 """
        if not len(cache_control.now_file_path):
            return
        menu = QMenu()
        create_action: QWidgetAction = QWidgetAction(self)
        create_action.setText("新建服装")
        create_action.triggered.connect(self._create_clothing)
        menu.addAction(create_action)
        delete_action: QWidgetAction = QWidgetAction(self)
        delete_action.setText("删除服装")
        delete_action.triggered.connect(self._delete_clothing)
        menu.addAction(delete_action)
        position = QCursor.pos()
        menu.exec(position)

    def _item_cliecked(self, item: ClothingListGroupItem):
        """ 点击选中 """
        cache_control.now_clothing_id = item.uid
        cache_control.update_signal.emit()

    def _create_clothing(self):
        """ 创建服装 """
        uid = str(uuid.uuid4())
        item = ClothingListGroupItem("未命名", uid)
        clothing_data = game_type.ClothingTem()
        clothing_data.name = "未命名"
        clothing_data.cid = uid
        cache_control.clothing_list_data[uid] = clothing_data
        self.clothing_list.addItem(item)
        cache_control.update_signal.emit()

    def _delete_clothing(self):
        """ 删除服装 """
        target_index = self.clothing_list.currentRow()
        item: ClothingListGroupItem = self.clothing_list.item(target_index)
        if "uid" not in item.__dict__:
            return
        if item.uid == cache_control.now_clothing_id:
            cache_control.now_clothing_id = ""
        del cache_control.clothing_list_data[item.uid]
        cache_control.update_signal.emit()
