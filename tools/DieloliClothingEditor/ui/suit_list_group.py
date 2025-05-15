from PySide6.QtWidgets import (
    QGroupBox, QListWidget, QAbstractItemView, QVBoxLayout, QListWidgetItem, QWidgetAction,
    QMenu
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QCursor
import uuid
import cache_control
import game_type


class SuitListGroupItem(QListWidgetItem):
    """ 套装列表的表单对象 """

    def __init__(self, suit_name: str, uid: str):
        """
        初始化表单对象
        Keyword arguments:
        suit_name -- 套装名
        uid -- 套装唯一id
        """
        super().__init__(suit_name)
        self.setToolTip(suit_name)
        self.uid = uid
        """ 套装唯一id """
        self.name = suit_name
        """ 套装名 """


class SuitListGroup(QGroupBox):
    """ 套装列表 """

    def __init__(self):
        super().__init__("套装列表")
        self.suit_list = QListWidget()
        self.suit_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.suit_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.suit_list.itemClicked.connect(self._item_cliecked)
        list_layout = QVBoxLayout()
        list_layout.addWidget(self.suit_list)
        self.setLayout(list_layout)
        self.suit_list.customContextMenuRequested.connect(self._right_button_menu)
        cache_control.update_signal.connect(self.update)

    def update(self):
        """ 更新套装列表 """
        self.suit_list.clear()
        if cache_control.now_suit_id not in cache_control.suit_list_data:
            cache_control.now_suit_id = ""
        i = 0
        set_item = None
        for suit_id in cache_control.suit_list_data:
            if not i and cache_control.now_suit_id == "":
                cache_control.now_suit_id = suit_id
            suit_data = cache_control.suit_list_data[suit_id]
            item = SuitListGroupItem(suit_data.name, suit_id)
            if suit_id == cache_control.now_suit_id:
                set_item = item
            self.suit_list.addItem(item)
            i += 1
        if set_item != None:
            set_item.setSelected(True)

    def _right_button_menu(self, old_position):
        """ 右键菜单 """
        if not len(cache_control.now_file_path):
            return
        menu = QMenu()
        create_action: QWidgetAction = QWidgetAction(self)
        create_action.setText("新建套装")
        create_action.triggered.connect(self._create_suit)
        menu.addAction(create_action)
        delete_action: QWidgetAction = QWidgetAction(self)
        delete_action.setText("删除套装")
        delete_action.triggered.connect(self._delete_suit)
        menu.addAction(delete_action)
        position = QCursor.pos()
        menu.exec(position)

    def _item_cliecked(self, item: SuitListGroupItem):
        """ 点击选中 """
        cache_control.now_suit_id = item.uid
        cache_control.update_signal.emit()

    def _create_suit(self):
        """ 创建套装 """
        uid = str(uuid.uuid4())
        item = SuitListGroupItem("未命名", uid)
        suit_data = game_type.ClothingSuit()
        suit_data.name = "未命名"
        suit_data.cid = uid
        cache_control.suit_list_data[uid] = suit_data
        self.suit_list.addItem(item)
        cache_control.now_suit_id = uid
        cache_control.update_signal.emit()

    def _delete_suit(self):
        """ 删除套装 """
        target_index = self.suit_list.currentRow()
        item: SuitListGroupItem = self.suit_list.item(target_index)
        if "uid" not in item.__dict__:
            return
        if item.uid == cache_control.now_suit_id:
            cache_control.now_suit_id = ""
        del cache_control.suit_list_data[item.uid]
        cache_control.update_signal.emit()
