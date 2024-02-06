import uuid
from PySide6.QtWidgets import(
    QGroupBox, QListWidget, QVBoxLayout, QListWidgetItem, QMenu, QWidgetAction,
    QAbstractItemView,
)
from PySide6.QtGui import QCursor
from PySide6.QtCore import Qt, QModelIndex
import cache_control
import game_type


class ClubListGroupItem(QListWidgetItem):
    """ 社团列表的表单对象 """

    def __init__(self, club_name: str, uid: str):
        """
        初始化表单对象
        Keyword arguments:
        club_name -- 社团名
        cid -- 社团唯一id
        """
        super().__init__(club_name)
        self.setToolTip(club_name)
        self.uid = uid
        """ 社团唯一id """
        self.name = club_name
        """ 社团名 """


class ClubListGroup(QGroupBox):
    """ 社团列表 """

    def __init__(self):
        super().__init__("社团列表")
        self.club_list = QListWidget()
        self.club_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.club_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.club_list.itemClicked.connect(self._item_cliecked)
        list_layout = QVBoxLayout()
        list_layout.addWidget(self.club_list)
        self.setLayout(list_layout)
        self.club_list.customContextMenuRequested.connect(self._right_button_menu)
        cache_control.update_signal.connect(self.update)

    def update(self):
        """ 更新社团列表 """
        self.club_list.clear()
        if cache_control.now_club_id not in cache_control.club_list_data:
            cache_control.now_club_id = ""
        i = 0
        set_item = None
        for club_id in cache_control.club_list_data:
            if not i and cache_control.now_club_id == "":
                cache_control.now_club_id = club_id
            club_data = cache_control.club_list_data[club_id]
            item = ClubListGroupItem(club_data.name, club_id)
            if club_id == cache_control.now_club_id:
                set_item = item
            self.club_list.addItem(item)
            i += 1
        if set_item != None:
            set_item.setSelected(True)

    def _right_button_menu(self, old_position):
        """ 右键菜单 """
        if not len(cache_control.now_file_path):
            return
        menu = QMenu()
        create_action: QWidgetAction = QWidgetAction(self)
        create_action.setText("新建社团")
        create_action.triggered.connect(self._create_club)
        menu.addAction(create_action)
        delete_action: QWidgetAction = QWidgetAction(self)
        delete_action.setText("删除社团")
        delete_action.triggered.connect(self._delete_club)
        menu.addAction(delete_action)
        position = QCursor.pos()
        menu.exec(position)

    def _item_cliecked(self, item: ClubListGroupItem):
        """
        点击选中
        Keyword arguments:
        model_index -- 目标序号
        """
        cache_control.now_club_id = item.uid
        cache_control.update_signal.emit()

    def _create_club(self):
        """ 创建社团 """
        uid = str(uuid.uuid4())
        item = ClubListGroupItem("未命名", uid)
        club_data = game_type.ClubData()
        club_data.name = "未命名"
        club_data.uid = uid
        cache_control.club_list_data[uid] = club_data
        self.club_list.addItem(item)
        cache_control.update_signal.emit()

    def _delete_club(self):
        """ 删除社团 """
        target_index = self.club_list.currentRow()
        item: ClubListGroupItem = self.club_list.item(target_index)
        if "uid" not in item.__dict__:
            return
        del cache_control.club_list_data[item.uid]
        cache_control.update_signal.emit()


