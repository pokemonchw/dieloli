import uuid
from PySide6.QtWidgets import (
    QGroupBox, QVBoxLayout, QListWidget, QListWidgetItem, QMenu, QWidgetAction,
    QAbstractItemView,
)
from PySide6.QtGui import QCursor
from PySide6.QtCore import Qt
import cache_control
import game_type

class ClubActivityItem(QListWidgetItem):
    """ 社团活动列表的表单对象 """

    def __init__(self, activity_name: str, uid: str):
        """
        初始化表单对象
        Keyword arguments:
        activity_name -- 活动名
        aid -- 活动唯一id
        """
        super().__init__(activity_name)
        self.setToolTip(activity_name)
        self.uid = uid
        """ 活动唯一id """
        self.name = activity_name
        """ 活动名 """


class ClubActivityList(QGroupBox):
    """ 社团活动列表面板 """

    def __init__(self):
        super().__init__("活动列表")
        main_layout = QVBoxLayout(self)
        self.activity_list = QListWidget()
        self.activity_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.activity_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.activity_list.customContextMenuRequested.connect(self._right_button_menu)
        self.activity_list.itemClicked.connect(self._item_cliecked)
        main_layout.addWidget(self.activity_list)
        cache_control.now_activity_id = ""
        """ 当前选中的活动id """
        cache_control.update_signal.connect(self._update)

    def _update(self):
        """ 更新活动列表 """
        self.activity_list.clear()
        if cache_control.now_club_id not in cache_control.club_list_data:
            cache_control.now_club_id = ""
        if cache_control.now_club_id == "":
            return
        club_data = cache_control.club_list_data[cache_control.now_club_id]
        i = 0
        set_item = None
        for activity_id in club_data.activity_list:
            if not i and cache_control.now_activity_id == "":
                cache_control.now_activity_id = activity_id
            i += 1
            activity_data = club_data.activity_list[activity_id]
            item = ClubActivityItem(activity_data.name, activity_data.uid)
            self.activity_list.addItem(item)
            if activity_data.uid == cache_control.now_activity_id:
                set_item = item
        if set_item != None:
            set_item.setSelected(True)

    def _right_button_menu(self, old_position):
        """ 右键菜单 """
        if not len(cache_control.now_club_id):
            return
        menu = QMenu()
        create_action: QWidgetAction = QWidgetAction(self)
        create_action.setText("新建活动")
        create_action.triggered.connect(self._create_activity)
        menu.addAction(create_action)
        delete_action: QWidgetAction = QWidgetAction(self)
        delete_action.setText("删除活动")
        delete_action.triggered.connect(self._delete_club)
        position = QCursor.pos()
        menu.exec(position)

    def _create_activity(self):
        """ 创建活动 """
        uid = str(uuid.uuid4())
        item = ClubActivityItem("未命名", uid)
        activity_data = game_type.ClubActivityData()
        activity_data.name = "未命名"
        activity_data.uid = uid
        cache_control.club_list_data[cache_control.now_club_id].activity_list[uid] = activity_data
        self.activity_list.addItem(item)
        cache_control.update_signal.emit()

    def _delete_club(self):
        """ 删除活动 """
        target_index = self.activity_list.currentRow()
        item: ClubActivityItem = self.activity_list.item(target_index)
        club_data = cache_control.club_list_data[cache_control.now_club_id]
        if "uid" not in club_data.activity_list:
            return
        del club_data.activity_list[item.uid]
        if item.uid == cache_control.now_activity_id:
            cache_control.now_activity_id = ""
            cache_control.update_activity_time_signal.emit()
        cache_control.update_signal.emit()
        cache_control.update_activity_time_signal.emit()


    def _item_cliecked(self, item: ClubActivityItem):
        """
        点击选中
        """
        cache_control.now_activity_id = item.uid
        cache_control.update_signal.emit()
        cache_control.update_activity_time_signal.emit()

