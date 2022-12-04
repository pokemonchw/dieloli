import uuid
from PySide6.QtWidgets import QListWidget,QAbstractItemView,QListWidgetItem, QMenu,QWidgetAction
from PySide6.QtGui import QFont,QCursor
from PySide6.QtCore import Qt, QModelIndex
from ui.premise_menu import PremiseMenu
from ui.effect_menu import EffectMenu
import cache_control
import game_type


class DataList(QListWidget):
    """表单主体"""

    def __init__(self):
        """初始化表单主体"""
        super(DataList,self).__init__()
        self.font = QFont()
        self.font.setPointSize(16)
        self.setFont(self.font)
        self.close_flag = 1
        self.edited_item = self.currentItem()
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        self.doubleClicked.connect(self.item_double_clicked)
        self.currentItemChanged.connect(self.close_edit)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.right_button_menu)
        self.update_clear = 0

    def item_double_clicked(self, model_index: QModelIndex):
        """
        双击目标
        Keyword arguments:
        model_index -- 目标序号
        """
        self.close_edit()
        item = self.item(model_index.row())
        self.edited_item = item
        self.openPersistentEditor(item)
        self.editItem(item)

    def close_edit(self):
        """关闭编辑"""
        item: QListWidgetItem = self.edited_item
        if item and self.isPersistentEditorOpen(item):
            uid = item.uid
            cache_control.now_target_data[uid].text = item.text()
            self.closePersistentEditor(item)

    def right_button_menu(self, old_position):
        """
        右键菜单
        Keyword arguments:
        position -- 鼠标点击位置
        """
        menu = QMenu()
        if not len(cache_control.now_file_path):
            return
        if not len(cache_control.now_state_machine_id):
            return
        create_action: QWidgetAction = QWidgetAction(self)
        create_action.setText("新增目标")
        create_action.triggered.connect(self.create_target)
        menu.addAction(create_action)
        menu.setFont(self.font)
        position = QCursor.pos()
        font = QFont()
        font.setPointSize(16)
        if self.itemAt(old_position):
            copy_action: QWidgetAction = QWidgetAction(self)
            copy_action.setText("复制目标")
            copy_action.triggered.connect(self.copy_target)
            menu.addAction(copy_action)
            delete_action: QWidgetAction = QWidgetAction(self)
            delete_action.setText("删除目标")
            delete_action.triggered.connect(self.delete_target)
            menu.addAction(delete_action)
            premise_action: QWidgetAction = QWidgetAction(self)
            premise_action.setText("设置前提")
            premise_action.triggered.connect(self.setting_premise)
            menu.addAction(premise_action)
            effect_action: QWidgetAction = QWidgetAction(self)
            effect_action.setText("设置效果")
            effect_action.triggered.connect(self.setting_effect)
            menu.addAction(effect_action)
        menu.exec(position)

    def create_target(self):
        """新增目标"""
        item = game_type.ListItem("空目标")
        item.uid = str(uuid.uuid4())
        target = game_type.Target()
        target.uid = item.uid
        target.state_machine_id = cache_control.now_state_machine_id
        target.text = item.text()
        cache_control.now_target_data[target.uid] = target
        self.addItem(item)
        self.close_edit()

    def delete_target(self):
        """删除目标"""
        target_index = self.currentRow()
        item = self.item(target_index)
        if not self.update_clear:
            del cache_control.now_target_data[item.uid]
        self.takeItem(target_index)
        self.close_edit()

    def copy_target(self):
        """复制目标"""
        target_index = self.currentRow()
        old_item = self.item(target_index)
        old_target = cache_control.now_target_data[old_item.uid]
        new_item = game_type.ListItem(old_item.text() + "(复制)")
        new_item.uid = str(uuid.uuid4())
        target = game_type.Target()
        target.uid = new_item.uid
        target.state_machine_id = old_target.state_machine_id
        for premise in old_target.premise:
            target.premise[premise] = old_target.premise[premise]
        for effect in old_target.effect:
            target.effect[effect] = old_target.effect[effect]
        target.text = old_target.text + "(复制)"
        cache_control.now_target_data[target.uid] = target
        self.insertItem(target_index + 1, new_item)
        self.close_edit()

    def setting_premise(self):
        """设置目标前提"""
        target_index = self.currentRow()
        item = self.item(target_index)
        cache_control.now_target_id = item.uid
        menu = PremiseMenu()
        menu.exec()

    def setting_effect(self):
        """设置目标效果"""
        target_index = self.currentRow()
        item = self.item(target_index)
        cache_control.now_target_id = item.uid
        menu = EffectMenu()
        menu.exec()

    def update(self):
        """根据选项刷新当前绘制的列表"""
        self.update_clear = 1
        self.edited_item = None
        self.clear()
        self.update_clear = 0
        for uid in cache_control.now_target_data:
            now_target: game_type.Target = cache_control.now_target_data[uid]
            if now_target.state_machine_id != cache_control.now_state_machine_id:
                continue
            item = game_type.ListItem(now_target.text)
            item.uid = uid
            self.addItem(item)
        self.close_edit()
