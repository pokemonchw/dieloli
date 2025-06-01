from typing import List
from PySide6.QtWidgets import QDialog, QTableWidget, QHBoxLayout, QTableWidgetItem, QHeaderView
from PySide6.QtCore import Slot
import cache_control
import game_type


def list_of_groups(init_list: List[any], children_list_len: int) -> List[List[any]]:
    """
    将列表分割为指定长度的列表集合
    Keyword arguments:
    init_list -- 原始列表
    children_list_len -- 指定长度
    Return arguments:
    List[Tuple[any]] -- 新列表
    """
    list_of_groups = zip(*(iter(init_list),) * children_list_len)
    end_list = [list(i) for i in list_of_groups]
    count = len(init_list) % children_list_len
    if count:
        end_list.append(init_list[-count:])
    return end_list


class SuitWearMenuItem(QTableWidgetItem):
    """
    服装指定单元格对象
    Keyword arguments:
    uid -- 服装id
    """

    def __init__(self, uid: str):
        if uid == "":
            super().__init__("移除")
        else:
            clothing_data = cache_control.clothing_list_data[uid]
            super().__init__(clothing_data.name)
        self.uid = uid


class SuitWearMenu(QDialog):
    """
    套装服装穿戴指定
    Keyword arguments:
    position -- 穿戴位置
    """

    def __init__(self, position: int):
        super(SuitWearMenu, self).__init__()
        self.setWindowTitle("指定" + cache_control.wear_type_data[position])
        self.position = position
        wear_list = [""]
        for clothing_key in cache_control.clothing_list_data:
            clothing_data = cache_control.clothing_list_data[clothing_key]
            if clothing_data.clothing_type == position:
                wear_list.append(clothing_data)
        col_slice: List[List[game_type.ClothingTem]] = list_of_groups(wear_list, 5)
        rows = len(col_slice)
        table_widget = QTableWidget(rows, 5, self)
        row_index = 0
        for row in col_slice:
            col_index = 0
            for col in row:
                if col == "":
                    table_widget.setItem(row_index, col_index, SuitWearMenuItem(""))
                else:
                    table_widget.setItem(row_index, col_index, SuitWearMenuItem(col.cid))
                col_index += 1
            row_index += 1
        table_widget.setEditTriggers(QTableWidget.NoEditTriggers)
        table_widget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table_widget.itemClicked.connect(self.on_item_clicked)
        table_widget.itemDoubleClicked.connect(self.on_item_double_clicked)
        main_layout = QHBoxLayout(self)
        main_layout.addWidget(table_widget)
        self.setLayout(main_layout)

    @Slot(QTableWidgetItem)
    def on_item_clicked(self, item: SuitWearMenuItem):
        suit_data = cache_control.suit_list_data[cache_control.now_suit_id]
        if item.uid == "":
            if self.position in suit_data.clothing_wear:
                del suit_data.clothing_wear[self.position]
                cache_control.update_signal.emit()
            return
        suit_data.clothing_wear[self.position] = item.uid
        cache_control.update_signal.emit()

    @Slot(QTableWidgetItem)
    def on_item_double_clicked(self, item: SuitWearMenuItem):
        suit_data = cache_control.suit_list_data[cache_control.now_suit_id]
        if item.uid == "":
            if self.position in suit_data.clothing_wear:
                del suit_data.clothing_wear[self.position]
                cache_control.update_signal.emit()
            self.accept()
            return
        suit_data.clothing_wear[self.position] = item.uid
        cache_control.update_signal.emit()
        self.accept()

