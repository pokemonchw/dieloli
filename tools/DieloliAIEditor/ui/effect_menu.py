from PySide6.QtWidgets import (
    QHBoxLayout,
    QDialog,
    QTreeWidget,
    QTreeWidgetItem,
    QAbstractItemView,
)
from PySide6.QtGui import QFont, Qt
from game_type import TreeItem
import cache_control


class EffectMenu(QDialog):
    """目标效果选择对象"""

    def __init__(self):
        """初始化目标效果复选框"""
        super(EffectMenu, self).__init__()
        self.setWindowTitle("效果设置")
        self.font = QFont()
        self.font.setPointSize(18)
        self.layout: QHBoxLayout = QHBoxLayout()
        all_type_list = list(cache_control.premise_type_data.keys())
        range_index = int(len(all_type_list) / 3) + 1
        range_a = all_type_list[:range_index]
        range_b = all_type_list[range_index : range_index * 2]
        range_c = all_type_list[range_index * 2 :]
        range_list = [range_a, range_b, range_c]
        index = 1
        for type_list in range_list:
            tree = QTreeWidget()
            tree.setHeaderLabel("效果列表" + str(index))
            index += 1
            tree.setSelectionMode(QAbstractItemView.SingleSelection)
            tree.setSelectionBehavior(QAbstractItemView.SelectRows)
            root_list = []
            for now_type in type_list:
                now_root = QTreeWidgetItem(tree)
                now_root.setText(0, now_type)
                premise_list = list(cache_control.premise_type_data[now_type])
                for premise in premise_list:
                    premise_node = TreeItem(now_root)
                    premise_node.cid = premise
                    premise_node.setText(0, cache_control.premise_data[premise])
                    if premise in cache_control.now_target_data[cache_control.now_target_id].effect:
                        premise_node.setCheckState(0, Qt.Checked)
                    else:
                        premise_node.setCheckState(0, Qt.Unchecked)
                root_list.append(now_root)
            tree.addTopLevelItems(root_list)
            tree.itemActivated.connect(self.click_item)
            tree.setFont(self.font)
            self.layout.addWidget(tree)
        self.setLayout(self.layout)

    def update_premise(self, item: TreeItem, column: int):
        """
        勾选选框时更新目标效果
        Keyword arguments:
        item -- 勾选的对象
        column -- 勾选位置
        """
        if item.checkState(column) == Qt.Checked:
            cache_control.now_target_data[cache_control.now_target_id].effect[item.cid] = 1
        elif item.cid in cache_control.now_target_data[cache_control.now_target_id].effect:
            del cache_control.now_target_data[cache_control.now_target_id].effect[item.cid]

    def click_item(self, item: TreeItem, column: int):
        """
        点击选项时勾选选框并更新口上前提
        Keyword arguments:
        item -- 点击的对象
        column -- 点击位置
        """
        if "cid" not in item.__dict__:
            return
        if item.checkState(column) == Qt.Checked:
            item.setCheckState(0, Qt.Unchecked)
            if item.cid in cache_control.now_target_data[cache_control.now_target_id].effect:
                del cache_control.now_target_data[cache_control.now_target_id].effect[item.cid]
        else:
            item.setCheckState(0, Qt.Checked)
            cache_control.now_target_data[cache_control.now_target_id].effect[item.cid] = 1
