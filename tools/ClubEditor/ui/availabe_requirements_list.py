from PySide6.QtWidgets import QTreeWidget, QTreeWidgetItem, QHBoxLayout, QAbstractItemView, QWidget
from PySide6.QtGui import Qt
import cache_control

class AvailabeRequirementsListItem(QTreeWidgetItem):
    """ 社团门槛选择列表对象 """

    def __init__(self,now_root, premise_id: str):
        """
        初始化树节点对象
        Keyword arguments:
        premise_id -- 前提id
        """
        super().__init__(now_root)
        premise = cache_control.premise_data[premise_id]
        self.setText(0, premise)
        self.cid = premise_id
        """ 传入的门槛id """
        self.name = premise
        """ 门槛描述 """

class AvailabeRequirementsList(QWidget):
    """ 社团门槛选择列表 """

    def __init__(self):
        """ 初始化对象 """
        super().__init__()
        club_judge = True
        if cache_control.now_club_id == "":
            club_judge = False
        if cache_control.now_club_id not in cache_control.club_list_data:
            club_judge = False
        self.layout = QHBoxLayout()
        all_type_list = list(cache_control.premise_type_data.keys())
        range_index = int(len(all_type_list) / 3) + 1
        range_a = all_type_list[:range_index]
        range_b = all_type_list[range_index : range_index * 2]
        range_c = all_type_list[range_index * 2 :]
        self.range_list = [range_a, range_b, range_c]
        index = 1
        self.tree_list = []
        for type_list in self.range_list:
            tree = QTreeWidget()
            tree.setHeaderLabel("门槛列表" + str(index))
            index += 1
            tree.setSelectionMode(QAbstractItemView.SingleSelection)
            tree.setSelectionBehavior(QAbstractItemView.SelectRows)
            root_list = []
            if not club_judge:
                tree.addTopLevelItems(root_list)
                tree.itemActivated.connect(self.click_item)
                self.layout.addWidget(tree)
                self.tree_list.append(tree)
                continue
            club_data = cache_control.club_list_data[cache_control.now_club_id]
            for now_type in type_list:
                now_root = QTreeWidgetItem(tree)
                now_root.setText(0, now_type)
                premise_list = list(cache_control.premise_type_data[now_type])
                for premise in premise_list:
                    premise_node = AvailabeRequirementsListItem(now_root, premise)
                    if premise in club_data.premise_data:
                        premise_node.setCheckState(0, Qt.Checked)
                    else:
                        premise_node.setCheckState(0, Qt.Unchecked)
                root_list.append(now_root)
            tree.addTopLevelItems(root_list)
            tree.itemActivated.connect(self.click_item)
            self.layout.addWidget(tree)
            self.tree_list.append(tree)
        self.setLayout(self.layout)
        cache_control.update_signal.connect(self._update)

    def _update(self):
        index = 0
        club_judge = True
        if cache_control.now_club_id == "":
            club_judge = False
        if cache_control.now_club_id not in cache_control.club_list_data:
            club_judge = False
        for tree in self.tree_list:
            tree.clear()
            type_list = self.range_list[index]
            index += 1
            if not club_judge:
                continue
            club_data = cache_control.club_list_data[cache_control.now_club_id]
            root_list = []
            for now_type in type_list:
                now_root = QTreeWidgetItem(tree)
                now_root.setText(0, now_type)
                premise_list = list(cache_control.premise_type_data[now_type])
                for premise in premise_list:
                    premise_node = AvailabeRequirementsListItem(now_root, premise)
                    if premise in club_data.premise_data:
                        premise_node.setCheckState(0, Qt.Checked)
                    else:
                        premise_node.setCheckState(0, Qt.Unchecked)
                root_list.append(now_root)
        cache_control.update_premise_signal.emit()

    def click_item(self, item: AvailabeRequirementsListItem, column: int):
        """
        点击选项时勾选选框并更新口上前提
        Keyword arguments:
        item -- 点击的对象
        column -- 点击位置
        """
        if "cid" not in item.__dict__:
            return
        club_data = cache_control.club_list_data[cache_control.now_club_id]
        if item.checkState(column) == Qt.Checked:
            item.setCheckState(0, Qt.Unchecked)
            if item.cid in club_data.premise_data:
                del club_data.premise_data[item.cid]
        else:
            item.setCheckState(0, Qt.Checked)
            club_data.premise_data[item.cid] = 1
        cache_control.update_premise_signal.emit()

