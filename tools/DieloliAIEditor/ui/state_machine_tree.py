from PySide6.QtWidgets import QTreeWidget,QAbstractItemView,QTreeWidgetItem
from PySide6.QtGui import QFont
from game_type import TreeItem
import cache_control


class StateMachineTree(QTreeWidget):
    """状态机选择对象"""

    def __init__(self):
        """ 初始化状态机选择对象 """
        super(StateMachineTree,self).__init__()
        self.font = QFont()
        self.font.setPointSize(18)
        self.setHeaderLabel("状态机列表")
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        state_machine_type_list = list(cache_control.state_machine_type_data.keys())
        root_list = []
        for now_type in state_machine_type_list:
            now_root = QTreeWidgetItem(self)
            now_root.setText(0,now_type)
            state_machine_list = list(cache_control.state_machine_type_data[now_type])
            for state_machine in state_machine_list:
                state_machine_node = TreeItem(now_root)
                state_machine_node.cid = state_machine
                state_machine_node.setText(0,cache_control.state_machine_data[state_machine])
            root_list.append(now_root)
        self.addTopLevelItems(root_list)
        self.setFont(self.font)
