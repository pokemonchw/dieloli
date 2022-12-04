from PySide6.QtWidgets import QTreeWidgetItem, QListWidgetItem


class Target:
    """目标对象"""

    def __init__(self):
        """初始化口上对象"""
        self.uid: str = ""
        """ 目标唯一id """
        self.text: str = ""
        """ 目标描述 """
        self.state_machine_id: str = ""
        """ 执行的状态机id """
        self.premise: dict = {}
        """ 目标的前提集合 """
        self.effect: dict = {}
        """ 目标的效果集合 """


class TreeItem(QTreeWidgetItem):
    """带有id的树节点对象"""

    def __init__(self, any):
        """ 初始化树节点对象 """
        super(TreeItem, self).__init__(any)
        self.cid = ""
        """ 传入的cid """


class ListItem(QListWidgetItem):
    """表单对象"""

    def __init__(self, any):
        """初始化表单对象"""
        super(ListItem, self).__init__(any)
        self.uid = ""
        """ 口上id """
