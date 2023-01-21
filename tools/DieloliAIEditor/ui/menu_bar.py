from PySide6.QtWidgets import QMenuBar, QMenu, QWidgetAction
from PySide6.QtGui import QFont


class MenuBar(QMenuBar):
    """顶部菜单栏"""

    def __init__(self):
        """初始化顶部菜单栏"""
        super(MenuBar, self).__init__()
        file_menu = QMenu("文件", self)
        self.font = QFont()
        self.font.setPointSize(16)
        self.setFont(self.font)
        self.select_ai_file_action = QWidgetAction(self)
        self.select_ai_file_action.setText("选择AI数据文件  Ctrl+O")
        self.new_ai_file_action = QWidgetAction(self)
        self.new_ai_file_action.setText("新建AI数据文件  Ctrl+N")
        self.save_ai_action = QWidgetAction(self)
        self.save_ai_action.setText("保存AI数据文件  Ctrl+S")
        self.exit_action = QWidgetAction(self)
        self.exit_action.setText("关闭编辑器      Ctrl+Q")
        file_menu.addActions(
            [
                self.select_ai_file_action,
                self.new_ai_file_action,
                self.save_ai_action,
                self.exit_action,
            ]
        )
        file_menu.setFont(self.font)
        self.addMenu(file_menu)
        self.status_menu = QMenu("",self)
        self.addMenu(self.status_menu)
