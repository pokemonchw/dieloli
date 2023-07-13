from PySide6.QtWidgets import QMenuBar, QMenu, QWidgetAction
from PySide6.QtGui import QFont
import cache_control


class MenuBar(QMenuBar):
    """顶部菜单栏"""

    def __init__(self):
        """初始化顶部菜单栏"""
        super(MenuBar, self).__init__()
        self.font = QFont()
        self.font.setPointSize(16)
        self.file_menu = FileMenu(self.font)
        self.addMenu(self.file_menu)
        self.needs_hierarchy_menu = QMenu("",self)
        self.addMenu(self.needs_hierarchy_menu)
        self.status_menu = QMenu("", self)
        self.addMenu(self.status_menu)


class FileMenu(QMenu):
    """ 文件选型菜单 """

    def __init__(self, now_font: QFont):
        """ 初始化菜单 """
        super(FileMenu, self).__init__("文件")
        self.setFont(now_font)
        self.select_ai_file_action = QWidgetAction(self)
        self.select_ai_file_action.setText("选择AI数据文件  Ctrl+O")
        self.new_ai_file_action = QWidgetAction(self)
        self.new_ai_file_action.setText("新建AI数据文件  Ctrl+N")
        self.save_ai_action = QWidgetAction(self)
        self.save_ai_action.setText("保存AI数据文件  Ctrl+S")
        self.exit_action = QWidgetAction(self)
        self.exit_action.setText("关闭编辑器      Ctrl+Q")
        self.addActions(
            [
                self.select_ai_file_action,
                self.new_ai_file_action,
                self.save_ai_action,
                self.exit_action,
            ]
        )
