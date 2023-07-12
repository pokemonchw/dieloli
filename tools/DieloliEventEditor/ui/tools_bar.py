from PySide6.QtWidgets import QMenuBar, QMenu, QWidgetAction
from PySide6.QtGui import QFont
import cache_control


class ToolsBar(QMenuBar):
    """筛选表单用菜单栏"""

    def __init__(self):
        """初始化顶部筛选栏"""
        super(ToolsBar, self).__init__()
        self.font = QFont()
        self.font.setPointSize(16)
        file_menu = QMenu("文件", self)
        self.select_event_file_action = QWidgetAction(self)
        self.select_event_file_action.setText("选择事件文件    Ctrl+O")
        self.new_event_file_action = QWidgetAction(self)
        self.new_event_file_action.setText("新建事件文件    Ctrl+N")
        self.save_event_action = QWidgetAction(self)
        self.save_event_action.setText("保存事件        Ctrl+S")
        self.exit_action = QWidgetAction(self)
        self.exit_action.setText("关闭编辑器      Ctrl+Q")
        file_menu.addActions(
            [
                self.select_event_file_action,
                self.new_event_file_action,
                self.save_event_action,
                self.exit_action,
            ]
        )
        self.addMenu(file_menu)
        self.npc_menu: QMenu = QMenu("0", self)
        self.npc_menu.setFixedWidth(50)
        self.addMenu(self.npc_menu)
        self.status_menu: QMenu = QMenu(cache_control.status_data[cache_control.now_status], self)
        self.status_menu.setFont(self.font)
        self.addMenu(self.status_menu)
        self.start_menu: QMenu = QMenu(cache_control.start_status, self)
        self.start_menu.setFont(self.font)
        self.addMenu(self.start_menu)
        self.setFont(self.font)
