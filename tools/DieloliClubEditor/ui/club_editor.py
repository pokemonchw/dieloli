from PySide6.QtWidgets import QMainWindow, QWidget, QHBoxLayout
from ui.club_list_widget import ClubListWidget


class ClubEditor(QMainWindow):
    """ 社团编辑器主面板 """

    def __init__(self):
        super().__init__()

    def init_ui(self):
        """ 初始化ui """
        # 设置主窗口的中央组件
        self.central_widget = QWidget()
        self.setCentralWidget = self.central_widget
        # 水平布局
        self.main_layout = QHBoxLayout(self.central_widget)
        # 左侧 - 社团列表
        self.club_list_widget = ClubListWidget()
        self.main_layout.addWidget(self.club_list_widget)
        self.club_list_widget.clicked.connect(self.on_club_selected)
        # 右侧 - 社团信息

    def on_club_selected(self):
        """ 选择社团 """
        pass
