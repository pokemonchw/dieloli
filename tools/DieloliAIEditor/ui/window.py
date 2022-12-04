from PySide6.QtWidgets import QHBoxLayout, QMainWindow, QWidget


class Window(QMainWindow):
    """编辑器主窗体"""

    def __init__(self):
        """初始化编辑器主窗体"""
        super(Window, self).__init__()
        self.setWindowTitle("DieloliAI编辑器")
        self.layout: QHBoxLayout = QHBoxLayout()
        """ 界面布局 """

    def add_widget(self, widget: QWidget):
        """
        添加小部件到布局中
        Keyword arguments:
        widget -- 小部件
        """
        self.layout.addWidget(widget)

    def completed_layout(self):
        """布局完成"""
        widget = QWidget()
        widget.setLayout(self.layout)
        self.setCentralWidget(widget)
