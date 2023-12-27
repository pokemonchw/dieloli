import sys
import json
import csv
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QHBoxLayout, QVBoxLayout, QWidget,
    QLabel, QLineEdit, QComboBox, QListWidget, QListWidgetItem, QTextEdit, QDateTimeEdit,
    QFileDialog, QTimeEdit, QGroupBox, QFormLayout, QScrollArea, QSizePolicy, QTabWidget,
)
from PySide6.QtGui import QAction
from PySide6.QtCore import QTime, Qt


class DayScheduleScrollArea(QScrollArea):
    def __init__(self, day_widget):
        super().__init__()
        self.setWidget(day_widget)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)


class TimeSlotWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QFormLayout(self)
        self.start_time = QTimeEdit(QTime(8, 0))
        self.end_time = QTimeEdit(QTime(10, 0))
        self.delete_button = QPushButton("删除")
        self.delete_button.clicked.connect(self.delete_slot)

        self.layout.addRow("开始时间:", self.start_time)
        self.layout.addRow("结束时间:", self.end_time)
        self.layout.addRow(self.delete_button)

    def delete_slot(self):
        self.setParent(None)
        self.deleteLater()

class DayScheduleWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout(self)
        self.time_slot_widgets = []

    def add_time_slot(self):
        time_slot_widget = TimeSlotWidget(self)
        self.time_slot_widgets.append(time_slot_widget)
        self.layout.addWidget(time_slot_widget)


class WeeklyScheduleWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QHBoxLayout(self)
        self.days = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
        self.day_widgets = {}

        for day in self.days:
            day_column_layout = QVBoxLayout()

            # 添加天的标签
            day_group = QGroupBox(day)
            day_group_layout = QVBoxLayout(day_group)

            # 创建 DayScheduleWidget
            day_schedule_widget = DayScheduleWidget()
            self.day_widgets[day] = day_schedule_widget

            # 创建滚动区域并将 DayScheduleWidget 放入其中
            scroll_area = DayScheduleScrollArea(day_schedule_widget)
            day_group_layout.addWidget(scroll_area)

            # 添加时间段的按钮
            add_time_slot_button = QPushButton("添加活动时间")
            add_time_slot_button.clicked.connect(day_schedule_widget.add_time_slot)
            day_group_layout.addWidget(add_time_slot_button)
            day_column_layout.addWidget(day_group)

            self.layout.addLayout(day_column_layout)


class ClubActivityWidget(QWidget):
    def __init__(self):
        super().__init__()
        main_layout = QHBoxLayout(self)

        # Activity List
        activity_list_group = QGroupBox("活动列表")
        activity_list_group_layout = QVBoxLayout(activity_list_group)
        self.activity_list = QListWidget()
        activity_list_group_layout.addWidget(self.activity_list)
        main_layout.addWidget(activity_list_group)

        # Activity Editor
        self.activity_editor = QWidget()
        editor_layout = QVBoxLayout(self.activity_editor)

        # Form layout for activity details
        form_layout = QFormLayout()
        self.activity_name = QLineEdit()
        self.activity_description = QTextEdit()
        self.activity_time = QDateTimeEdit()
        self.activity_location = QLineEdit()

        form_layout.addRow("活动名称:", self.activity_name)
        #form_layout.addRow("活动内容:", self.activity_description)
        form_layout.addRow("活动地点:", self.activity_location)
        self.description_combo = QComboBox()
        self.description_combo.addItems(["唱歌", "跳舞", "篮球"])
        form_layout.addRow("活动内容:", self.description_combo)
        editor_layout.addLayout(form_layout)
        #form_layout.addRow("活动时间:", self.activity_time)
        self.schedule_group = QGroupBox("每周活动时间")
        self.schedule_group.setMinimumWidth(1600)
        self.schedule_layout = QVBoxLayout(self.schedule_group)
        self.scroll_area = QScrollArea()  # 使用滚动区域以适应多个时间段
        self.weekly_schedule_widget = WeeklyScheduleWidget()
        self.scroll_area.setWidget(self.weekly_schedule_widget)
        self.scroll_area.setWidgetResizable(True)
        self.schedule_layout.addWidget(self.scroll_area)
        editor_layout.addWidget(self.schedule_group)


        # Editor Buttons
        self.add_activity_button = QPushButton("添加活动")
        self.remove_activity_button = QPushButton("删除活动")
        editor_layout.addWidget(self.add_activity_button)
        editor_layout.addWidget(self.remove_activity_button)

        main_layout.addWidget(self.activity_editor)


class ClubInfoWidget(QWidget):
    def __init__(self):
        super().__init__()
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(QLabel("社团名称:"))
        self.club_name_edit = QLineEdit()
        main_layout.addWidget(self.club_name_edit)

        main_layout.addWidget(QLabel("社团主题:"))
        self.club_theme_combo = QComboBox()
        self.club_theme_combo.addItems(["艺术", "科学", "体育", "文学"])  # 示例主题
        main_layout.addWidget(self.club_theme_combo)

        # 添加社团门槛选择列表
        self.requirements_list_widget = QListWidget()
        self.requirements_list_widget.setSelectionMode(QListWidget.MultiSelection)
        self.load_requirements("./requirements.csv")  # 替换为您的 CSV 文件路径
        main_layout.addWidget(QLabel("社团门槛:"))
        main_layout.addWidget(self.requirements_list_widget)
        self.setLayout(main_layout)

    def load_requirements(self, csv_path):
        with open(csv_path, "r", encoding="utf-8") as file:
            reader = csv.reader(file)
            for row in reader:
                item = QListWidgetItem(row[0])
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)  # 添加复选框
                item.setCheckState(Qt.Unchecked)  # 默认未勾选
                self.requirements_list_widget.addItem(item)


class ClubWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.tabs = QTabWidget()
        self.info = ClubInfoWidget()
        self.activity = ClubActivityWidget()
        self.tabs.addTab(self.info, "社团信息")
        self.tabs.addTab(self.activity, "社团活动")
        self.main_layout = QVBoxLayout()
        self.main_layout.addWidget(self.tabs)
        self.setLayout(self.main_layout)


class ClubEditor(QMainWindow):
    def __init__(self):
        super().__init__()
        # 设置主窗口的中央组件
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # 主水平布局
        self.main_layout = QHBoxLayout(self.central_widget)

        # 左侧 - 社团列表
        club_list_group = QGroupBox("社团列表")
        club_list_group_layout = QVBoxLayout(club_list_group)
        self.club_list_widget = QListWidget()
        club_list_group_layout.addWidget(self.club_list_widget)
        self.main_layout.addWidget(club_list_group)
        self.club_list_widget.clicked.connect(self.on_club_selected)

        # 右侧 - 表单和操作按钮
        self.right_side_layout = QVBoxLayout()
        self.main_layout.addLayout(self.right_side_layout)

        # 菜单栏
        self.menu_bar = self.menuBar()
        self.file_menu = self.menu_bar.addMenu("文件")

        # "新建" 操作
        self.new_action = QAction("新建", self)
        self.new_action.triggered.connect(self.new_file)
        self.file_menu.addAction(self.new_action)

        # "打开" 操作
        self.open_action = QAction("打开", self)
        self.open_action.triggered.connect(self.open_file)
        self.file_menu.addAction(self.open_action)

        # "保存" 操作
        self.save_action = QAction("保存", self)
        self.save_action.triggered.connect(self.save_file)
        self.file_menu.addAction(self.save_action)

        # 表单元素
        self.club_widget = ClubWidget()
        self.right_side_layout.addWidget(self.club_widget)
        # 提交按钮
        self.submit_button = QPushButton("提交")
        self.right_side_layout.addWidget(self.submit_button)
        self.submit_button.clicked.connect(self.submit_club_info)

    def load_requirements(self, csv_path):
        with open(csv_path, "r", encoding="utf-8") as file:
            reader = csv.reader(file)
            for row in reader:
                item = QListWidgetItem(row[0])
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)  # 添加复选框
                item.setCheckState(Qt.Unchecked)  # 默认未勾选
                self.requirements_list_widget.addItem(item)

    def new_file(self):
        # 清空社团列表
        self.club_list_widget.clear()

    def open_file(self):
        # 打开文件对话框并读取 JSON 文件
        file_name, _ = QFileDialog.getOpenFileName(self, "打开文件", "", "JSON files (*.json)")
        if file_name:
            with open(file_name, 'r', encoding='utf-8') as file:
                clubs = json.load(file)
                self.club_list_widget.clear()
                for club in clubs:
                    self.club_list_widget.addItem(club['name'])

    def save_file(self):
        # 打开文件保存对话框并保存社团数据到 JSON 文件
        file_name, _ = QFileDialog.getSaveFileName(self, "保存文件", "", "JSON files (*.json)")
        if file_name:
            clubs = []
            for index in range(self.club_list_widget.count()):
                club_name = self.club_list_widget.item(index).text()
                clubs.append({'name': club_name})
            with open(file_name, 'w', encoding='utf-8') as file:
                json.dump(clubs, file, ensure_ascii=False, indent=4)

    def on_club_selected(self, index):
        # 处理社团列表中选中的项目
        club_name = self.club_list_widget.item(index.row()).text()
        print(f"Selected Club: {club_name}")

    def submit_club_info(self):
        # 处理提交按钮的点击事件
        club_name = self.club_name_edit.text()
        self.club_list_widget.addItem(club_name)
        self.club_name_edit.clear()
        # 根据需要更新其他字段
        # ...

# 运行应用程序
if __name__ == "__main__":
    app = QApplication(sys.argv)

    mainWin = ClubEditor()
    mainWin.show()

    sys.exit(app.exec())

