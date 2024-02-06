from PySide6.QtWidgets import (
    QGroupBox, QVBoxLayout, QScrollArea, QWidget, QHBoxLayout, QSizePolicy,
    QFormLayout, QTimeEdit, QPushButton
)
from PySide6.QtCore import Qt, QTime


class DayScheduleScrollArea(QScrollArea):
    def __init__(self, day_widget):
        super().__init__()
        self.setWidget(day_widget)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)


class TimeSlotWidget(QGroupBox):
    """ 活动时间组件 """

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
    """ 活动日容器 """

    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout(self)
        self.time_slot_widgets = []

    def add_time_slot(self):
        time_slot_widget = TimeSlotWidget(self)
        self.time_slot_widgets.append(time_slot_widget)
        self.layout.addWidget(time_slot_widget)


class WeeklyScheduleWidget(QWidget):
    """ 每周活动时间面板 """

    def __init__(self):
        super().__init__()
        self.layout = QHBoxLayout(self)
        days = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
        self.day_widgets = {}
        for day in days:
            day_column_layout = QVBoxLayout()
            # 添加天的标签
            day_group = QGroupBox(day)
            day_group_layout = QVBoxLayout(day_group)
            # 创建活动日容器
            day_schedule_widget = DayScheduleWidget()
            self.day_widgets[day] = day_schedule_widget
            scroll_area = DayScheduleScrollArea(day_schedule_widget)
            day_group_layout.addWidget(scroll_area)
            # 添加时间段的按钮
            add_time_slot_button = QPushButton("添加活动时间")
            add_time_slot_button.clicked.connect(day_schedule_widget.add_time_slot)
            day_group_layout.addWidget(add_time_slot_button)
            day_column_layout.addWidget(day_group)
            self.layout.addLayout(day_column_layout)


class ClubActivityTimeEditor(QGroupBox):
    """ 社团活动时间编辑面板 """

    def __init__(self):
        super().__init__("每周活动时间")
        self.setMinimumWidth(1600)
        self.main_layout = QVBoxLayout(self)
        scroll_area = QScrollArea()
        weekly_schedule_widget = WeeklyScheduleWidget()
        scroll_area.setWidget(weekly_schedule_widget)
        scroll_area.setWidgetResizable(True)
        self.main_layout.addWidget(scroll_area)

