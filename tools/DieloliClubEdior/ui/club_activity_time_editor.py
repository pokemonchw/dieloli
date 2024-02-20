import uuid
from PySide6.QtWidgets import (
    QGroupBox, QVBoxLayout, QScrollArea, QWidget, QHBoxLayout, QSizePolicy,
    QFormLayout, QTimeEdit, QPushButton
)
from PySide6.QtCore import Qt, QTime
import cache_control
import game_type


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
        self.uid = str(uuid.uuid4())
        self.layout = QFormLayout(self)
        self.start_time = QTimeEdit(QTime(8, 0))
        self.end_time = QTimeEdit(QTime(10, 0))
        self.delete_button = QPushButton("删除")
        self.delete_button.clicked.connect(self.delete_slot)
        self.layout.addRow("开始时间:", self.start_time)
        self.layout.addRow("结束时间:", self.end_time)
        self.layout.addRow(self.delete_button)

    def delete_slot(self):
        parent = self.parent()
        del parent.time_slot_widgets[self.uid]
        self.setParent(None)
        self.deleteLater()


class DayScheduleWidget(QWidget):
    """ 活动日容器 """

    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout(self)
        self.time_slot_widgets = {}

    def add_time_slot(self):
        time_slot_widget = TimeSlotWidget(self)
        self.time_slot_widgets[time_slot_widget.uid] = time_slot_widget
        self.layout.addWidget(time_slot_widget)


class WeeklyScheduleWidget(QWidget):
    """ 每周活动时间面板 """

    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout(self)
        layout = QHBoxLayout()
        self.day_widgets = {}
        days = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
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
            layout.addLayout(day_column_layout)
        self.layout.addLayout(layout)
        save_button = QPushButton("保存")
        save_button.clicked.connect(self.save_activity_time)
        self.layout.addWidget(save_button)
        cache_control.update_activity_time_signal.connect(self._update)

    def save_activity_time(self):
        """ 保存活动时间 """
        if cache_control.now_club_id == "":
            return
        if cache_control.now_activity_id == "":
            return
        club_data = cache_control.club_list_data[cache_control.now_club_id]
        activity_data: game_type.ClubActivityData = club_data.activity_list[cache_control.now_activity_id]
        days_data = {
            "周一":0,
            "周二":1,
            "周三":2,
            "周四":3,
            "周五":4,
            "周六":5,
            "周日":6
        }
        for day in self.day_widgets:
            day_widget: DayScheduleWidget = self.day_widgets[day]
            for time_uid in day_widget.time_slot_widgets:
                time_slot_widget: TimeSlotWidget = day_widget.time_slot_widgets[time_uid]
                now_time_data = game_type.ClubActivityTimeData()
                now_time_data.uid = time_uid
                now_time_data.week_day = days_data[day]
                start_time: QTime = time_slot_widget.start_time.time()
                now_time_data.start_hour = start_time.hour()
                now_time_data.start_minute = start_time.minute()
                end_time: QTime = time_slot_widget.end_time.time()
                now_time_data.end_hour = end_time.hour()
                now_time_data.end_minute = end_time.minute()
                activity_data.activity_time_list[time_uid] = now_time_data

    def _update(self):
        for day in self.day_widgets:
            day_schedule_widget: DayScheduleWidget = self.day_widgets[day]
            time_uid_list = list(day_schedule_widget.time_slot_widgets.keys())
            for time_uid in time_uid_list:
                time_slot_widget = day_schedule_widget.time_slot_widgets[time_uid]
                time_slot_widget.delete_slot()
        if cache_control.now_club_id == "":
            return
        if cache_control.now_activity_id == "":
            return
        days = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
        club_data = cache_control.club_list_data[cache_control.now_club_id]
        activity_data: game_type.ClubActivityData = club_data.activity_list[cache_control.now_activity_id]
        for time_uid in activity_data.activity_time_list:
            now_data = activity_data.activity_time_list[time_uid]
            day = days[now_data.week_day]
            day_widget: DayScheduleWidget = self.day_widgets[day]
            time_slot_widget = TimeSlotWidget(day_widget)
            time_slot_widget.uid = now_data.uid
            time_slot_widget.start_time = QTimeEdit(QTime(now_data.start_hour, now_data.start_minute))
            time_slot_widget.end_time = QTimeEdit(QTime(now_data.end_hour, now_data.end_minute))
            day_widget.layout.addWidget(time_slot_widget)
            day_widget.time_slot_widgets[now_data.uid] = time_slot_widget


class ClubActivityTimeEditor(QGroupBox):
    """ 社团活动时间编辑面板 """

    def __init__(self):
        super().__init__("每周活动时间")
        self.setMinimumWidth(1600)
        self.main_layout = QVBoxLayout(self)
        scroll_area = QScrollArea()
        self.weekly_schedule_widget = WeeklyScheduleWidget()
        scroll_area.setWidget(self.weekly_schedule_widget)
        scroll_area.setWidgetResizable(True)
        self.main_layout.addWidget(scroll_area)

