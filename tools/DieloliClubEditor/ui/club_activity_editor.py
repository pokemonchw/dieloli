from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QLineEdit
)
from PySide6.QtCore import QEvent
from ui.club_activity_time_editor import ClubActivityTimeEditor
from ui.club_activity_description_combo import ClubActivityDescriptionCombo
from ui.club_activity_position_combo import ClubActivityPositionCombo
import cache_control


class ActivityNameEdit(QLineEdit):
    """ 活动名称输入框 """

    def __init__(self):
        """ 初始化输入框 """
        super().__init__()
        self.installEventFilter(self)
        cache_control.update_signal.connect(self._update)
        self._update()
        self.inside = False

    def _update(self):
        if cache_control.now_club_id == "":
            if len(cache_control.club_list_data):
                cache_control.now_club_id = list(cache_control.club_list_data.keys())[0]
        if cache_control.now_club_id == "":
            return
        club_data = cache_control.club_list_data[cache_control.now_club_id]
        if cache_control.now_activity_id not in club_data.activity_list:
            cache_control.now_activity_id = ""
        if cache_control.now_activity_id == "":
            if len(club_data.activity_list):
                cache_control.now_activity_id = list(club_data.activity_list.keys())[0]
        if cache_control.now_activity_id == "":
            return
        activity_data = club_data.activity_list[cache_control.now_activity_id]
        self.setText(activity_data.name)

    def eventFilter(self, obj, event):
        # 检测焦点丢失事件
        now_judge = False
        if event.type() == QEvent.MouseMove:
            # 更新鼠标是否在QLineEdit内的状态
            self.inside = self.rect().contains(event.pos())
        if event.type() == QEvent.FocusOut:
            now_judge = True
        if now_judge:
            if cache_control.now_club_id == "":
                if len(cache_control.club_list_data):
                    cache_control.now_club_id = list(cache_control.club_list_data.keys())[0]
            if cache_control.now_club_id == "":
                return
            club_data = cache_control.club_list_data[cache_control.now_club_id]
            if cache_control.now_activity_id == "":
                if len(club_data.activity_list):
                    cache_control.now_activity_id = list(club_data.activity_list.keys())[0]
            if cache_control.now_activity_id == "":
                return
            activity_data = club_data.activity_list[cache_control.now_activity_id]
            activity_data.name = self.text()
            cache_control.update_signal.emit()
        return super().eventFilter(obj, event)


class ClubActivityEditor(QWidget):
    """ 社团活动信息编辑面板 """

    def __init__(self):
        super().__init__()
        main_layout = QVBoxLayout(self)
        form_layout = QFormLayout()
        self.activity_name = ActivityNameEdit()
        self.activity_localtion = QLineEdit()
        self.positon_combo = ClubActivityPositionCombo()
        form_layout.addRow("活动名称:", self.activity_name)
        form_layout.addRow("活动地点:", self.positon_combo)
        self.description_combo = ClubActivityDescriptionCombo(list(cache_control.activity_list_data.keys()))
        form_layout.addRow("活动内容:", self.description_combo)
        main_layout.addLayout(form_layout)
        activity_time = ClubActivityTimeEditor()
        main_layout.addWidget(activity_time)
        self.setLayout(main_layout)

