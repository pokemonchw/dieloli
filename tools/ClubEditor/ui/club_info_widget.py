from PySide6.QtWidgets import QWidget, QVBoxLayout, QLineEdit, QLabel
from PySide6.QtCore import QEvent
from ui.club_theme_combo import ClubThemeCombo
from ui.club_requments import ClubRequments
import cache_control


class ClubNameEdit(QLineEdit):
    """ 社团名称输入框 """

    def __init__(self):
        """ 初始化输入框 """
        super().__init__()
        self.installEventFilter(self)
        cache_control.update_signal.connect(self._update)
        self._update()

    def _update(self):
        if cache_control.now_club_id in cache_control.club_list_data:
            club_data = cache_control.club_list_data[cache_control.now_club_id]
            self.setText(club_data.name)

    def eventFilter(self, obj, event):
        # 检测焦点丢失事件
        if event.type() == QEvent.FocusOut:
            if cache_control.now_club_id != "":
                if cache_control.now_club_id in cache_control.club_list_data:
                    club_data = cache_control.club_list_data[cache_control.now_club_id]
                    club_data.name = self.text()
                    cache_control.update_signal.emit()
        return super().eventFilter(obj, event)


class ClubInfoWidget(QWidget):
    """ 社团信息面板 """

    def __init__(self):
        super().__init__()
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(QLabel("社团名称:"))
        self.club_name_edit = ClubNameEdit()
        main_layout.addWidget(self.club_name_edit)
        main_layout.addWidget(QLabel("社团主题:"))
        self.club_theme_combo = ClubThemeCombo()
        main_layout.addWidget(self.club_theme_combo)
        main_layout.addWidget(QLabel("社团门槛:"))
        self.club_requments = ClubRequments()
        main_layout.addLayout(self.club_requments)
        cache_control.update_signal.connect(self._update)

    def _update(self):
        """ 更新社团信息 """
        if cache_control.now_club_id not in cache_control.club_list_data:
            cache_control.now_club_id = ""
        if cache_control.now_club_id == "":
            self.club_name_edit.setText("")
            self.club_theme_combo.setCurrentIndex(0)
            return
        club_data = cache_control.club_list_data[cache_control.now_club_id]
        self.club_name_edit.setText(club_data.name)
        self.club_theme_combo.setCurrentIndex(club_data.theme)


