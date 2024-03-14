from PySide6.QtWidgets import QComboBox
import cache_control

class ClubThemeCombo(QComboBox):
    """ 社团主题选单 """

    def __init__(self):
        super().__init__()
        self.addItems(list(cache_control.club_theme_data.keys()))
        cache_control.update_signal.connect(self._update)
        self.currentIndexChanged.connect(self._set_club_theme)
        if cache_control.now_club_id == "":
            return
        if cache_control.now_club_id not in cache_control.club_list_data:
            return
        club_data = cache_control.club_list_data[cache_control.now_club_id]
        self.setCurrentIndex(club_data.theme)

    def _update(self):
        """ 更新主题设置 """
        if cache_control.now_club_id == "":
            return
        if cache_control.now_club_id not in cache_control.club_list_data:
            return
        club_data = cache_control.club_list_data[cache_control.now_club_id]
        self.setCurrentIndex(club_data.theme)

    def _set_club_theme(self, index):
        """ 设置社团主题 """
        if cache_control.now_club_id == "":
            return
        if cache_control.now_club_id not in cache_control.club_list_data:
            return
        club_data = cache_control.club_list_data[cache_control.now_club_id]
        club_data.theme = index

