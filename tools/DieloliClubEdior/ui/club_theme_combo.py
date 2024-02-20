from PySide6.QtWidgets import QComboBox
import cache_control

class ClubThemeCombo(QComboBox):
    """ 社团主题选单 """

    def __init__(self):
        super().__init__()
        self.addItems(list(cache_control.club_theme_data.keys()))
        if cache_control.now_club_id == "":
            return
        if cache_control.now_club_id not in cache_control.club_list_data:
            return
        club_data = cache_control.club_list_data[cache_control.now_club_id]
