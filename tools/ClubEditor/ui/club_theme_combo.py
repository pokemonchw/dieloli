from PySide6.QtWidgets import QComboBox
import cache_control

class ClubThemeCombo(QComboBox):
    """ 社团主题选单 """

    def __init__(self):
        super().__init__()
        self.addItems(list(cache_control.club_theme_data.keys()))
