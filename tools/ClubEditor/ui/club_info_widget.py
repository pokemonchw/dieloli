from PySide6.QtWidgets import QWidget, QVBoxLayout, QLineEdit, QLabel
from ui.club_theme_combo import ClubThemeCombo
from ui.club_requments import ClubRequments


class ClubInfoWidget(QWidget):
    """ 社团信息面板 """

    def __init__(self):
        super().__init__()
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(QLabel("社团名称:"))
        self.club_name_edit = QLineEdit()
        main_layout.addWidget(self.club_name_edit)
        main_layout.addWidget(QLabel("社团主题:"))
        club_theme_combo = ClubThemeCombo()
        main_layout.addWidget(club_theme_combo)
        main_layout.addWidget(QLabel("社团门槛:"))
        club_requments = ClubRequments()
        main_layout.addLayout(club_requments)

