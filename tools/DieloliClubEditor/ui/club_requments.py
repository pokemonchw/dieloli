from PySide6.QtWidgets import QHBoxLayout, QWidget
from ui.select_requirements_list import SelectRequirementsList
from ui.availabe_requirements_list import AvailabeRequirementsList

class ClubRequments(QHBoxLayout):
    """ 社团门槛面板 """

    def __init__(self):
        super().__init__()
        selected_requirements_list = SelectRequirementsList()
        self.addWidget(selected_requirements_list,1)
        availabe_requirements_list = AvailabeRequirementsList()
        self.addWidget(availabe_requirements_list,3)
