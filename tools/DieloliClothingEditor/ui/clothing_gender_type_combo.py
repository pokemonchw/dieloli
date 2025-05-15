from PySide6.QtWidgets import QComboBox
import cache_control

class ClothingGenderTypeCombo(QComboBox):
    """ 服装性别类型选单 """

    def __init__(self):
        super().__init__()
        self.addItems(list(cache_control.gender_type.keys()))
        cache_control.update_signal.connect(self._update)
        self.currentIndexChanged.connect(self._set_gender_type)
        if cache_control.now_clothing_id == "":
            return
        clothing_data = cache_control.clothing_list_data[cache_control.now_clothing_id]
        self.setCurrentIndex(clothing_data.sex)

    def _update(self):
        """ 更新性别类型设置 """
        if cache_control.now_clothing_id == "":
            return
        clothing_data = cache_control.clothing_list_data[cache_control.now_clothing_id]
        self.setCurrentIndex(clothing_data.sex)

    def _set_gender_type(self, index):
        """ 设置服装性别类型 """
        if cache_control.now_clothing_id == "":
            return
        clothing_data = cache_control.clothing_list_data[cache_control.now_clothing_id]
        clothing_data.sex = index
