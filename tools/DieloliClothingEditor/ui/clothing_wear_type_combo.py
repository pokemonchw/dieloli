from PySide6.QtWidgets import QComboBox
import cache_control

class ClothingWearTypeCombo(QComboBox):
    """ 服装穿戴位置类型选单 """

    def __init__(self):
        super().__init__()
        self.addItems(list(cache_control.wear_type.keys()))
        cache_control.update_signal.connect(self._update)
        self.currentIndexChanged.connect(self._set_wear_type)
        if cache_control.now_clothing_id == "":
            return
        clothing_data = cache_control.clothing_list_data[cache_control.now_clothing_id]
        self.setCurrentIndex(clothing_data.clothing_type)

    def _update(self):
        """ 更新穿戴类型设置 """
        if cache_control.now_clothing_id == "":
            return
        clothing_data = cache_control.clothing_list_data[cache_control.now_clothing_id]
        self.setCurrentIndex(clothing_data.clothing_type)

    def _set_wear_type(self, index):
        """ 设置服装穿戴类型 """
        if cache_control.now_clothing_id == "":
            return
        clothing_data = cache_control.clothing_list_data[cache_control.now_clothing_id]
        clothing_data.clothing_type = index
