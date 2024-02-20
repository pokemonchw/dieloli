from PySide6.QtWidgets import QListWidget, QListWidgetItem
import cache_control

class SelectRequirementsList(QListWidget):
    """ 已经选择的门槛列表 """

    def __init__(self):
        super().__init__()
        self._update()
        cache_control.update_signal.connect(self._update)
        cache_control.update_premise_signal.connect(self._update)

    def _update(self):
        """ 刷新门槛列表 """
        self.clear()
        if cache_control.now_club_id == "":
            return
        if cache_control.now_club_id not in cache_control.club_list_data:
            return
        club_data = cache_control.club_list_data[cache_control.now_club_id]
        for premise_id in club_data.premise_data:
            item = QListWidgetItem(cache_control.premise_data[premise_id])
            item.setToolTip(item.text())
            self.addItem(item)

