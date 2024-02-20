import os
import json
from PySide6.QtWidgets import QMenuBar, QMenu, QWidgetAction, QFileDialog
import cache_control
import json_handle
import game_type

class MenuBar(QMenuBar):
    """ 顶部菜单栏 """

    def __init__(self):
        """ 初始化顶部菜单栏 """
        super().__init__()
        self.file_menu = FileMenu()
        self.addMenu(self.file_menu)

class FileMenu(QMenu):
    """ 文件选项菜单 """

    def __init__(self):
        super().__init__("文件")
        self.select_club_file_action = QWidgetAction(self)
        self.select_club_file_action.setText("选择社团数据文件 Ctrl+O")
        self.new_club_file_action = QWidgetAction(self)
        self.new_club_file_action.setText("新建社团数据文件 Ctrl+N")
        self.save_club_file_action = QWidgetAction(self)
        self.save_club_file_action.setText("保存社团数据文件 Ctrl+S")
        self.exit_action = QWidgetAction(self)
        self.exit_action.setText("关闭编辑器       Ctrl+Q")
        self.addActions(
            [
                self.select_club_file_action,
                self.new_club_file_action,
                self.save_club_file_action,
                self.exit_action,
            ]
        )
        self.select_club_file_action.triggered.connect(self._load_club_data)
        self.new_club_file_action.triggered.connect(self._create_club_data)
        self.save_club_file_action.triggered.connect(self._save_club_data)
        self.exit_action.triggered.connect(self._exit_editor)

    def _load_club_data(self):
        """ 载入社团数据文件 """
        now_file = QFileDialog.getOpenFileName(self, "选择文件", ".", "*.json")
        file_path = now_file[0]
        if file_path:
            cache_control.now_file_path = file_path
            now_data = json_handle.load_json(file_path)
            i = 0
            for k in now_data:
                if not i:
                    cache_control.now_club_id = k
                i += 1
                club_data = game_type.ClubData()
                club_data.__dict__ = now_data[k]
                for activity_uid in club_data.activity_list:
                    activity_dict = club_data.activity_list[activity_uid]
                    activity_data = game_type.ClubActivityData()
                    activity_data.__dict__ = activity_dict
                    for time_uid in activity_data.activity_time_list:
                        time_dict = activity_data.activity_time_list[time_uid]
                        time_data = game_type.ClubActivityTimeData()
                        time_data.__dict__ = time_dict
                        activity_data.activity_time_list[time_uid] = time_data
                    club_data.activity_list[activity_uid] = activity_data
                cache_control.club_list_data[club_data.uid] = club_data
            cache_control.update_signal.emit()

    def _create_club_data(self):
        """ 新建社团数据文件 """
        dialog = QFileDialog()
        dialog.setFileMode(QFileDialog.AnyFile)
        dialog.setNameFilter("Json (*.json)")
        if dialog.exec():
            file_names = dialog.selectedFiles()
            file_path = file_names[0]
            if not file_path.endswith(".json"):
                file_path += ".json"
                cache_control.now_file_path = file_path
                cache_control.club_list_data = {}
                cache_control.update_signal.emit()
                cache_control.update_premise_signal.emit()
                cache_control.update_activity_time_signal.emit()

    def _save_club_data(self):
        """ 保存社团数据文件 """
        if cache_control.now_file_path:
            with open(cache_control.now_file_path, "w", encoding="utf-8") as club_data_file:
                now_data = {}
                for ck in cache_control.club_list_data:
                    club_data = cache_control.club_list_data[ck]
                    club_dict = {}
                    for k in club_data.__dict__:
                        if k != "activity_list":
                            club_dict[k] = club_data.__dict__[k]
                        else:
                            activity_list_dict = {}
                            club_dict.setdefault(k ,{})
                            for activity_id in club_data.activity_list:
                                activity_data = club_data.activity_list[activity_id]
                                activity_dict = {}
                                for ak in activity_data.__dict__:
                                    if ak != "activity_time_list":
                                        activity_dict[ak] = activity_data.__dict__[ak]
                                    else:
                                        activity_dict.setdefault(ak, {})
                                        for time_uid in activity_data.activity_time_list:
                                            time_data = activity_data.activity_time_list[time_uid]
                                            activity_dict[k][time_uid] = time_data.__dict__
                                activity_list_dict[ak] = activity_dict
                            club_dict[k] = activity_list_dict
                    now_data[ck] = club_dict
                json.dump(now_data, club_data_file, ensure_ascii=0)

    def _exit_editor(self):
        """ 关闭编辑器 """
        os._exit(0)

