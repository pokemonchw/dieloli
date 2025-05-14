from PySide6.QtWidgets import QMenuBar, QMenu, QFileDialog, QWidgetAction
import json
import cache_control
import json_handle
import game_type


class MenuBar(QMenuBar):
    """ 顶部菜单栏 """

    def __init__(self):
        super().__init__()
        self.file_menu = FileMenu()
        self.addMenu(self.file_menu)


class FileMenu(QMenu):
    """ 文件选择菜单 """

    def __init__(self):
        super().__init__("文件")
        self.select_clothing_file_action = QWidgetAction(self)
        self.select_clothing_file_action.setText("选择服装数据文件 Ctrl+O")
        self.new_clothing_file_action = QWidgetAction(self)
        self.new_clothing_file_action.setText("新建服装数据文件 Ctrl+N")
        self.save_clothing_file_action = QWidgetAction(self)
        self.save_clothing_file_action.setText("保存服装数据文件 Ctrl+S")
        self.exit_action = QWidgetAction(self)
        self.exit_action.setText("关闭编辑器       Ctrl+Q")
        self.addActions(
            [
self.select_clothing_file_action,
                self.new_clothing_file_action,
                self.save_clothing_file_action,
                self.exit_action,
            ]
        )
        self.select_clothing_file_action.triggered.connect(self._load_clothing_data)
        self.new_clothing_file_action.triggered.connect(self._create_clothing_data)
        self.save_clothing_file_action.triggered.connect(self._save_clothing_data)
        self.exit_action.triggered.connect(self._exit_editor)

    def _load_clothing_data(self):
        """ 载入服装数据文件 """
        now_file = QFileDialog.getOpenFileName(self, "选择文件", ".", "*.json")
        file_path = now_file[0]
        if file_path:
            cache_control.now_file_path = file_path
            now_data = json_handle.load_json(file_path)
            i = 0
            for k in now_data["clothing"]:
                if not i:
                    cache_control.now_clothing_id = k
                i += 1
                clothing_data = game_type.ClothingTem()
                clothing_data.__dict__ = now_data["clothing"][k]
                cache_control.clothing_list_data[k] = clothing_data
            i = 0
            for k in now_data["suit"]:
                if not i:
                    cache_control.now_suit_id = k
                i += 1
                suit_data = game_type.ClothingSuit()
                suit_data.__dict__ = now_data["suit"][k]
                cache_control.suit_list_data[k] = suit_data
            cache_control.update_signal.emit()

    def _create_clothing_data(self):
        """ 新建服装数据文件 """
        dialog = QFileDialog()
        dialog.setFileMode(QFileDialog.AnyFile)
        dialog.setNameFilter("Json (*.json)")
        if dialog.exec():
            file_names = dialog.selectedFiles()
            file_path = file_names[0]
            if not file_path.endswith(".json"):
                file_path += ".json"
                cache_control.now_file_path = file_path
                cache_control.clothing_list_data = {}
                cache_control.suit_list_data = {}
                cache_control.update_signal.emit()

    def _save_clothing_data(self):
        """ 保存服装数据文件 """
        if cache_control.now_file_path:
            with open(cache_control.now_file_path, "w", encoding="utf-8") as clothing_data_file:
                all_clothing_data_dict = {}
                for clothing_key in cache_control.clothing_list_data:
                    clothing_data = cache_control.clothing_list_data[clothing_key]
                    all_clothing_data_dict[clothing_key] = clothing_data.__dict__
                all_suit_data_dict = {}
                for suit_key in cache_control.suit_list_data:
                    suit_data = cache_control.suit_list_data[suit_key]
                    all_suit_data_dict[suit_key] = suit_data.__dict__
                all_data = {}
                all_data["clothing"] = all_clothing_data_dict
                all_data["suit"] = all_suit_data_dict
                json.dump(all_data, clothing_data_file, ensure_ascii=0)

    def _exit_editor(self):
        """ 关闭编辑器 """
        os._exit(0)

