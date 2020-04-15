import os
from Script.Core import game_path_config, json_handle

game_path = game_path_config.game_path


def load_config_data() -> dict:
    """
    读取游戏配置数据
    """
    config_path = os.path.join(game_path, "data", "core_cfg.json")
    config_data = json_handle.load_json(config_path)
    return config_data


def get_font_data(font_id: str) -> dict:
    """
    读取游戏字体样式配置数据
    Keyword arguments:
    list_id -- 字体样式
    """
    font_path = os.path.join(game_path, "data", "FontConfig.json")
    font_data = json_handle.load_json(font_path)
    return font_data[font_id]


def get_font_data_list() -> list:
    """
    读取游戏字体样式配置列表
    """
    font_path = os.path.join(game_path, "data", "FontConfig.json")
    font_data = json_handle.load_json(font_path)
    font_list = list(font_data.keys())
    return font_list


# 配置数据定义
config_data = load_config_data()
game_name = config_data["game_name"]
verson = config_data["verson"]
author = config_data["author"]
verson_time = config_data["verson_time"]
background_color = config_data["background_color"]
language = config_data["language"]
window_width = config_data["window_width"]
window_hight = config_data["window_hight"]
textbox_width = config_data["textbox_width"]
textbox_hight = config_data["textbox_hight"]
text_width = int(config_data["text_width"])
text_hight = int(config_data["text_hight"])
inputbox_width = int(config_data["inputbox_width"])
cursor = config_data["cursor"]
year = config_data["year"]
month = config_data["month"]
day = config_data["day"]
hour = config_data["hour"]
minute = config_data["minute"]
max_save = config_data["max_save"]
save_page = config_data["save_page"]
character_list_show = config_data["characterlist_show"]
text_wait = config_data["text_wait"]
home_url = config_data["home_url"]
licenses_url = config_data["licenses_url"]
random_npc_max = config_data["random_npc_max"]
proportion_teacher = config_data["proportion_teacher"]
proportion_student = config_data["proportion_student"]
threading_pool_max = config_data["threading_pool_max"]
in_scene_see_player_max = config_data["insceneseeplayer_max"]
see_character_clothes_max = config_data["seecharacterclothes_max"]
see_character_wearitem_max = config_data["seecharacterwearitem_max"]
see_character_item_max = config_data["seecharacteritem_max"]
