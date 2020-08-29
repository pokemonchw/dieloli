import os
import pickle
import shutil
from Script.Core import (
    era_print,
    text_loading,
    cache_contorl,
    game_config,
    game_path_config,
)
from Script.Design import character_handle

game_path = game_path_config.game_path


def get_save_dir_path(save_id: str) -> str:
    """
    按存档id获取存档所在系统路径
    Keyword arguments:
    save_id -- 存档id
    """
    save_path = os.path.join(game_path, "save")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    return os.path.join(save_path, save_id)


def judge_save_file_exist(save_id: str) -> bool:
    """
    判断存档id对应的存档是否存在
    Keyword arguments:
    save_id -- 存档id
    """
    save_path = get_save_dir_path(save_id)
    return os.path.exists(save_path)


def establish_save(save_id: str):
    """
    将游戏数据存入指定id的存档内
    Keyword arguments:
    save_id -- 存档id
    """
    character_data = cache_contorl.character_data
    game_time = cache_contorl.game_time
    scane_data = cache_contorl.scene_data
    map_data = cache_contorl.map_data
    npc_tem_data = cache_contorl.npc_tem_data
    random_npc_list = cache_contorl.random_npc_list
    game_verson = game_config.verson
    total_bodyfat_by_age = cache_contorl.total_bodyfat_by_age
    average_bodyfat_by_age = cache_contorl.average_bodyfat_by_age
    total_number_of_people_of_all_ages = cache_contorl.total_number_of_people_of_all_ages
    total_height_by_age = cache_contorl.total_height_by_age
    average_height_by_age = cache_contorl.average_height_by_age
    recipe_data = cache_contorl.recipe_data
    save_verson = {
        "game_verson": game_verson,
        "game_time": game_time,
        "character_name": character_data[0].name,
    }
    data = {
        "1": character_data,
        "2": game_time,
        "0": save_verson,
        "3": scane_data,
        "4": map_data,
        "5": npc_tem_data,
        "6": random_npc_list,
        "7": total_bodyfat_by_age,
        "8": average_bodyfat_by_age,
        "9": total_number_of_people_of_all_ages,
        "10": total_height_by_age,
        "11": average_height_by_age,
        "12": recipe_data,
    }
    for data_id in data:
        write_save_data(save_id, data_id, data[data_id])


def load_save_info_head(save_id: str) -> dict:
    """
    获取存档的头部信息
    Keyword arguments:
    save_id -- 存档id
    """
    save_path = get_save_dir_path(save_id)
    file_path = os.path.join(save_path, "0")
    with open(file_path, "rb") as f:
        return pickle.load(f)


def write_save_data(save_id: str, data_id: str, write_data: dict):
    """
    将存档数据写入文件
    Keyword arguments:
    save_id -- 存档id
    data_id -- 要写入的数据在存档下的文件id
    write_data -- 要写入的数据
    """
    save_path = get_save_dir_path(save_id)
    file_path = os.path.join(save_path, data_id)
    if not judge_save_file_exist(save_id):
        os.makedirs(save_path)
    with open(file_path, "wb") as f:
        pickle.dump(write_data, f)


def load_save(save_id: str) -> dict:
    """
    按存档id读取存档数据
    Keyword arguments:
    save_id -- 存档id
    """
    save_path = get_save_dir_path(save_id)
    data = {}
    file_list = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"]
    for file_name in file_list:
        file_path = os.path.join(save_path, file_name)
        with open(file_path, "rb") as f:
            data[file_name] = pickle.load(f)
    return data


def input_load_save(save_id: str):
    """
    载入存档存档id对应数据，覆盖当前游戏内存
    Keyword arguments:
    save_id -- 存档id
    """
    save_data = load_save(save_id)
    cache_contorl.character_data = save_data["1"]
    cache_contorl.game_time = save_data["2"]
    cache_contorl.scene_data = save_data["3"]
    cache_contorl.map_data = save_data["4"]
    cache_contorl.npc_tem_data = save_data["5"]
    cache_contorl.random_npc_list = save_data["6"]
    cache_contorl.total_bodyfat_by_age = save_data["7"]
    cache_contorl.average_bodyfat_by_age = save_data["8"]
    cache_contorl.total_number_of_people_of_all_ages = save_data["9"]
    cache_contorl.total_height_by_age = save_data["10"]
    cache_contorl.average_height_by_age = save_data["11"]
    cache_contorl.recipe_data = save_data["12"]
    character_handle.init_character_position()


def get_save_page_save_id(page_save_value: int, input_id: int) -> int:
    """
    按存档页计算，当前页面输入数值对应存档id
    Keyword arguments:
    page_save_value -- 存档页Id
    input_id -- 当前输入数值
    """
    save_panel_page = int(cache_contorl.panel_state["SeeSaveListPanel"]) + 1
    start_save_id = page_save_value * (save_panel_page - 1)
    save_id = start_save_id + input_id
    return int(save_id)


def remove_save(save_id: str):
    """
    删除存档id对应存档
    Keyword arguments:
    save_id -- 存档id
    """
    save_path = get_save_dir_path(save_id)
    if os.path.isdir(save_path):
        shutil.rmtree(save_path)
    else:
        error_text = text_loading.get_text_data(text_loading.error_path, "not_save_error")
        era_print.line_feed_print(error_text)
