import os
import pickle
import shutil
import datetime
from Script.Core import (
    cache_control,
    game_path_config,
    game_type,
)
from Script.Design import character_handle
from Script.Config import normal_config

game_path = game_path_config.game_path
cache: game_type.Cache = cache_control.cache
""" 游戏缓存数据 """


def get_save_dir_path(save_id: str) -> str:
    """
    按存档id获取存档所在系统路径
    Keyword arguments:
    save_id -- 存档id
    """
    save_path = os.path.join("save")
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
    save_verson = {
        "game_verson": normal_config.config_normal.verson,
        "game_time": cache.game_time,
        "character_name": cache.character_data[0].name,
        "save_time": datetime.datetime.now(),
    }
    data = {
        "1": cache,
        "0": save_verson,
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
    Return arguments:
    game_type.Cache -- 游戏缓存数据
    """
    save_path = get_save_dir_path(save_id)
    file_path = os.path.join(save_path, "1")
    with open(file_path, "rb") as f:
        return pickle.load(f)


def input_load_save(save_id: str):
    """
    载入存档存档id对应数据，覆盖当前游戏内存
    Keyword arguments:
    save_id -- 存档id
    """
    cache.__dict__ = load_save(save_id).__dict__


def remove_save(save_id: str):
    """
    删除存档id对应存档
    Keyword arguments:
    save_id -- 存档id
    """
    save_path = get_save_dir_path(save_id)
    if os.path.isdir(save_path):
        shutil.rmtree(save_path)
