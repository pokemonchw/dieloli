import os
import pickle
from typing import List
import cache_control

all_scene_data_path = os.path.join("..", "..", "data", "SceneData")
""" 预处理的所有场景数据路径 """
all_map_data_path = os.path.join("..", "..", "data", "MapData")
""" 预处理的所有地图数据路径 """


def init_map_data():
    """载入地图和场景数据"""
    if (
        os.path.exists(all_scene_data_path)
        and os.path.exists(all_map_data_path)
    ):
        with open(all_scene_data_path, "rb") as all_scene_data_file:
            cache_control.scene_data = pickle.load(all_scene_data_file)
        with open(all_map_data_path, "rb") as all_map_data_file:
            cache_control.map_data = pickle.load(all_map_data_file)
    else:
        print("载入地图数据失败")


def get_map_system_path_for_str(path_str: str) -> list:
    """
    将地图系统路径文本转换为地图系统路径
    """
    return path_str.split(os.sep)


def get_map_system_path_str(now_path: List[str]) -> str:
    """
    将游戏地图系统路径转换为字符串
    Keyword arguments:
    now_path -- 游戏地图路径
    Return arguments:
    str -- 地图路径字符串
    """
    return os.sep.join(now_path)

