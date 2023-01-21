import random
import bisect
from typing import List
from Script.Core import (
    cache_control,
    game_type,
)
from Script.Design import handle_premise, map_handle, constant
from Script.Config import game_config

cache: game_type.Cache = cache_control.cache
""" 游戏缓存数据 """


def get_random_name_for_sex(sex_grade: int) -> str:
    """
    按性别随机生成姓名
    Keyword arguments:
    sex_grade -- 性别
    """
    while 1:
        family_random = random.randint(1, constant.family_region_int_list[-1])
        family_region_index = bisect.bisect_left(constant.family_region_int_list, family_random)
        family_region = constant.family_region_int_list[family_region_index]
        family_name = constant.family_region_list[family_region]
        if sex_grade not in {0, 1}:
            sex_grade = random.randint(0, 1)
        if sex_grade:
            name_random = random.randint(1, constant.girls_region_int_list[-1])
            name_region_index = bisect.bisect_left(constant.girls_region_int_list, name_random)
            name_region = constant.girls_region_int_list[name_region_index]
            name = constant.girls_region_list[name_region]
        else:
            name_random = random.randint(1, constant.boys_region_int_list[-2])
            name_region_index = bisect.bisect_left(constant.boys_region_int_list, name_random)
            name_region = constant.boys_region_int_list[name_region_index]
            name = constant.boys_region_list[name_region]
        now_name = f"{family_name}{name}"
        if now_name not in cache.npc_name_data:
            cache.npc_name_data.add(now_name)
            return family_name + name


def get_scene_path_text(scene_path: List[str]) -> str:
    """
    从场景路径获取场景地址描述文本
    例:主教学楼-1F-101室
    Keyword arguments:
    scene_path -- 场景路径
    Return arguments:
    str -- 场景地址描述文本
    """
    map_list = map_handle.get_map_hierarchy_list_for_scene_path(scene_path, [])
    map_list.reverse()
    scene_path_str = map_handle.get_map_system_path_str_for_list(scene_path)
    scene_path_text = ""
    for now_map in map_list:
        now_map_map_system_str = map_handle.get_map_system_path_str_for_list(now_map)
        map_name = cache.map_data[now_map_map_system_str].map_name
        scene_path_text += map_name + "-"
    return scene_path_text + cache.scene_data[scene_path_str].scene_name


def get_map_path_text(map_path: List[str]) -> str:
    """
    从地图路径获取地图地址描述文本
    例:主教学楼-1F
    Keyword arguments:
    map_path -- 地图路径
    Return arguments:
    str -- 地图地址描述文本
    """
    map_list = map_handle.get_map_hierarchy_list_for_scene_path(map_path, [])
    map_list.reverse()
    map_list.append(map_path)
    now_path_text = ""
    for now_map in map_list:
        now_map_map_system_str = map_handle.get_map_system_path_str_for_list(now_map)
        map_name = cache.map_data[now_map_map_system_str].map_name
        now_path_text += map_name + "-"
    return now_path_text.rstrip("-")


def get_value_text(value: float) -> str:
    """
    获取数值显示的文本
    Keyword arguments:
    value -- 数值
    Return arguments:
    str -- 文本显示
    """
    value = round(value, 2)
    company = ["K", "M", "G", "T", "P", "E", "Z", "Y", "B", "N", "D"]
    int_value = int(value)
    value_str = str(int_value)
    if value > 0:
        if len(value_str) < 4:
            value_str = str(value)
        else:
            company_index = int((len(value_str) - 1) / 3)
            if company_index >= len(company):
                company_index = len(company) - 1
            value_str = value_str[: -company_index * 3]
            value_str += company[company_index - 1]
            value_str = "+" + value_str
    else:
        if len(value_str) < 5:
            value_str = str(value)
        else:
            company_index = int((len(value_str) - 2) / 3)
            if company_index >= len(company):
                company_index = len(company) - 1
            value_str = value_str[: -company_index * 3]
            value_str += company[company_index - 1]
    return value_str
