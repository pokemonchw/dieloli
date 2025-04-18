import math
import random
import bisect
from types import FunctionType
from typing import List
from Script.Core import (
    cache_control,
    game_type,
    get_text
)
from Script.Design import handle_premise, map_handle, constant
from Script.Config import game_config

cache: game_type.Cache = cache_control.cache
""" 游戏缓存数据 """
_: FunctionType = get_text._
""" 翻译api """


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
        if now_name not in cache.npc_name_data and now_name not in constant.adv_name_set:
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
    units = ["", "K", "M", "G", "T", "P", "E", "Z", "Y", "B", "N", "D"]
    rounded_value = round(value, 2)
    if abs(rounded_value) < 1000:
        return str(rounded_value)
    magnitude = min(int(math.log10(abs(value)) / 3), len(units) - 1)
    scaled_value = rounded_value / (1000 ** magnitude)
    return f"{scaled_value:.2f}{units[magnitude]}"


def get_hungry_text(value: float) -> str:
    """
    获取饥饿值的描述文本
    Keyword argumenst:
    value -- 数值
    Return arguments:
    str -- 描述文本
    """
    if value < 20:
        return _("吃饱了")
    elif value < 40:
        return _("肚子空空")
    elif value < 60:
        return _("感到饥饿")
    elif value < 80:
        return _("快饿晕了")
    else:
        return _("即将饿死")


def get_thirsty_text(value: float) -> str:
    """
    获取口渴值的描述文本
    Keyword argumenst:
    value -- 数值
    Return arguments:
    str -- 描述文本
    """
    if value < 20:
        return _("喝饱了")
    elif value < 40:
        return _("感到口渴")
    elif value < 60:
        return _("开始缺水")
    elif value < 90:
        return _("脱水了")
    else:
        return _("即将渴死")
