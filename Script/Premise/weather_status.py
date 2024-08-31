import datetime
from typing import List
from Script.Design import handle_premise, game_time, character, constant
from Script.Core import game_type, cache_control
from Script.Config import game_config

cache: game_type.Cache = cache_control.cache
""" 游戏缓存数据 """


@handle_premise.add_premise(constant.Premise.IS_CLOUDY)
def handle_is_cloudy(character_id: int) -> int:
    """
    校验当前的天气是否是多云
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    if cache.weather == 0:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.IS_HEAVY_RAIN)
def handle_is_heavy_rain(character_id: int) -> int:
    """
    校验当前的天气是否是暴雨
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    if cache.weather == 1:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.IS_RAINY)
def handle_is_rainy(character_id: int) -> int:
    """
    校验当前的天气是否是雨天
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    if cache.weather == 2:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.IS_LIGHT_RAIN)
def handle_is_light_rain(character_id: int) -> int:
    """
    校验当前的天气是否是小雨
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    if cache.weather == 3:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.IS_THUNDER_STORM)
def handle_is_thunder_storm(character_id: int) -> int:
    """
    校验当前的天气是否是雷雨
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    if cache.weather == 4:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.IS_SUNNY)
def handle_is_sunny(character_id: int) -> int:
    """
    校验当前的天气是否是晴天
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    if cache.weather == 5:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.IS_SLEET)
def handle_is_sleet(character_id: int) -> int:
    """
    校验当前的天气是否是雨夹雪
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    if cache.weather == 6:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.IS_HEAVY_SNOW)
def handle_is_heavy_snow(character_id: int) -> int:
    """
    校验当前的天气是否是大雪
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    if cache.weather == 7:
        return 1
    return 0
