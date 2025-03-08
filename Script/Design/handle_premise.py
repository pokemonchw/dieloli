import math
import datetime
from uuid import UUID
from functools import wraps
from types import FunctionType
from Script.Core import cache_control, game_type
from Script.Design import map_handle, game_time, attr_calculation, character, course, constant
from Script.Config import game_config

cache: game_type.Cache = cache_control.cache
""" 游戏缓存数据 """


def add_premise(premise: int) -> FunctionType:
    """
    添加前提
    Keyword arguments:
    premise -- 前提id
    Return arguments:
    FunctionType -- 前提处理函数对象
    """
    def decoraror(func):
        constant.handle_premise_data[premise] = func

    return decoraror


def handle_premise(premise: int, character_id: int) -> int:
    """
    调用前提id对应的前提处理函数
    Keyword arguments:
    premise -- 前提id
    character_id -- 角色id
    Return arguments:
    int -- 前提权重加成
    """
    try:
        return constant.handle_premise_data[premise](character_id)
    except Exception as e:
        print(e)
        return 0
