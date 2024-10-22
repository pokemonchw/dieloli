import random
import time
import datetime
import queue
from functools import wraps
from typing import Set, List
from types import FunctionType
from threading import Thread
from Script.Core import cache_control, game_type, get_text
from Script.Design import update, character, attr_calculation, course, game_time, clothing, constant
from Script.UI.Panel import see_character_info_panel, see_save_info_panel
from Script.Config import normal_config, game_config
from Script.UI.Moudle import draw


cache: game_type.Cache = cache_control.cache
""" 游戏缓存数据 """
_: FunctionType = get_text._
""" 翻译api """
width: int = normal_config.config_normal.text_width
""" 屏幕宽度 """
instruct_queue = queue.Queue()
""" 待处理的指令队列 """


def init_instruct_handle_thread():
    """初始化指令处理线程"""
    while 1:
        instruct_queue.get()


instruct_handle_thread = Thread(target=init_instruct_handle_thread)
""" 指令处理线程 """
instruct_handle_thread.start()


def handle_instruct(instruct: int):
    """
    处理执行指令
    Keyword arguments:
    instruct -- 指令id
    """
    instruct_queue.put(instruct)
    if instruct in constant.instruct_premise_data:
        constant.handle_instruct_data[instruct]()


def add_instruct(instruct_id: int, instruct_type: int, name: str, premise_set: Set):
    """
    添加指令处理
    Keyword arguments:
    instruct_id -- 指令id
    instruct_type -- 指令类型
    name -- 指令绘制文本
    premise_set -- 指令所需前提集合
    """

    def decorator(func: FunctionType):
        constant.handle_instruct_data[instruct_id] = func
        constant.instruct_premise_data[instruct_id] = premise_set
        constant.instruct_type_data.setdefault(instruct_type, set())
        constant.instruct_type_data[instruct_type].add(instruct_id)
        constant.handle_instruct_name_data[instruct_id] = name
        constant.instruct_to_type_data[instruct_id] = instruct_type

    return decorator
