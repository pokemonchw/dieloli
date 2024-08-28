import random
import time
import datetime
import queue
from functools import wraps
from typing import Set, List
from types import FunctionType
from threading import Thread
from Script.Core import cache_control, game_type, get_text, save_handle
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
debug_queue = queue.Queue()
""" 待处理的指令队列 """


def init_debug_handle_thread():
    """初始化debug指令处理线程"""
    while 1:
        debug_queue.get()
        save_handle.establish_save("auto")


debug_handle_thread = Thread(target=init_debug_handle_thread)
""" debug指令处理线程 """
debug_handle_thread.start()


def handle_debug(debug: int):
    """
    处理debug指令
    Keyword arguments:
    debug -- 指令id
    """
    debug_queue.put(debug)
    if debug in constant.debug_premise_data:
        constant.handle_debug_data[debug]()


def add_debug(debug_id: int, debug_type: int, name: str, premise_set):
    """
    添加debug处理
    Keyword arguments:
    debug_id -- debug指令id
    debug_type -- debug指令类型
    name -- debug指令文本
    """

    def decorator(func: FunctionType):
        constant.handle_debug_data[debug_id] = func
        constant.debug_premise_data[debug_id] = premise_set
        constant.debug_type_data.setdefault(debug_type, set())
        constant.debug_type_data[debug_type].add(debug_id)
        constant.handle_debug_name_data[debug_id] = name

    return decorator
