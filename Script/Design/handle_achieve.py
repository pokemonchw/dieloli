from types import FunctionType
from typing import Dict
import queue
import threading
import pickle
import multiprocessing
import platform
import os
from Script.Core import cache_control, game_path_config, get_text
from Script.Config import game_config, config_def, normal_config
from Script.UI.Moudle import draw


achieve_handler_data: Dict[str, FunctionType] = {}
""" 所有成就的验证器数据 """
_: FunctionType = get_text._
""" 翻译api """
achieve_queue: multiprocessing.Queue = multiprocessing.Queue()
""" 成就数据队列 """

def add_achieve(achieve_id: str):
    """
    添加成就验证器
    Keyword arguments:
    achieve_id -- 成就id
    """

    def decorator(func: FunctionType):
        achieve_handler_data[achieve_id] = func

    return decorator


def check_all_achieve():
    """检查成就是否已完成"""
    for cid in game_config.config_achieve:
        if cid in cache_control.achieve.completed_data and cache_control.achieve.completed_data[cid]:
            continue
        if cid in achieve_handler_data:
            now_judge = achieve_handler_data[cid]()
            if now_judge:
                cache_control.achieve.completed_data[cid] = True
                now_config: config_def.Achieve = game_config.config_achieve[cid]
                now_draw = draw.NormalDraw()
                now_draw.text = _("达成了新的成就") + " -> " + now_config.name + "\n"
                now_draw.width = normal_config.config_normal.text_width
                now_draw.draw()
    save_achieve()


def load_achieve():
    """载入成就数据"""
    achieve_file_path = os.path.join(game_path_config.SAVE_PATH, "achieve")
    if os.path.exists(achieve_file_path):
        with open(achieve_file_path, "rb") as f:
            cache_control.achieve.__dict__.update(pickle.load(f).__dict__)


def save_achieve():
    """保存成就数据"""
    if platform.system() == "Linux":
        now_process = multiprocessing.Process(target=save_achieve_linux)
        now_process.start()
        now_process.join()
    else:
        achieve_queue.put(cache_control.achieve)


def save_achieve_linux():
    """
    针对linux的并行成就保存函数
    笔记:得益于unix的fork机制,子进程直接复制了一份内存,效率高,且不用创建传参管道,数据进程安全,不受玩家操作影响
    """
    achieve_file_path = os.path.join(game_path_config.SAVE_PATH,"achieve")
    with open(achieve_file_path, "wb+") as f:
        pickle.dump(cache_control.achieve, f)

