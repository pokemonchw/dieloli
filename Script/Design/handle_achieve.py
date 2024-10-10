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
    achieve_file_path = os.path.join(game_path_config.SAVE_PATH,"achieve")
    with open(achieve_file_path, "wb+") as f:
        pickle.dump(cache_control.achieve, f)

