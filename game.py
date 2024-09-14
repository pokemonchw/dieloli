#!/usr/bin/python3
# -*- coding: UTF-8 -*-
import os
import sys
import time
from types import FunctionType
import platform
import pickle
import multiprocessing
from Script.Config import normal_config
from Script.Core import game_path_config

normal_config.init_normal_config()


def save_achieve_windows(save_queue: multiprocessing.Queue):
    """
    针对windows的并行成就保存函数
    笔记:由于windows不支持fork机制,数据无法从主进程直接继承,pickle转换数据效率过低且不安全,最后决定使用线程安全的queue来传递数据(稳定性待测试)
    Keyword arguments:
    save_queue -- 传入数据的消息队列
    """
    while 1:
        data = save_queue.get()
        achieve_file_path = os.path.join(game_path_config.SAVE_PATH,"achieve")
        with open(achieve_file_path, "wb+") as f:
            pickle.dump(data, f)


def establish_save_windows(now_save_queue: multiprocessing.Queue, save_path_dir: str):
    """
    针对windows的并行自动存档函数
    笔记:由于windows不支持fork机制,数据无法从主进程直接继承,pickle转换数据效率过低且不安全,最后决定使用线程安全的queue来传递数据(稳定性待测试)
    Keyword arguments:
    save_queue -- 传递存档数据的消息队列
    save_path_dir -- 游戏存档目录
    """
    while 1:
        data = now_save_queue.get()
        save_id = data[0]
        save_data = data[1]
        save_version = data[2]
        save_path = os.path.join(game_path_config.SAVE_PATH, save_id)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        head_file_path = os.path.join(save_path, "0")
        with open(head_file_path, "wb+") as f:
            pickle.dump(save_version, f)
        data_file_path = os.path.join(save_path, "1")
        with open(data_file_path, "wb+") as f:
            pickle.dump(save_data, f)


if __name__ == "__main__":

    multiprocessing.freeze_support()
    multiprocessing.set_executable(sys.executable)
    current_file_path = os.path.realpath(__file__)
    current_dir = os.path.dirname(current_file_path)
    os.chdir(current_dir)

    from Script.Core import game_type, cache_control

    cache_control.cache = game_type.Cache()

    from Script.Core import get_text
    from Script.Config import game_config, name_config

    _: FunctionType = get_text._
    """ 翻译api """

    game_config.init()
    name_config.init_name_data()

    from Script.Config import map_config

    map_config.init_map_data()

    from Script.Design import start_flow, instruct, handle_achieve, debug, achieve, adv
    from Script.Core import game_init, save_handle
    import Script.Premise
    import Script.Settle
    import Script.StateMachine
    import Script.UI.Flow
    from Script.Core import main_frame

    handle_achieve.load_achieve()
    if platform.system() != "Linux":
        achieve_process = multiprocessing.Process(target=save_achieve_windows,args=(handle_achieve.achieve_queue, game_path_config.SAVE_PATH))
        achieve_process.start()
        save_process = multiprocessing.Process(target=establish_save_windows, args=(save_handle.save_queue, game_path_config.SAVE_PATH))
        save_process.start()

    game_init.run(start_flow.start_frame)
    main_frame.run()
