#!/usr/bin/python3
# -*- coding: UTF-8 -*-
import os
import sys
import time
from types import FunctionType


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    multiprocessing.set_executable(sys.executable)
    current_file_path = os.path.realpath(__file__)
    current_dir = os.path.dirname(current_file_path)
    os.chdir(current_dir)

    from Script.Core import game_path_config
    from Script.Config import normal_config
    from Script.Core import game_type, cache_control

    cache_control.cache = game_type.Cache()
    normal_config.init_normal_config()

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
    save_handle.start_save_write_processing()
    handle_achieve.load_achieve()
    handle_achieve.start_save_achieve_processing()

    game_init.run(start_flow.start_frame)
    main_frame.run()
