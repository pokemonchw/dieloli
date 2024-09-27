#!/usr/bin/python3
# -*- coding: UTF-8 -*-
import os
import sys
import time
from types import FunctionType
import platform
import pickle
from Script.Config import normal_config
from Script.Core import game_path_config

normal_config.init_normal_config()


if __name__ == "__main__":
    if getattr(sys, 'frozen', False):
        os.chdir(os.path.dirname(sys.executable))
    else:
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print(os.getcwd())

    from Script.Core import game_type, cache_control

    cache_control.cache = game_type.Cache()

    from Script.Core import get_text

    get_text.rebuild_mo()
    get_text.init_translation()

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

    game_init.run(start_flow.start_frame)
    main_frame.run()
