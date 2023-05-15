#!/usr/bin/python3
# -*- coding: UTF-8 -*-
import time
from types import FunctionType
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

from Script.Design import start_flow, instruct
from Script.Core import game_init
import Script.Premise
import Script.Settle
import Script.StateMachine
import Script.UI.Flow
from Script.Core import main_frame
import multiprocessing

if __name__ == "__main__":
    multiprocessing.freeze_support()

game_init.run(start_flow.start_frame)
main_frame.start_main_window()
