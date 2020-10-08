#!/usr/bin/python3
# -*- coding: UTF-8 -*-
import sys
from types import FunctionType
from Script.Config import normal_config,name_config,map_config


normal_config.init_normal_config()
name_config.init_name_data()
map_config.init_map_data()

from Script.Core import get_text
from Script.Config import game_config

_: FunctionType = get_text._
""" 翻译api """

if sys.version_info < (3, 8, 6):
    print(_("python3版本过旧(低于python3.8.6),请升级"))
    exit(0)


game_config.init()


from Script.Design import start_flow, handle_target, handle_premise
from Script.Core import game_init
import Script.Talk
import Script.Settle
import Script.UI.Panel

game_init.run(start_flow.start_frame)
