#!/usr/bin/python3
# -*- coding: UTF-8 -*-
from Script.Core import game_type, cache_control
from Script.Config import normal_config

cache_control.cache = game_type.Cache()
normal_config.init_normal_config()


from Script.Config import game_config, name_config
from Script.Core import get_text

get_text.init_translation()
game_config.init()
name_config.init_name_data()


from Script.Config import map_config


map_config.init_map_data()


print("Cache Building End")
