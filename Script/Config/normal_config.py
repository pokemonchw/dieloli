import os
import configparser
from typing import Dict,List
from Script.Core import game_type


config_normal:game_type.NormalConfig = game_type.NormalConfig()
""" 游戏通用配置数据 """


def init_normal_config():
    """ 初始化游戏通用配置数据 """
    ini_config = configparser.ConfigParser()
    ini_config.read("config.ini")
    ini_data = ini_config["game"]
    for k in ini_data.keys():
        try:
            config_normal.__dict__[k] = int(ini_data[k])
        except:
            config_normal.__dict__[k] = ini_data[k]