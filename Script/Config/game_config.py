import os
import configparser
from typing import Dict
from Script.Config import config_def
from Script.Core import game_type,json_handle

data_path = os.path.join("data","data.json")
""" 原始json数据文件路径 """
config_normal = game_type.NormalConfig()
""" 游戏通用配置数据 """
config_data = {}
""" 原始json数据 """
config_font:Dict[int,config_def.FontConfig] = {}
""" 字体配置数据 """
config_font_data:Dict[str,int] = {}
""" 字体名字对应字体id """
config_bar:Dict[int,config_def.BarConfig] = {}
""" 比例条配置数据 """
config_bar_data:Dict[str,int] = {}
""" 比例条名字对应比例条id """
config_chest:Dict[int,config_def.ChestTem] = {}
""" 罩杯配置数据 """
config_attr_tem:Dict[int,config_def.AttrTem] = {}
""" 性别对应角色各项基础属性模板 """
config_age_tem:Dict[int,config_def.AgeTem] = {}
""" 年龄段年龄范围模板 """


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


def load_data_json():
    """ 载入data.json内配置数据 """
    global config_data
    config_data = json_handle.load_json(data_path)


def load_font_data():
    """ 载入字体配置数据 """
    for font_data in config_data["FontConfig"]["data"]:
        now_font = config_def.FontConfig()
        now_font.__dict__ = font_data
        config_font[now_font.cid] = now_font
        config_font_data[now_font.name] = now_font.cid


def load_bar_data():
    """ 载入比例条配置数据 """
    for bar_data in config_data["BarConfig"]["data"]:
        now_bar = config_def.BarConfig()
        now_bar.__dict__ = bar_data
        config_bar[now_bar.cid] = now_bar
        config_bar_data[now_bar.name] = now_bar.cid


def load_chest_tem_data():
    """ 载入罩杯配置数据 """
    for chest_data in config_data["ChestTem"]["data"]:
        now_chest = config_def.ChestTem()
        now_chest.__dict__ = chest_data
        config_chest[now_chest.cid] = now_chest


def load_attr_tem():
    """ 载入性别对应角色各项基础属性模板 """
    for attr_tem in config_data["AttrTem"]["data"]:
        now_tem = config_def.AttrTem()
        now_tem.__dict__ = attr_tem
        config_attr_tem[now_tem.cid] = now_tem


def load_age_tem():
    """ 载入各年龄段对应年龄范围模板 """
    for age_tem in config_data["AgeTem"]["data"]:
        now_tem = config_def.AgeTem()
        now_tem.__dict__ = age_tem
        config_age_tem[now_tem.cid] = now_tem


def init():
    """ 初始化游戏配置数据 """
    load_data_json()
    init_normal_config()
    load_font_data()
    load_bar_data()
    load_chest_tem_data()
    load_attr_tem()
    load_age_tem()
