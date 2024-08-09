import os
import json
import configparser
from Script.Core import game_type, game_path_config


config_normal: game_type.NormalConfig = game_type.NormalConfig()
""" 游戏通用配置数据 """
package_path = os.path.join("package.json")


def init_normal_config():
    """初始化游戏通用配置数据"""
    ini_config = configparser.ConfigParser()
    ini_config.read(game_path_config.CONFIG_PATH, encoding="utf8")
    ini_data = ini_config["game"]
    for k in ini_data.keys():
        try:
            config_normal.__dict__[k] = int(ini_data[k])
        except ValueError:
            config_normal.__dict__[k] = ini_data[k]
    if os.path.exists(package_path):
        with open(package_path, "r") as package_file:
            version_data = json.load(package_file)
            config_normal.verson = "Past." + version_data["version"]
    else:
        config_normal.verson = "Orgin"


def change_normal_config(now_key: str, now_value: str):
    """
    更改游戏通用配置数据
    Keyword arguments:
    now_key -- 更改的键
    now_value -- 更改的值
    """
    ini_config = configparser.ConfigParser()
    ini_config.read(game_path_config.CONFIG_PATH, encoding="utf8")
    ini_config.set("game",now_key,str(now_value))
    with open(game_path_config.CONFIG_PATH,"w",encoding="utf-8") as f:
        ini_config.write(f)
