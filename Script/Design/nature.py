import random
from typing import Dict
from Script.Core import constant
from Script.Config import game_config


def get_random_nature() -> Dict[int,Dict[int,int]]:
    """
    随机生成角色性格
    Return arguments:
    Dict[int,Dict[int,int]] -- 角色性格数据 性格分类:性格id:性格数值
    """
    nature_data = {}
    for k in game_config.config_nature:
        nature = game_config.config_nature[k]
        nature_data.setdefault(nature.type,())
        nature_data[nature.type][nature.cid] = random.uniform(0,100)
    return nature_data
