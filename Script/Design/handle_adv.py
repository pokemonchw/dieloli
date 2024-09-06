from types import FunctionType
from typing import Dict
from Script.Core import game_type, cache_control
from Script.Design import constant

adv_handler_data: Dict[str, FunctionType] = {}
""" 所有剧情npc验证器 """
cache: game_type.Cache = cache_control.cache
""" 游戏缓存数据 """


def add_adv_handler(adv_id: str, adv_name: str):
    """
    添加剧情npc处理器
    Keyword arguments:
    adv_id -- 剧情npc id
    adv_name -- 剧情npc名字
    """

    def decorator(func: FunctionType):
        adv_handler_data[adv_id] = func
        constant.adv_name_data[adv_id] = adv_name
        constant.adv_name_set.add(adv_name)

    return decorator


def handle_all_adv_npc():
    """
    查找所有角色是否符合某个剧情npc的条件，若符合，则将角色替换为剧情npc
    Keyword arguments:
    character_id -- 角色id
    """
    clone_adv_set = set()
    for character_id in cache.character_data:
        for adv_id in adv_handler_data:
            if adv_id in clone_adv_set:
                continue
            if adv_handler_data[adv_id](character_id):
                clone_adv_set.add(adv_id)
                character_data: game_type.Character = cache.character_data[character_id]
                character_data.adv = adv_id
                character_data.name = constant.adv_name_data[adv_id]
