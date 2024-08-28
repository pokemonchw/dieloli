from Script.Core import game_type, cache_control
from Script.Design import handle_premise, constant

cache: game_type.Cache = cache_control.cache
""" 游戏缓存数据 """


@handle_premise.add_premise(constant.Premise.DEBUG_ON)
def handle_debug_on(character_id: int) -> int:
    """
    校验debug模式是否已开启
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    if cache.debug:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.DEBUG_OFF)
def handle_debug_off(character_id: int) -> int:
    """
    校验debug模式是否已关闭
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    if cache.debug:
        return 0
    return 1
