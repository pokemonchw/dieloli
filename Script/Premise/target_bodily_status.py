import math
from Script.Design import handle_premise
from Script.Core import constant, game_type, cache_control

cache: game_type.Cache = cache_control.cache
""" 游戏缓存数据 """


@handle_premise.add_premise(constant.Premise.TARGET_HYPOSTHENIA)
def handle_target_hyposthenia(character_id: int) -> int:
    """
    校验交互对象是否体力不足
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    now_weight = int((target_data.hit_point_max - target_data.hit_point) / 5)
    now_weight += int((target_data.mana_point_max - target_data.mana_point) / 10)
    return now_weight


@handle_premise.add_premise(constant.Premise.TARGET_PHYSICAL_STRENGHT)
def handle_target_physical_strenght(character_id: int) -> int:
    """
    校验交互对象是否体力充沛
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    now_weight = int((target_data.hit_point_max / 2 - target_data.hit_point) / 5)
    now_weight += int((target_data.mana_point_max / 2 - target_data.mana_point) / 10)
    now_weight = max(now_weight, 0)
    return now_weight


@handle_premise.add_premise(constant.Premise.TARGET_IS_WARM)
def handle_target_is_warm(character_id: int) -> int:
    """
    校验交互对象是否感觉温暖
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    now_warm = 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    for clothing_type in target_data.put_on:
        clothing_id = target_data.put_on[clothing_type]
        if clothing_id is not None and clothing_id != "":
            now_clothing: game_type.Clothing = target_data.clothing[clothing_type][clothing_id]
            now_warm += now_clothing.warm
    return now_warm / 20 * 100


@handle_premise.add_premise(constant.Premise.TARGET_NOT_WARM)
def handle_target_not_warm(character_id: int) -> int:
    """
    校验交互对象是否未感觉温暖
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    now_warm = 0
    for clothing_type in target_data.put_on:
        clothing_id = target_data.put_on[clothing_type]
        if clothing_id is not None and clothing_id != "":
            now_clothing: game_type.Clothing = target_data.clothing[clothing_type][clothing_id]
            now_warm += now_clothing.warm
    if now_warm > 17:
        return 0
    return (17 - now_warm) / 17 * 100


@handle_premise.add_premise(constant.Premise.TARGET_IS_ACHE)
def handle_target_is_ache(character_id: int) -> int:
    """
    校验交互对象是否感到疼痛
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    target_data.status.setdefault(23, 0)
    if target_data.status[23] > 10:
        return math.floor(target_data.status[23]) / 10
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_NOT_ACHE)
def handle_target_not_ache(character_id: int) -> int:
    """
    校验交互对象是否未感到疼痛
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    target_data.status.setdefault(23, 0)
    if target_data.status[23] > 10:
        return 0
    return (10 - target_data.status[23]) * 10


@handle_premise.add_premise(constant.Premise.TARGET_IS_VERTIGO)
def handle_target_is_vetigo(character_id: int) -> int:
    """
    校验交互对象是否感到眩晕
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    target_data.status.setdefault(24, 0)
    if target_data.status[24] > 10:
        return math.floor(target_data.status[24]) / 10
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_NOT_VERTIGO)
def handle_target_not_vertigo(character_id: int) -> int:
    """
    校验交互对象是否未感到眩晕
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    target_data.status.setdefault(24, 0)
    if target_data.status[24] > 10:
        return 0
    return (10 - target_data.status[24]) * 10


@handle_premise.add_premise(constant.Premise.TARGET_IS_TIRED)
def handle_target_is_tired(character_id: int) -> int:
    """
    校验交互对象是否感到疲惫
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    target_data.status.setdefault(25, 0)
    if target_data.status[25] > 10:
        return math.floor(target_data.status[25]) / 10
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_NOT_TIRED)
def handle_target_not_tired(character_id: int) -> int:
    """
    校验交互对象是否未感到疲惫
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    target_data.status.setdefault(25, 0)
    if target_data.status[25] > 10:
        return 0
    return (10 - target_data.status[25]) * 10


@handle_premise.add_premise(constant.Premise.TARGET_IS_INTOXICATED)
def handle_target_is_intoxicated(character_id: int) -> int:
    """
    校验交互对象是否处于迷醉状态
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    target_data.status.setdefault(26, 0)
    if target_data.status[26] > 10:
        return math.floor(target_data.status[26]) / 10
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_NOT_INTOXICATED)
def handle_target_not_intoxicated(character_id: int) -> int:
    """
    校验交互对象是否未处于迷醉状态
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    target_data.status.setdefault(26, 0)
    if target_data.status[26] > 10:
        return 0
    return (10 - target_data.status[26]) * 10


@handle_premise.add_premise(constant.Premise.TARGET_HUNGER)
def handle_target_hunger(character_id: int) -> int:
    """
    校验交互对象是否处于饥饿状态
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    target_data.status.setdefault(27, 0)
    if target_data.status[27] > 15:
        return target_data.status[27] * 10
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_NOT_HUNGER)
def handle_target_not_hunger(character_id: int) -> int:
    """
    校验交互对象是否未处于饥饿状态
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    target_data.status.setdefault(27, 0)
    return 100 - target_data.status[27]


@handle_premise.add_premise(constant.Premise.TARGET_THIRSTY)
def handle_target_thirsty(character_id: int) -> int:
    """
    校验交互对象是否处于口渴状态
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    target_data.status.setdefault(28, 0)
    if target_data.status[28] > 15:
        return target_data.status[28] * 10
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_NOT_THIRSTY)
def handle_target_not_thirsty(character_id: int) -> int:
    """
    校验交互对象是否未处于口渴状态
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    target_data.status.setdefault(28, 0)
    return 100 - target_data.status[28]
